import os
import time
import json
import re
import threading
import nltk
from collections import Counter, defaultdict
from functools import lru_cache

import numpy as np
import pandas as pd
import requests
import torch
import spacy
from spacy.cli import download
from transformers import pipeline, AutoModelForSequenceClassification, AutoTokenizer
from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.responses import FileResponse, StreamingResponse


def _get_hf_token():
    """Fetches the Hugging Face Hub token from Colab Secrets when available,
    falling back to the HF_TOKEN environment variable everywhere else. Not
    required for these public models to load, but avoids the "unauthenticated
    requests" rate-limit warning and speeds up downloads."""
    try:
        from google.colab import userdata
        token = userdata.get("HF_TOKEN")
        if token:
            return token
    except Exception:
        pass
    return os.environ.get("HF_TOKEN")  # OK to stay None -- these are public models


HF_TOKEN = _get_hf_token()
from fastapi.middleware.cors import CORSMiddleware

# ==========================================
# CONFIG & MODEL LOADING (once, at process startup)
# ==========================================
API_KEY = os.environ.get("GOOGLE_PLACES_API_KEY")
if not API_KEY:
    raise RuntimeError("GOOGLE_PLACES_API_KEY environment variable is not set.")

PLACES_SEARCH_URL = "https://places.googleapis.com/v1/places:searchText"
PLACES_AUTOCOMPLETE_URL = "https://places.googleapis.com/v1/places:autocomplete"
FIELD_MASK = (
    "places.id,"
    "places.displayName,"
    "places.rating,"
    "places.userRatingCount,"
    "places.location,"
    "places.reviews,"
    "places.priceLevel,"
    "places.priceRange,"
    "places.primaryTypeDisplayName,"
    "places.formattedAddress,"
    "places.shortFormattedAddress,"
    "nextPageToken"
)


# ==========================================
# REQUEST LIMITS
# Google Places bills per request and this runs behind a public URL, so one
# crawler or one stuck key could spend real money. A free-tier billing
# account can't set Google's own quota caps, and spend-cap enforcement
# doesn't cover Places -- so these are the only ceiling that actually
# exists. Deliberately in-memory: a single-instance deployment needs
# nothing more, and counters resetting on restart is the safe direction to
# fail (a restart costs at most one window, never unbounded spend).
# ==========================================
RATE_LIMITS = {
    # bucket -> (max requests, window seconds). Typeahead fires on nearly
    # every keystroke, so it needs a far looser ceiling than a search, which
    # costs several paginated Places calls plus the whole NLP pipeline.
    "suggest": (int(os.environ.get("SUGGEST_PER_MINUTE", "60")), 60),
    "search": (int(os.environ.get("SEARCH_PER_HOUR", "12")), 3600),
}
DAILY_SEARCH_CAP = int(os.environ.get("DAILY_SEARCH_CAP", "1000"))

_rate_hits = defaultdict(list)
_rate_lock = threading.Lock()
_daily_searches = {"day": None, "count": 0}


class DailyLimitReached(RuntimeError):
    """Raised when the whole deployment has spent its Places budget for the day."""


def _client_key(request):
    """Who to count this request against. Behind Spaces -- and most hosts --
    the socket peer is a proxy, so the first X-Forwarded-For hop is the real
    caller. Spoofable, but this guards against cost, not attackers."""
    forwarded = request.headers.get("x-forwarded-for", "")
    if forwarded:
        return forwarded.split(",")[0].strip()
    return request.client.host if request.client else "unknown"


def _prune_rate_state(now):
    """Drop clients whose window has fully elapsed, so a long-running process
    doesn't accumulate an entry per IP forever."""
    for key, hits in list(_rate_hits.items()):
        window = RATE_LIMITS[key[0]][1]
        fresh = [t for t in hits if now - t < window]
        if fresh:
            _rate_hits[key] = fresh
        else:
            del _rate_hits[key]


def _check_rate_limit(bucket, request):
    limit, window = RATE_LIMITS[bucket]
    key = (bucket, _client_key(request))
    now = time.time()
    with _rate_lock:
        if len(_rate_hits) > 1000:
            _prune_rate_state(now)
        hits = [t for t in _rate_hits[key] if now - t < window]
        if len(hits) >= limit:
            _rate_hits[key] = hits
            retry_after = max(1, int(window - (now - hits[0])) + 1)
            raise HTTPException(
                status_code=429,
                detail="You're going a bit fast -- give it a moment and try again.",
                headers={"Retry-After": str(retry_after)},
            )
        hits.append(now)
        _rate_hits[key] = hits


def _count_search_against_daily_cap():
    """Counted where Google is actually called, not at the endpoint, so a
    cache hit costs nothing and doesn't eat into the day's budget."""
    today = time.strftime("%Y-%m-%d", time.gmtime())
    with _rate_lock:
        if _daily_searches["day"] != today:
            _daily_searches["day"] = today
            _daily_searches["count"] = 0
        if _daily_searches["count"] >= DAILY_SEARCH_CAP:
            raise DailyLimitReached(
                "This demo has hit its daily search limit. It resets at midnight UTC -- "
                "please come back tomorrow."
            )
        _daily_searches["count"] += 1


def load_spacy_model():
    try:
        return spacy.load("en_core_web_sm")
    except OSError:
        download("en_core_web_sm")
        return spacy.load("en_core_web_sm")


nlp = load_spacy_model()

# Auto-detects GPU when available, falls back to CPU (this is a CPU deployment target).
_device = 0 if torch.cuda.is_available() else -1
sentiment_analyzer = pipeline(
    "sentiment-analysis",
    model="distilbert-base-uncased-finetuned-sst-2-english",
    device=_device,
    token=HF_TOKEN,
    batch_size=16,
)

# Zero-shot aspect classifier (food/service/price/ambiance), replacing fixed
# keyword lists so paraphrases are still picked up. Loaded once at startup.
# Uses a small distilbert-based MNLI model (much faster than the original
# distilbart-mnli-12-3 on CPU) and, when running on CPU, dynamic int8
# quantization on top of that for a further 2-4x speedup with only a
# small, usually negligible accuracy cost -- this was the single biggest
# latency contributor in the pipeline (~2:40/search before this change).
ASPECT_LABELS = ["food quality", "service quality", "price or value", "ambiance"]
ASPECT_CONFIDENCE_THRESHOLD = 0.5
ASPECT_MODEL_NAME = "typeform/distilbert-base-uncased-mnli"

_aspect_tokenizer = AutoTokenizer.from_pretrained(ASPECT_MODEL_NAME, token=HF_TOKEN)
_aspect_model = AutoModelForSequenceClassification.from_pretrained(ASPECT_MODEL_NAME, token=HF_TOKEN)

if _device == -1:
    # Not every platform ships a working PyTorch quantization engine (notably
    # Apple Silicon Macs often lack both fbgemm and qnnpack in the standard
    # CPU build) -- quantization is a bonus speedup, not a requirement, so
    # fall back to the unquantized (but still much smaller than the original)
    # model rather than crash the server on unsupported platforms.
    try:
        _aspect_model = torch.quantization.quantize_dynamic(
            _aspect_model, {torch.nn.Linear}, dtype=torch.qint8
        )
    except RuntimeError as exc:
        print(f"[startup] Skipping CPU quantization (unsupported on this platform): {exc}")

aspect_classifier = pipeline(
    "zero-shot-classification",
    model=_aspect_model,
    tokenizer=_aspect_tokenizer,
    device=_device,
    batch_size=16,
)


# ==========================================
# CORE PIPELINE (fetch -> sentiment -> scoring -> insights)
# Kept in lockstep with the notebook cells this logic was validated in.
# ==========================================
def _fetch_places_pages(query_string, max_pages=1, page_delay_seconds=2.0, min_rating=3.5):
    headers = {
        "Content-Type": "application/json",
        "X-Goog-Api-Key": API_KEY,
        "X-Goog-FieldMask": FIELD_MASK,
    }
    all_places = []
    page_token = None
    for _ in range(max_pages):
        payload = {"textQuery": query_string, "languageCode": "en", "regionCode": "AE"}
        # Experimental: asks Google itself to exclude low-rated places from
        # what it returns (a filter, not a sort -- rankPreference still only
        # supports RELEVANCE/DISTANCE). Untested against the live API from
        # here; if Google rejects the field name, this call will raise
        # RuntimeError below with the API's own error message, making it
        # obvious to remove/adjust.
        if min_rating is not None:
            payload["minRating"] = min_rating
        if page_token:
            payload["pageToken"] = page_token
        try:
            response = requests.post(PLACES_SEARCH_URL, headers=headers, json=payload, timeout=15)
            response.raise_for_status()
        except requests.exceptions.RequestException as exc:
            raise RuntimeError(f"Google Places API request failed: {exc}") from exc
        data = response.json()
        all_places.extend(data.get("places", []))
        page_token = data.get("nextPageToken")
        if not page_token:
            break
        time.sleep(page_delay_seconds)
    return all_places


# Rough bounding rectangle around the Dubai emirate, used to restrict
# area-suggest autocomplete to Dubai. Deliberately a bit generous (not a
# tight fit) so real Dubai neighborhoods near the emirate's edges aren't
# accidentally excluded -- the substring check below is the real filter.
DUBAI_BOUNDS = {
    "low": {"latitude": 24.70, "longitude": 54.85},
    "high": {"latitude": 25.40, "longitude": 55.60},
}

# The six other emirates -- a suggestion naming one of these is rejected
# even if it falls inside DUBAI_BOUNDS (the box isn't a perfect fit to
# Dubai's actual irregular border).
OTHER_EMIRATES = [
    "abu dhabi", "sharjah", "ajman", "umm al quwain", "ras al khaimah", "fujairah",
]


# The typeahead fires on nearly every keystroke, and typing/backspacing
# revisits the same prefixes constantly -- serving those from memory keeps
# the dropdown instant instead of paying a Google round trip each time.
_area_suggest_cache = {}
_AREA_SUGGEST_TTL = 30 * 60


def _autocomplete_dubai(query):
    """One Dubai-scoped Places Autocomplete (New) call, cached. Never raises
    -- a request failure or unexpected response shape yields no suggestions.

    Deliberately does NOT filter by place type. Restricting to
    locality/sublocality/neighborhood dropped real Dubai communities:
    a developer-built residential area like Azizi Riviera isn't typed as
    any of those, so it never surfaced. Dubai-scoping is left to the
    geographic box plus the other-emirate check below.
    """
    cache_key = query.lower()
    cached = _area_suggest_cache.get(cache_key)
    if cached and cached[1] > time.time():
        return cached[0]

    headers = {"Content-Type": "application/json", "X-Goog-Api-Key": API_KEY}
    payload = {
        "input": query,
        "includedRegionCodes": ["ae"],
        "locationRestriction": {"rectangle": DUBAI_BOUNDS},
    }
    try:
        response = requests.post(PLACES_AUTOCOMPLETE_URL, headers=headers, json=payload, timeout=10)
        response.raise_for_status()
        data = response.json()
    except (requests.exceptions.RequestException, ValueError) as exc:
        print(f"[area-suggest] request failed: {exc}")
        return []

    # Each suggestion carries the canonical string plus Google's own split of
    # it. "text" stays exactly what it was -- it is what gets submitted and
    # searched, and the mirdiff fix depends on that being the full formatted
    # name. "main" and "secondary" only change how the dropdown is drawn:
    # "Dubai Marina" on one line, "Dubai - United Arab Emirates" muted beneath,
    # instead of one long string wrapping to three lines.
    #
    # structuredFormat is documented but treated as optional here: if it is
    # ever missing the entry still has "text", and the frontend falls back to
    # splitting that itself.
    suggestions = []
    for item in data.get("suggestions", []):
        prediction = item.get("placePrediction", {})
        text = prediction.get("text", {}).get("text")
        if not text or any(emirate in text.lower() for emirate in OTHER_EMIRATES):
            continue
        structured = prediction.get("structuredFormat") or {}
        suggestions.append({
            "text": text,
            "main": (structured.get("mainText") or {}).get("text") or "",
            "secondary": (structured.get("secondaryText") or {}).get("text") or "",
        })
    _area_suggest_cache[cache_key] = (suggestions, time.time() + _AREA_SUGGEST_TTL)
    return suggestions


def suggest_dubai_areas(query):
    """Dubai-only typeahead suggestions for a partially typed area.

    When the full text matches nothing, drops a character at a time and
    retries, so a typo or an out-of-scope place ("lahore") still comes back
    with the nearest Dubai areas to pick from rather than an empty dropdown
    and a dead end.
    """
    if not query or not query.strip():
        return []
    cleaned = query.strip()
    while len(cleaned) >= 2:
        matches = _autocomplete_dubai(cleaned)
        if matches:
            return matches
        cleaned = cleaned[:-1]
    return []


def _clean_cuisine_label(raw):
    if not raw:
        return "General Dining"
    cleaned = re.sub(r"\s*restaurants?\s*$", "", raw, flags=re.IGNORECASE).strip()
    return cleaned or raw


# When the user named a cuisine, that is what the column reports -- for every
# row, not just the ones Google labelled vaguely. Google's own label describes
# the venue's primary type, which for an Italian search can come back as
# "Pizza", "Family Restaurant" or "Fine Dining": three different words for
# results the user asked for as Italian, in a column they expect to read
# Italian throughout. The label is Google's answer to a different question.
#
# This only applies to a filtered search. With no cuisine given, Google's label
# is the only signal there is and is used as-is -- which is also the case where
# the Cuisine Mix chart is shown, so nothing there collapses to one slice.
def _resolve_cuisine_label(cuisine_label, searched_cuisine):
    if searched_cuisine and searched_cuisine.strip():
        return searched_cuisine.strip().title()
    return cuisine_label


# Tried strictest first -- a dense, popular area may have 20+ places at 4.8,
# but most area/cuisine combos won't, so this usually cascades down. Each
# step is one extra sequential Places API call, so this list is a direct
# latency-vs-coverage tradeoff, not free: a 5.0 tier was tried here briefly,
# but a real average of exactly 5.0 across the 20+ reviews needed to clear
# top_n is essentially never real data -- it added a whole extra round trip
# to every single search for ~0% success rate, and was dropped for that
# reason. 4.9 is kept for the rare hyper-premium cluster (e.g. a
# Downtown/Marina fine-dining pocket) that might still clear it.
RATING_CASCADE = [4.9, 4.8, 4.5, 4.0, 3.5, None]


# Text Search ranks by relevance, not location, and this call carries no
# locationRestriction -- we hold the area as text, not coordinates. So a search
# for Dubai Marina can return a famous Downtown venue purely because it matches
# the words well. Filter what comes back against the area's own name.
#
# "Dubai" and the country words are stripped because every address in the
# emirate contains them; what is left is the part that actually distinguishes
# one neighbourhood from another ("marina", "barsha", "mirdif").
AREA_STOPWORDS = {"dubai", "united", "arab", "emirates", "uae", "the", "area", "and"}


def _area_match_tokens(area):
    head = re.split(r"\s+[-\u2013]\s+|,", area or "")[0]
    return {
        token for token in re.findall(r"[a-z]+", head.lower())
        if token not in AREA_STOPWORDS and len(token) > 2
    }


def _filter_places_to_area(places, area):
    """Keep only results whose address names the area that was searched.

    Deliberately an address test and not a geographic one. Proximity was tried
    and reverted: Dubai's communities sit too close together for distance to
    stand in for membership. From Dubai Design District, a 4km radius takes in
    Downtown at 2.5km, Business Bay at 3.1km and DIFC at 3.2km -- three
    separate communities, and tightening the radius below that would be too
    small for a larger one. Dubai Marina only made the idea look workable
    because it is unusually isolated, with Downtown 18.7km away.

    The cost is that a venue whose formattedAddress is just "Dubai - United
    Arab Emirates" is dropped even when it sits in the area. That is the
    accepted trade: a shorter list of results that are certainly in the area
    beats a longer one salted with the community next door.
    """
    tokens = _area_match_tokens(area)
    if not tokens:
        # Nothing distinctive to match on -- a bare "Dubai", say. Filtering on
        # no signal would drop everything.
        return places
    kept = [
        place for place in places
        if any(
            token in (place.get("formattedAddress") or place.get("shortFormattedAddress") or "").lower()
            for token in tokens
        )
    ]
    if not kept:
        # Addresses in this area evidently do not repeat its name. Relevance
        # -ranked results beat an empty page, so keep them and say so.
        print(f"[area-filter] nothing addressed in {sorted(tokens)}; keeping all {len(places)}")
        return places
    print(f"[area-filter] {sorted(tokens)}: kept {len(kept)} of {len(places)}, dropped {len(places) - len(kept)}")
    return kept


def _fetch_best_rated_places(query_string, max_pages, top_n, area=""):
    places = []
    for threshold in RATING_CASCADE:
        places = _filter_places_to_area(
            _fetch_places_pages(query_string, max_pages=max_pages, min_rating=threshold),
            area,
        )
        # Counted after filtering, so the cascade widens the rating threshold
        # until there are enough results IN THE AREA, rather than enough
        # anywhere in Dubai.
        if len(places) >= top_n:
            break
    return places


def get_reviews_for_area(area, cuisine="", max_budget=None, max_pages=3, top_n=20):
    if not area or not area.strip():
        raise ValueError("area is required (e.g. 'Dubai Marina').")

    _count_search_against_daily_cap()

    query_string = f"{cuisine} restaurants in {area}".strip()
    places = _fetch_best_rated_places(query_string, max_pages=max_pages, top_n=top_n, area=area)

    if not places:
        empty_summary = pd.DataFrame([{
            "total_restaurants": 0,
            "total_reviews": 0,
            "exact_price_count": 0,
            "estimated_price_count": 0,
            "transparency_note": f"No restaurants found for '{query_string}'. Try a different area, cuisine, or budget.",
        }])
        return pd.DataFrame(), empty_summary

    explicit_prices = [
        float(p["priceRange"]["startPrice"]["units"])
        for p in places
        if p.get("priceRange")
        and p["priceRange"].get("startPrice", {}).get("units")
        and p["priceRange"].get("startPrice", {}).get("currencyCode") == "AED"
    ]

    if len(explicit_prices) >= 4:
        q1, q2, q3 = np.percentile(explicit_prices, [25, 50, 75])
        min_p = min(explicit_prices)
    else:
        min_p, q1, q2, q3 = 25.0, 60.0, 150.0, 350.0

    quartile_map = {
        "PRICE_LEVEL_INEXPENSIVE": {"min": min_p, "max": q1, "label": f"AED {int(min_p)} - {int(q1)}"},
        "PRICE_LEVEL_MODERATE": {"min": q1, "max": q2, "label": f"AED {int(q1)} - {int(q2)}"},
        "PRICE_LEVEL_EXPENSIVE": {"min": q2, "max": q3, "label": f"AED {int(q2)} - {int(q3)}"},
        # Open-ended top bracket -- Google reports no upper number here, so
        # 1.5x the 75th percentile is a rough stand-in for plotting purposes.
        "PRICE_LEVEL_VERY_EXPENSIVE": {"min": q3, "max": q3 * 1.5, "label": f"AED {int(q3)}+"},
    }
    UNKNOWN_PRICE = {"min": q2, "max": q2, "label": f"AED {int(q2)}"}

    # Pass 1: compute each place's price info and apply the budget filter
    # across the FULL fetched candidate pool (not just whichever page they
    # landed on), so narrowing to the highest-rated spots next isn't skewed
    # by restaurants that wouldn't have fit the budget anyway.
    candidates = []
    for place in places:
        place_id = place.get("id")
        restaurant_name = place.get("displayName", {}).get("text")
        if not place_id or not restaurant_name:
            continue

        p_range = place.get("priceRange")
        if (p_range
            and p_range.get("startPrice", {}).get("units")
            and p_range.get("startPrice", {}).get("currencyCode") == "AED"):
            start = float(p_range["startPrice"]["units"])
            end = p_range.get("endPrice", {}).get("units", "")
            min_price = start
            max_price = float(end) if end else start * 1.5
            price_display = f"AED {int(start)} - {int(end)}" if end else f"AED {int(start)}+"
            price_source = "Verified Google Price"
            is_exact = True
        else:
            raw_level = place.get("priceLevel")
            meta = quartile_map.get(raw_level, UNKNOWN_PRICE)
            min_price = meta["min"]
            max_price = meta["max"]
            price_display = meta["label"]
            price_source = "Area Quartile Estimate" if raw_level in quartile_map else "Unknown (Area Median Used)"
            is_exact = False

        if max_budget is not None and min_price > max_budget:
            continue

        candidates.append({
            "place": place,
            "place_id": place_id,
            "restaurant_name": restaurant_name,
            "min_price": min_price,
            "max_price": max_price,
            "price_display": price_display,
            "price_source": price_source,
            "is_exact": is_exact,
        })

    # Keep only the highest-rated candidates. Google's Text Search only
    # supports rankPreference RELEVANCE or DISTANCE -- there is no native
    # "sort by rating" -- so without this step we'd analyze whichever
    # restaurants happened to rank first for the search text, not
    # necessarily the best ones in the area.
    candidates.sort(
        key=lambda c: (c["place"].get("rating", 0) or 0, c["place"].get("userRatingCount", 0) or 0),
        reverse=True,
    )

    parsed_reviews = []
    exact_price_count = 0
    estimated_price_count = 0
    analyzed_restaurants = set()

    for c in candidates:
        if len(analyzed_restaurants) >= top_n:
            break

        place = c["place"]
        cuisine_type = _clean_cuisine_label(place.get("primaryTypeDisplayName", {}).get("text", "General Dining"))
        cuisine_type = _resolve_cuisine_label(cuisine_type, cuisine)
        location = place.get("location", {})
        lat = location.get("latitude")
        lng = location.get("longitude")
        user_rating_count = place.get("userRatingCount", 0)
        address = place.get("formattedAddress") or place.get("shortFormattedAddress", "Dubai, UAE")

        place_reviews = []
        for review in place.get("reviews", []):
            text = review.get("text", {}).get("text")
            publish_time = review.get("publishTime", "")
            if text:
                place_reviews.append({
                    "place_id": c["place_id"],
                    "restaurant_name": c["restaurant_name"],
                    "cuisine": cuisine_type,
                    "price_range": c["price_display"],
                    "price_numeric": c["min_price"],
                    "price_numeric_max": c["max_price"],
                    "price_source": c["price_source"],
                    "review_rating": review.get("rating"),
                    "google_rating": place.get("rating"),
                    "review_text": text,
                    "publish_time": publish_time,
                    "user_rating_count": user_rating_count,
                    "latitude": lat,
                    "longitude": lng,
                    "formatted_address": address,
                })

        if not place_reviews:
            # No usable review text -- skip without counting it, so a
            # lower-rated candidate backfills the slot instead.
            continue

        analyzed_restaurants.add(c["place_id"])
        if c["is_exact"]:
            exact_price_count += 1
        else:
            estimated_price_count += 1
        parsed_reviews.extend(place_reviews)

    df_reviews = pd.DataFrame(parsed_reviews)
    venue_count = len(analyzed_restaurants)
    df_summary = pd.DataFrame([{
        "total_restaurants": venue_count,
        "total_reviews": len(df_reviews),
        "exact_price_count": exact_price_count,
        "estimated_price_count": estimated_price_count,
        "transparency_note": f"Found {len(places)} restaurants in the area. Analyzed the top "
                             f"{venue_count} venue{'s' if venue_count != 1 else ''} by Google rating, across {len(df_reviews)} reviews. "
                             f"{exact_price_count} using direct menu prices, "
                             f"{estimated_price_count} estimated via local area quartiles.",
    }])
    return df_reviews, df_summary


def analyze_sentiment(df_reviews):
    if df_reviews.empty or "review_text" not in df_reviews.columns:
        return df_reviews
    texts = df_reviews["review_text"].tolist()
    predictions = sentiment_analyzer(texts, truncation=True, max_length=512)
    df_reviews["sentiment_label"] = [pred["label"] for pred in predictions]
    df_reviews["sentiment_score"] = [round(pred["score"], 4) for pred in predictions]
    return df_reviews


def compute_advanced_metrics(df_reviews, df_aspects=None):
    if df_reviews.empty:
        return pd.DataFrame()

    df_reviews = df_reviews.copy()
    df_reviews["publish_time"] = pd.to_datetime(df_reviews["publish_time"], errors="coerce")
    top5_reviews = (
        df_reviews.sort_values(by=["place_id", "publish_time"], ascending=[True, False])
        .groupby("place_id")
        .head(5)
    )

    venues = top5_reviews.groupby("place_id", as_index=False).first()
    venues = venues.rename(columns={"user_rating_count": "total_google_ratings"})

    _sampled_avg_rating = venues["place_id"].map(
        top5_reviews.groupby("place_id")["review_rating"].mean()
    )
    if "google_rating" in venues.columns:
        venues["avg_google_rating"] = venues["google_rating"].fillna(_sampled_avg_rating).round(1)
    else:
        venues["avg_google_rating"] = _sampled_avg_rating.round(1)

    if "sentiment_label" in top5_reviews.columns:
        venues["positive_sentiment_pct"] = venues["place_id"].map(
            top5_reviews.groupby("place_id")["sentiment_label"].apply(
                lambda s: (s == "POSITIVE").mean() * 100.0
            )
        ).round(1)
    else:
        venues["positive_sentiment_pct"] = venues["place_id"].map(
            top5_reviews.groupby("place_id")["review_rating"].apply(
                lambda r: (r >= 4).mean() * 100.0
            )
        ).round(1)

    venues["short_formatted_address"] = venues["formatted_address"].apply(
        lambda addr: str(addr).split("-")[0].strip()
    )

    venues["volume_confidence_score"] = venues["total_google_ratings"].apply(
        lambda x: min(100.0, (np.log10(x + 1) / np.log10(2500.0)) * 100.0)
    )

    venues["avg_google_rating_scaled"] = (venues["avg_google_rating"] / 5.0) * 100.0
    venues["sentiment_momentum_score"] = np.clip(
        50.0 + (venues["positive_sentiment_pct"] - venues["avg_google_rating_scaled"]),
        0.0, 100.0
    )

    if df_aspects is not None and not df_aspects.empty:
        fs_hits = df_aspects[df_aspects["aspect"].isin(["food quality", "service quality"])]
        aspect_scores = (fs_hits.groupby("place_id")["is_positive"].mean() * 100.0).to_dict()
    else:
        aspect_scores = {}

    venues["aspect_sentiment_score"] = venues["place_id"].map(aspect_scores)
    venues["aspect_sentiment_score"] = venues["aspect_sentiment_score"].fillna(venues["positive_sentiment_pct"])

    risk_scores = {}
    for place_id, group in top5_reviews.groupby("place_id"):
        high_risk = group.apply(
            lambda r: (r.get("review_rating", 5) <= 2) or (r.get("sentiment_label") == "NEGATIVE" and r.get("sentiment_score", 0.0) >= 0.85),
            axis=1
        )
        risk_scores[place_id] = float((high_risk.mean()) * 100.0)

    venues["negativity_risk_score"] = venues["place_id"].map(risk_scores).fillna(0.0)

    if df_aspects is not None and not df_aspects.empty:
        value_hits = df_aspects[df_aspects["aspect"] == "price or value"]
        value_scores = (value_hits.groupby("place_id")["is_positive"].mean() * 100.0).to_dict()
    else:
        value_scores = {}

    venues["value_keyword_score"] = venues["place_id"].map(value_scores).fillna(50.0)
    venues["price_factor"] = venues["price_numeric"].apply(lambda p: max(50.0, 100.0 - ((p / 500.0) * 50.0)))
    venues["price_value_score"] = (
        (0.50 * venues["positive_sentiment_pct"]) +
        (0.30 * venues["value_keyword_score"]) +
        (0.20 * venues["price_factor"])
    )

    raw_model_score = (
        (0.30 * venues["positive_sentiment_pct"]) +
        (0.25 * venues["avg_google_rating_scaled"]) +
        (0.15 * venues["volume_confidence_score"]) +
        (0.10 * venues["sentiment_momentum_score"]) +
        (0.10 * venues["aspect_sentiment_score"]) +
        (0.10 * venues["price_value_score"]) -
        (0.10 * venues["negativity_risk_score"])
    )

    tie_breaker = (venues["total_google_ratings"] % 97) * 0.001
    venues["model_score"] = np.clip(raw_model_score + tie_breaker, 0.0, 100.0).round(2)

    return venues.sort_values(by="model_score", ascending=False).reset_index(drop=True)


# What the seed verbs below expand to, captured from a working WordNet
# corpus. The seeds and their sense indices are fixed, so this set is
# deterministic -- which makes it a safe fallback for when the corpus can't
# be loaded at all (a fresh machine, no network, or macOS shipping Python
# without CA certificates). Previously each seed fell back to just itself,
# so the set quietly collapsed from 7 verbs to 3 with nothing in the logs
# to say so -- and losing "try" measurably hurts dish extraction, since
# reviews say "we tried the lamb ouzi" far more often than "we tasted" it.
CONSUMPTION_VERB_FALLBACK = {
    "order", "sample", "taste", "try", "advocate", "recommend", "urge",
}

# How reviews are actually written. Seeding WordNet with order/taste/recommend
# gave {advocate, order, recommend, sample, taste, try, urge} -- two verbs
# nobody has ever applied to food, and nothing covering "we HAD the biryani",
# which is the most common way a dish gets named. Measured against nine
# realistic phrasings, that set caught two; the parser had already found the
# right noun phrase in most of the other seven ("Mezze Platter", "Seafood
# Risotto") and simply never looked, because the verb was not listed.
#
# Casting this wide is safe because precision comes from the food filter
# downstream, not from the verb: "we had a great time" yields "Time" and "we
# got the bill" yields "Bill", and _phrase_could_be_food rejects both. Broad
# recall here, precision there.
CORE_CONSUMPTION_VERBS = {
    "eat", "have", "get", "share", "enjoy", "love", "serve", "devour",
    "savor", "savour", "finish", "grab", "order", "try", "taste", "recommend",
}


def _wordnet_verb_synonyms(seed_verb, sense_index):
    """Synonyms for one sense of a seed verb, or None if WordNet is unusable."""
    try:
        from nltk.corpus import wordnet as wn
        try:
            synsets = wn.synsets(seed_verb, pos=wn.VERB)
        except LookupError:
            nltk.download("wordnet", quiet=True)
            synsets = wn.synsets(seed_verb, pos=wn.VERB)
        if synsets and sense_index < len(synsets):
            return {lemma.split("_")[0].lower() for lemma in synsets[sense_index].lemma_names()}
    except Exception:
        pass
    return None


def _build_consumption_verbs():
    expanded = [
        _wordnet_verb_synonyms("order", 1),
        _wordnet_verb_synonyms("taste", 2),
        _wordnet_verb_synonyms("recommend", 0),
    ]
    if any(group is None for group in expanded):
        print("[wordnet] corpus unavailable -- using the stored consumption-verb set")
        return set(CONSUMPTION_VERB_FALLBACK) | CORE_CONSUMPTION_VERBS
    return set().union(*expanded) | CORE_CONSUMPTION_VERBS


CONSUMPTION_VERBS = _build_consumption_verbs()
# PERSON is deliberately absent. en_core_web_sm is a small model and labels
# unfamiliar words by shape, so it intermittently tags dish names as people --
# "biryani" in "we had the chicken biryani", "knafeh" in "we ordered the
# knafeh" -- and the chunk was then thrown away. That misfires precisely on
# the transliterated local dishes this is meant to surface, while a person's
# name as the object of "we ate/ordered/tried" is vanishingly rare. The place
# and organisation labels stay: those reject real confusions like a venue or
# a chain name being read as a dish.
NON_FOOD_ENTITY_LABELS = {"GPE", "LOC", "ORG", "NORP", "FAC"}
GENERIC_FALLBACK_WORDS = {
    "place", "restaurant", "experience", "food", "menu", "price", "view",
    "service", "staff", "detail", "attention", "presentation", "wait",
    "crowd", "night", "dinner", "lunch", "visit", "quality", "thing",
    "ambiance", "atmosphere", "value", "way", "time", "portion", "spot"
}
DEFAULT_VIBE_FREQUENCIES = {
    "Welcoming": 1, "Cozy": 1, "Lively": 1, "Warm": 1,
    "Friendly": 1, "Excellent": 1, "Great": 1, "Elegant": 1
}


def _entity_token_spans(doc):
    spans = set()
    for ent in doc.ents:
        if ent.label_ in NON_FOOD_ENTITY_LABELS:
            spans.update(range(ent.start, ent.end))
    return spans


def _phrase_for_token(token, doc, entity_spans):
    for chunk in doc.noun_chunks:
        if chunk.start <= token.i < chunk.end:
            if entity_spans & set(range(chunk.start, chunk.end)):
                return None
            words = [
                t.text.lower() for t in chunk
                if t.pos_ in ("NOUN", "PROPN") and not t.is_stop and t.is_alpha
            ]
            # A word spaCy has never seen often lands as an appositive of the
            # word before it rather than inside the same noun chunk, and it
            # gets its own chunk: "the lamb ouzi" parses as ["the lamb"] +
            # ["ouzi"], so the dish came back as "Lamb". That failure targets
            # exactly the transliterated local dishes this is meant to surface,
            # since those are the words the parser does not know. Pull any
            # appositive that directly follows the chunk back into the phrase.
            for child in token.children:
                if (child.dep_ == "appos" and child.i >= chunk.end
                        and child.i <= chunk.end + 1
                        and child.pos_ in ("NOUN", "PROPN")
                        and child.is_alpha and not child.is_stop):
                    words.append(child.text.lower())
            if words and len(words) <= 3:
                return " ".join(words).title()
            return None

    # No noun chunk covers the object at all. spaCy produces chunks for "I"
    # alone in "I enjoyed the lamb chops", so the dish was simply lost to a
    # parse gap. Rebuild the phrase from the token and its own modifiers.
    if token.i in entity_spans:
        return None
    parts = [
        child for child in token.children
        if child.dep_ in ("compound", "appos")
        and child.pos_ in ("NOUN", "PROPN") and child.is_alpha and not child.is_stop
    ]
    words = [
        t.text.lower() for t in sorted(parts + [token], key=lambda t: t.i)
        if t.pos_ in ("NOUN", "PROPN") and not t.is_stop and t.is_alpha
    ]
    if words and len(words) <= 3:
        return " ".join(words).title()
    return None


# WordNet hypernyms that mean "this is edible". Used as a NEGATIVE test: a
# candidate is rejected only when WordNet knows the word and none of its
# senses descend from one of these. A positive "is this food?" test would
# throw away machboos, knafeh and ouzi -- the local dishes this app exists to
# surface -- because WordNet has never heard of them. Here, unknown to
# WordNet counts as evidence FOR a dish name, not against.
FOOD_HYPERNYM_ROOTS = {
    "food.n.01", "food.n.02", "dish.n.02", "beverage.n.01",
    "helping.n.01", "produce.n.01", "meat.n.01", "baked_goods.n.01",
}

# WordNet files these under food, correctly -- a meal IS food. They are still
# useless as the answer to "what is this place known for", so they are rejected
# by name. Everything else is handled by the descendant rule below.
GENERIC_FOOD_WORDS = {
    "meal", "meals", "course", "courses", "plate", "plates", "platter",
    "platters", "menu", "menus", "breakfast", "lunch", "dinner", "brunch",
    "starter", "starters", "main", "mains", "side", "sides",
}


@lru_cache(maxsize=4096)
def _word_could_be_food(word):
    """False only when WordNet is confident this word is never edible.

    "spot", "care" and "experience" are all grammatically valid objects of
    "we tried ..." and were being served as famous dishes; WordNet knows all
    three and knows none of them as food.
    """
    try:
        from nltk.corpus import wordnet as wn
        synsets = wn.synsets(word, pos=wn.NOUN)
    except Exception:
        return True  # no corpus -- don't filter on a signal we can't read
    if not synsets:
        # Not a noun WordNet knows. If it knows the word at all, it's a
        # describing word spaCy tagged as a noun in context ("unlimited",
        # "delicious") -- not a dish. A word WordNet has never seen is the
        # case we're protecting: a transliterated local dish name.
        return not wn.synsets(word)
    if word in GENERIC_FOOD_WORDS:
        return False
    # A STRICT descendant of a food root, not a root itself. "Portions" was
    # reaching the dish slot because it resolves to helping.n.01 -- which is
    # one of the roots -- so the old "is a root anywhere in the path" test
    # matched it against itself. The same held for "portion", "serving",
    # "helping", "dish", "food" and "cuisine": every one of them names the
    # category rather than anything in it. Requiring a proper ancestor keeps
    # pizza, biryani, gravy, mocha and every other real dish, since those sit
    # underneath a root rather than on it.
    for syn in synsets:
        if syn.name() in FOOD_HYPERNYM_ROOTS:
            continue
        for path in syn.hypernym_paths():
            if any(h.name() in FOOD_HYPERNYM_ROOTS for h in path[:-1]):
                return True
    return False


def _phrase_could_be_food(phrase):
    """One edible word carries the phrase: "chicken poke bowl" is a dish even
    though WordNet files "bowl" as a container."""
    return any(_word_could_be_food(w) for w in phrase.lower().split())


def extract_dish_and_vibe(texts):
    if not texts:
        return "Chef Special", "Welcoming, Cozy, Lively, Warm, Friendly, Excellent, Great, Elegant", dict(DEFAULT_VIBE_FREQUENCIES)

    doc = nlp(" ".join(texts))

    vibe_counts = Counter(
        token.lemma_.title() for token in doc
        if token.pos_ == "ADJ" and not token.is_stop and token.is_alpha and len(token.text) > 2
    )
    top_vibes = [v[0] for v in vibe_counts.most_common(10)]
    vibe_check = ", ".join(top_vibes) if top_vibes else "Welcoming, Cozy, Lively, Warm, Friendly, Excellent, Great, Elegant"
    vibe_word_frequencies = dict(vibe_counts.most_common(50)) if vibe_counts else dict(DEFAULT_VIBE_FREQUENCIES)

    entity_spans = _entity_token_spans(doc)
    primary_candidates = []
    fallback_candidates = []

    for token in doc:
        if token.dep_ in ("dobj", "obj") and token.head.lemma_.lower() in CONSUMPTION_VERBS:
            phrase = _phrase_for_token(token, doc, entity_spans)
            if phrase:
                primary_candidates.append(phrase)
        elif token.dep_ == "nsubj" and token.head.lemma_.lower() == "be":
            phrase = _phrase_for_token(token, doc, entity_spans)
            if phrase and not set(phrase.lower().split()) & GENERIC_FALLBACK_WORDS:
                fallback_candidates.append(phrase)

    dish_candidates = [d for d in primary_candidates if _phrase_could_be_food(d)]
    if not dish_candidates:
        dish_candidates = [d for d in fallback_candidates if _phrase_could_be_food(d)]

    # Prefer multi-word phrases, then frequency. The failures were all bare
    # nouns ("Spot", "Care", "Gravy") while the good ones were qualified
    # ("Chicken Poke Bowl", "Masala Omelette") -- length is a decent proxy for
    # "someone named an actual dish" rather than "someone used a verb".
    counts = Counter(dish_candidates)
    famous_dish = max(counts, key=lambda d: (len(d.split()) > 1, counts[d]), default="Chef Special")
    return famous_dish, vibe_check, vibe_word_frequencies


def classify_review_aspects(df_reviews):
    if df_reviews.empty or "review_text" not in df_reviews.columns:
        return pd.DataFrame(columns=["place_id", "aspect", "is_positive"])

    sentences, sentence_place_ids = [], []
    for _, row in df_reviews.iterrows():
        text = str(row.get("review_text", ""))
        if not text.strip():
            continue
        for sent in nlp(text).sents:
            clean = sent.text.strip()
            if len(clean) >= 3:
                sentences.append(clean)
                sentence_place_ids.append(row["place_id"])

    if not sentences:
        return pd.DataFrame(columns=["place_id", "aspect", "is_positive"])

    aspect_predictions = aspect_classifier(sentences, ASPECT_LABELS, multi_label=True)
    sentiment_predictions = sentiment_analyzer(sentences, truncation=True, max_length=512)

    rows = []
    for place_id, aspect_pred, sentiment_pred in zip(sentence_place_ids, aspect_predictions, sentiment_predictions):
        is_positive = sentiment_pred["label"] == "POSITIVE"
        for label, score in zip(aspect_pred["labels"], aspect_pred["scores"]):
            if score >= ASPECT_CONFIDENCE_THRESHOLD:
                rows.append({"place_id": place_id, "aspect": label, "is_positive": is_positive})

    return pd.DataFrame(rows)


def _build_scorecard_row(place_id, group, model_score):
    """One restaurant's full scorecard record. `group` is that restaurant's
    review-level rows (from analyze_sentiment output); `model_score` is
    computed separately since it comes from compute_advanced_metrics."""
    total_reviews = len(group)
    pos_reviews = (group["sentiment_label"] == "POSITIVE").sum()
    pos_ratio = round((pos_reviews / total_reviews) * 100, 1)
    google_rating = group["google_rating"].iloc[0] if "google_rating" in group.columns else None
    avg_rating = round(google_rating, 1) if pd.notna(google_rating) else round(group["review_rating"].mean(), 1)

    restaurant_name = group["restaurant_name"].iloc[0]
    price_range = group["price_range"].iloc[0]
    cuisine = group["cuisine"].iloc[0]
    lat = group["latitude"].iloc[0] if "latitude" in group.columns else None
    lng = group["longitude"].iloc[0] if "longitude" in group.columns else None
    user_rating_count = group["user_rating_count"].iloc[0] if "user_rating_count" in group.columns else 0
    price_numeric = group["price_numeric"].iloc[0] if "price_numeric" in group.columns else None
    price_numeric_max = group["price_numeric_max"].iloc[0] if "price_numeric_max" in group.columns else price_numeric
    address = group["formatted_address"].iloc[0] if "formatted_address" in group.columns else None
    price_source = group["price_source"].iloc[0] if "price_source" in group.columns else None

    pos_texts = group[group["sentiment_label"] == "POSITIVE"]["review_text"].tolist()
    famous_dish, vibe_check, vibe_word_frequencies = extract_dish_and_vibe(pos_texts if pos_texts else group["review_text"].tolist())

    if "publish_time" in group.columns and group["publish_time"].notna().any():
        recent_group = group.sort_values(by="publish_time", ascending=False)
    else:
        recent_group = group

    recent_3_reviews = recent_group[
        ["review_text", "publish_time", "review_rating", "sentiment_label"]
    ].head(3).to_dict(orient="records")

    pos_snippet = group[group["sentiment_label"] == "POSITIVE"]["review_text"].head(1).values
    neg_snippet = group[group["sentiment_label"] == "NEGATIVE"]["review_text"].head(1).values

    return {
        "place_id": place_id,
        "restaurant_name": restaurant_name,
        "model_score": model_score,
        "cuisine": cuisine,
        "price_range": price_range,
        "price_numeric": price_numeric,
        "price_numeric_max": price_numeric_max,
        "price_source": price_source,
        "address": address,
        "avg_google_rating": avg_rating,
        "total_google_ratings": user_rating_count,
        "positive_sentiment_pct": pos_ratio,
        "reviews_analyzed": total_reviews,
        "famous_dish": famous_dish,
        "vibe_check": vibe_check,
        "vibe_word_frequencies": vibe_word_frequencies,
        "latitude": lat,
        "longitude": lng,
        "recent_3_reviews": recent_3_reviews,
        "sample_positive_review": pos_snippet[0] if len(pos_snippet) > 0 else "N/A",
        "sample_negative_review": neg_snippet[0] if len(neg_snippet) > 0 else "N/A",
    }


def generate_restaurant_insights(df_analyzed):
    if df_analyzed.empty:
        return pd.DataFrame(), {}

    if "publish_time" in df_analyzed.columns:
        df_analyzed["publish_time"] = pd.to_datetime(df_analyzed["publish_time"], errors="coerce")

    agg_list = [
        _build_scorecard_row(place_id, group, group["model_score"].iloc[0] if "model_score" in group.columns else 0.0)
        for place_id, group in df_analyzed.groupby("place_id")
    ]

    df_scorecard = pd.DataFrame(agg_list)
    df_scorecard = df_scorecard.sort_values(
        by=["model_score", "positive_sentiment_pct"], ascending=[False, False]
    ).reset_index(drop=True)

    top_venue = df_scorecard.iloc[0].to_dict() if not df_scorecard.empty else {}
    return df_scorecard, top_venue


def _jsonable(obj):
    if isinstance(obj, dict):
        return {k: _jsonable(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_jsonable(v) for v in obj]
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        val = float(obj)
        return None if np.isnan(val) else val
    if isinstance(obj, pd.Timestamp):
        return obj.isoformat() if pd.notnull(obj) else None
    if isinstance(obj, float) and pd.isna(obj):
        return None
    return obj


class TTLCache:
    def __init__(self, ttl_seconds=6 * 3600):
        self.ttl_seconds = ttl_seconds
        self._store = {}

    def get(self, key):
        entry = self._store.get(key)
        if entry is None:
            return None
        value, expires_at = entry
        if time.time() > expires_at:
            del self._store[key]
            return None
        return value

    def set(self, key, value):
        self._store[key] = (value, time.time() + self.ttl_seconds)


recommendation_cache = TTLCache(ttl_seconds=6 * 3600)

POPULAR_AREAS = [
    "Dubai Marina", "Downtown Dubai", "Jumeirah Beach Residence (JBR)", "Business Bay",
    "Palm Jumeirah", "Deira", "Bur Dubai", "Al Barsha", "Jumeirah Lake Towers (JLT)",
    "Dubai Hills Estate", "DIFC", "Al Quoz", "City Walk", "La Mer", "Umm Suqeim",
    "Al Karama", "Satwa", "Mirdif", "Arabian Ranches", "Dubai Silicon Oasis",
    "International City", "Discovery Gardens",
]


def _aggregate_stats(df_scorecard):
    return {
        "restaurants_analyzed": int(len(df_scorecard)),
        "reviews_analyzed": int(df_scorecard["reviews_analyzed"].sum()),
        "avg_google_rating": round(float(df_scorecard["avg_google_rating"].mean()), 1),
        "avg_sentiment_pct": round(float(df_scorecard["positive_sentiment_pct"].mean()), 1),
        "avg_spend_per_person": round(float(df_scorecard["price_numeric"].dropna().mean()), 0)
            if df_scorecard["price_numeric"].notna().any() else None,
    }


def get_recommendations(area, cuisine="", max_budget=None, use_cache=True):
    cache_key = (area.strip().lower(), (cuisine or "").strip().lower(), max_budget)
    if use_cache:
        cached = recommendation_cache.get(cache_key)
        if cached is not None:
            return cached

    df_reviews, df_summary = get_reviews_for_area(area=area, cuisine=cuisine, max_budget=max_budget)

    if df_reviews.empty:
        result = {
            "summary": df_summary.loc[0, "transparency_note"],
            "top_pick": None,
            "restaurants": [],
            "stats": {
                "restaurants_analyzed": 0, "reviews_analyzed": 0,
                "avg_google_rating": None, "avg_sentiment_pct": None, "avg_spend_per_person": None,
            },
        }
        if use_cache:
            recommendation_cache.set(cache_key, result)
        return result

    df_analyzed = analyze_sentiment(df_reviews)
    df_aspects = classify_review_aspects(df_analyzed)
    df_scores = compute_advanced_metrics(df_analyzed, df_aspects)
    df_analyzed = df_analyzed.merge(df_scores[["place_id", "model_score"]], on="place_id", how="left")
    df_scorecard, top_venue = generate_restaurant_insights(df_analyzed)

    result = _jsonable({
        "summary": df_summary.loc[0, "transparency_note"],
        "top_pick": top_venue if top_venue else None,
        "restaurants": df_scorecard.to_dict(orient="records"),
        "stats": _aggregate_stats(df_scorecard),
    })

    if use_cache:
        recommendation_cache.set(cache_key, result)

    return result


def stream_recommendation_events(area, cuisine="", max_budget=None, use_cache=True):
    cache_key = (area.strip().lower(), (cuisine or "").strip().lower(), max_budget)

    if use_cache:
        cached = recommendation_cache.get(cache_key)
        if cached is not None:
            for row in cached.get("restaurants", []):
                yield {"type": "restaurant", "restaurant": row}
            yield {
                "type": "done",
                "summary": cached.get("summary"),
                "top_pick": cached.get("top_pick"),
                "stats": cached.get("stats"),
            }
            return

    df_reviews, df_summary = get_reviews_for_area(area=area, cuisine=cuisine, max_budget=max_budget)

    if df_reviews.empty:
        result = {
            "summary": df_summary.loc[0, "transparency_note"],
            "top_pick": None,
            "restaurants": [],
            "stats": {
                "restaurants_analyzed": 0, "reviews_analyzed": 0,
                "avg_google_rating": None, "avg_sentiment_pct": None, "avg_spend_per_person": None,
            },
        }
        if use_cache:
            recommendation_cache.set(cache_key, result)
        yield {"type": "done", **result}
        return

    df_analyzed = analyze_sentiment(df_reviews)
    if "publish_time" in df_analyzed.columns:
        df_analyzed["publish_time"] = pd.to_datetime(df_analyzed["publish_time"], errors="coerce")

    place_groups = list(df_analyzed.groupby("place_id"))
    yield {"type": "start", "total": len(place_groups)}

    scorecard_rows = []
    for place_id, group in place_groups:
        df_aspects_group = classify_review_aspects(group)
        scored = compute_advanced_metrics(group, df_aspects_group)
        model_score = float(scored["model_score"].iloc[0]) if not scored.empty else 0.0
        row = _jsonable(_build_scorecard_row(place_id, group, model_score))
        scorecard_rows.append(row)
        yield {"type": "restaurant", "restaurant": row}

    scorecard_rows.sort(key=lambda r: (r["model_score"], r["positive_sentiment_pct"]), reverse=True)
    df_scorecard = pd.DataFrame(scorecard_rows)
    top_venue = scorecard_rows[0] if scorecard_rows else None

    result = {
        "summary": df_summary.loc[0, "transparency_note"],
        "top_pick": top_venue,
        "restaurants": scorecard_rows,
        "stats": _aggregate_stats(df_scorecard),
    }
    if use_cache:
        recommendation_cache.set(cache_key, result)

    yield {"type": "done", "summary": result["summary"], "top_pick": result["top_pick"], "stats": result["stats"]}


def _prewarm_cache_loop():
    refresh_interval = max(60, recommendation_cache.ttl_seconds - 300)
    while True:
        for area in POPULAR_AREAS:
            try:
                get_recommendations(area=area, cuisine="", max_budget=None, use_cache=True)
            except Exception as exc:
                print(f"[prewarm] failed for {area!r}: {exc}")
            time.sleep(5)  # small gap between areas so we don't hammer Google/HF back-to-back
        time.sleep(refresh_interval)


if os.environ.get("PREWARM_CACHE", "").lower() in ("1", "true", "yes"):
    threading.Thread(target=_prewarm_cache_loop, daemon=True).start()
else:
    print("[prewarm] disabled -- set PREWARM_CACHE=1 to warm popular areas in the background")


# ==========================================
# FASTAPI APP
# ==========================================
app = FastAPI(title="Dubai Restaurant Recommendation API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["GET"],
    allow_headers=["*"],
)


@app.get("/api/area-suggest")
def area_suggest(request: Request, q: str = Query("", description="Partial area name typed so far")):
    _check_rate_limit("suggest", request)
    return {"suggestions": suggest_dubai_areas(q)}


@app.get("/api/recommend")
def recommend(
    request: Request,
    area: str = Query(..., min_length=1, description="Required. e.g. 'Dubai Marina'"),
    cuisine: str = Query("", description="Optional cuisine filter, e.g. 'Italian'"),
    budget: float = Query(None, description="Optional max budget in AED"),
):
    _check_rate_limit("search", request)
    try:
        return get_recommendations(area=area, cuisine=cuisine, max_budget=budget)
    except DailyLimitReached as exc:
        raise HTTPException(status_code=429, detail=str(exc))
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    except RuntimeError as exc:
        raise HTTPException(status_code=502, detail=str(exc))


@app.get("/api/recommend/stream")
def recommend_stream(
    request: Request,
    area: str = Query(..., min_length=1, description="Required. e.g. 'Dubai Marina'"),
    cuisine: str = Query("", description="Optional cuisine filter, e.g. 'Italian'"),
    budget: float = Query(None, description="Optional max budget in AED"),
):
    _check_rate_limit("search", request)

    def event_generator():
        try:
            for event in stream_recommendation_events(area=area, cuisine=cuisine, max_budget=budget):
                yield json.dumps(event) + "\n"
        except DailyLimitReached as exc:
            # Tagged so the page can show it as a limit rather than a failed search.
            yield json.dumps({"type": "error", "kind": "limit", "message": str(exc)}) + "\n"
        except (ValueError, RuntimeError) as exc:
            yield json.dumps({"type": "error", "message": str(exc)}) + "\n"

    return StreamingResponse(event_generator(), media_type="application/x-ndjson")


FRONTEND_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "frontend")


@app.get("/")
def serve_search_page():
    return FileResponse(os.path.join(FRONTEND_DIR, "search_page.html"))


@app.get("/results")
def serve_results_page():
    return FileResponse(os.path.join(FRONTEND_DIR, "results_page.html"))


@app.get("/health")
def health():
    return {"status": "ok"}
