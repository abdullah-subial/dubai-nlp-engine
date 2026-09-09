---
title: Dine Dubai
emoji: 🍽️
colorFrom: red
colorTo: yellow
sdk: docker
app_port: 7860
pinned: false
---

# Dine Dubai

A restaurant recommendation engine for Dubai that reads what guests actually
wrote, not just how many stars they left.

Star ratings compress a whole dinner into one number, and a 4.5 with 8,000
ratings means something very different from a 4.5 with twelve. This reads the
review text with an NLP pipeline — sentiment, what people praise or complain
about, whether they felt it was worth the price — and combines that with
Google's rating into a single **Model Score** out of 100.

Two pages: search by area, cuisine and budget; get a ranked dashboard with a
top pick, the metrics behind every result, and charts showing how the area's
restaurants compare on price versus quality.

---

## How it works

```
Area + cuisine + budget
        │
        ▼
Google Places Text Search ──── rating cascade: try 4.9 first, loosen until
        │                      there are enough candidates to rank
        ▼
Reviews for the top-rated venues
        │
        ├─► DistilBERT sentiment ──── positive/negative per review
        ├─► Zero-shot classification ─ food / service / price / ambiance
        └─► spaCy + WordNet ────────── the dish people name, the words they
        │                              reach for
        ▼
Weighted Model Score ──► streamed to the page restaurant-by-restaurant
```

Results stream as each restaurant finishes analysis rather than waiting for the
whole batch, so the first result appears in seconds instead of at the end.

### The Model Score

Six signals make up the full 100%, and a separate penalty of up to 10 points
can be deducted afterwards:

| Weight | Signal | What it measures |
|---|---|---|
| 30% | Guest sentiment | Share of reviews an AI model reads as positive |
| 25% | Google rating | The venue's overall star rating |
| 15% | Review volume | Confidence from total ratings, with diminishing returns |
| 10% | Sentiment momentum | Whether recent reviews read better than the stars suggest |
| 10% | Aspect score | Sentiment specifically about food and service |
| 10% | Price-to-value | How reasonable guests find the price for what they get |
| −10 pts | Negativity risk | Deducted for very low ratings or strongly negative recent reviews |

### Design decisions worth knowing

- **Area input is locked to verified Dubai suggestions.** Typing a free-text
  area was how a search for "mirdiff" quietly returned Dubai Mall restaurants —
  Google's autocomplete tolerates typos, its text search does not. The field now
  only submits a value that came from the live suggestion list.
- **Prices are estimated when Google has none.** Venues without published prices
  get an estimate from local area quartiles, marked with an asterisk rather than
  presented as fact.
- **Cold starts are cheap because models are baked into the image**, not
  downloaded at boot.

---

## Running it

### Docker (matches production)

```bash
docker build -t dine-dubai .

docker run --rm -p 8000:7860 \
  -e GOOGLE_PLACES_API_KEY="your_key" \
  dine-dubai
```

Open <http://localhost:8000>.

### Locally

```bash
python3 -m venv venv && source venv/bin/activate
pip install -r requirements.txt
export GOOGLE_PLACES_API_KEY="your_key"
uvicorn server:app --reload --port 8000
```

### Environment variables

| Variable | Required | Purpose |
|---|---|---|
| `GOOGLE_PLACES_API_KEY` | Yes — the app won't start without it | Places Text Search and Autocomplete |
| `HF_TOKEN` | No | Avoids Hugging Face rate limits |
| `PORT` | No (default 7860) | Most hosts inject this |
| `PREWARM_CACHE` | No (default off) | Set to `1` in production to warm popular areas in the background. Off by default because it saturates a laptop's CPU |

Full deployment instructions, including Hugging Face Spaces, are in
[DEPLOY.md](DEPLOY.md).

---

## API

| Endpoint | Purpose |
|---|---|
| `GET /` | Search page |
| `GET /results` | Results dashboard |
| `GET /api/area-suggest?q=` | Dubai-only area typeahead |
| `GET /api/recommend?area=&cuisine=&budget=` | Full analysis, one JSON response |
| `GET /api/recommend/stream?...` | Same, streamed per restaurant (NDJSON) |
| `GET /health` | Liveness check |

---

## Repository layout

| Path | Role |
|---|---|
| `server.py` | Backend and page serving — one process does both |
| `frontend/` | The two pages, plain HTML/CSS/JS with no build step |
| `Dubai_NLP_Engine.ipynb` | Where the pipeline was developed. **`server.py` is generated from cell 10** — change the notebook first |
| `prefetch_models.py` | Downloads models at image build time |
| `Dockerfile` | CPU-only torch, models baked in, non-root |
| `DEPLOY.md` | Deployment guide |

---

## Models

| Purpose | Model |
|---|---|
| Sentiment | `distilbert-base-uncased-finetuned-sst-2-english` |
| Aspect classification | `typeform/distilbert-base-uncased-mnli` (zero-shot) |
| Dish and phrase extraction | spaCy `en_core_web_sm` + WordNet |

---

## Limitations

- Scores come from each venue's most recent Google reviews, not its full
  history — a current read, not a permanent verdict.
- Dubai only. Other emirates are rejected rather than silently mixed in.
- An independent analysis of public reviews for guest discovery. Not an
  official rating, certification, or endorsement of any restaurant.
