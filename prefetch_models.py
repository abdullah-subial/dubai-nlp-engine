"""Downloads every model and corpus the server needs, at image build time.

server.py fetches these lazily on first import, which is fine on a laptop but
wrong in a container: it makes every cold start re-download ~500MB, turns a
Hugging Face rate-limit or a network blip into a failed boot, and means the
image isn't actually self-contained. Baking them in trades image size for a
start-up that is fast and offline.

Run as its own build step, not by importing server.py -- server.py raises at
import when GOOGLE_PLACES_API_KEY is unset, and the build has no business
holding a real API key.
"""

import pathlib
import sys

SPACY_MODEL = "en_core_web_sm"
SENTIMENT_MODEL = "distilbert-base-uncased-finetuned-sst-2-english"
ASPECT_MODEL = "typeform/distilbert-base-uncased-mnli"


def _assert_in_sync_with_server():
    """Fail the build if these names have drifted from what server.py loads.

    Without this, renaming a model in server.py still produces a green build
    and an image that silently re-downloads the real model on every cold
    start -- the exact failure this script exists to prevent.
    """
    server_src = (pathlib.Path(__file__).parent / "server.py").read_text()
    for name in (SPACY_MODEL, SENTIMENT_MODEL, ASPECT_MODEL):
        if name not in server_src:
            sys.exit(
                f"prefetch_models.py is out of sync with server.py: {name!r} "
                "is not referenced there. Update both together."
            )


def main():
    _assert_in_sync_with_server()

    print(f"[prefetch] spaCy: {SPACY_MODEL}")
    from spacy.cli import download as spacy_download
    spacy_download(SPACY_MODEL)

    print(f"[prefetch] sentiment: {SENTIMENT_MODEL}")
    from transformers import AutoModelForSequenceClassification, AutoTokenizer
    AutoTokenizer.from_pretrained(SENTIMENT_MODEL)
    AutoModelForSequenceClassification.from_pretrained(SENTIMENT_MODEL)

    print(f"[prefetch] aspect classifier: {ASPECT_MODEL}")
    AutoTokenizer.from_pretrained(ASPECT_MODEL)
    AutoModelForSequenceClassification.from_pretrained(ASPECT_MODEL)

    # Only used to expand the consumption-verb set. server.py falls back to a
    # stored set when this is missing, so a failure here is not fatal -- but
    # in an image with a network there's no reason not to have the real thing.
    print("[prefetch] nltk: wordnet")
    import nltk
    if not nltk.download("wordnet"):
        print("[prefetch] wordnet download failed -- server will use its stored fallback")

    print("[prefetch] done")


if __name__ == "__main__":
    main()
