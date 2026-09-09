# Dine Dubai -- single-process FastAPI app serving both pages and the API.
#
# Matches the Python the app is developed and QA'd on. This was briefly 3.11,
# on the theory that an older interpreter has broader wheel coverage -- but the
# pinned versions come from a 3.13 venv, and some of them (numpy 2.5.x) require
# 3.12+, so 3.11 couldn't install them. Pinning versions from one interpreter
# and building on another defeats the point of pinning; keep these in step.
FROM python:3.13-slim

# PYTHONUNBUFFERED: startup logs ([prewarm], [wordnet], [startup]) reach the
#   platform's log viewer immediately instead of sitting in a buffer.
# HF_HOME / NLTK_DATA: pin the caches to paths we control, so the models baked
#   in at build time are the ones found at runtime.
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    HF_HOME=/opt/models/hf \
    NLTK_DATA=/opt/models/nltk

WORKDIR /app

# --- Dependencies -----------------------------------------------------------
# Copied and installed before the app code so editing a .py or .html file
# doesn't invalidate this layer and re-download ~1GB of wheels.
COPY requirements.txt .

# torch comes from PyTorch's CPU-only wheel index. The default PyPI build
# bundles CUDA and drags in roughly 2.5GB of GPU libraries that a CPU host will
# never execute -- installing the CPU build first keeps the image about an
# order of magnitude smaller. The spec is read straight out of requirements.txt
# so there is only one place to bump it; the second install then sees torch
# already satisfied and leaves the CPU build in place.
RUN TORCH_SPEC="$(grep -iE '^torch([=<>!~]|$)' requirements.txt || echo torch)" \
 && pip install --no-cache-dir "$TORCH_SPEC" --index-url https://download.pytorch.org/whl/cpu \
 && pip install --no-cache-dir -r requirements.txt

# --- Models -----------------------------------------------------------------
# Baked in so cold starts are fast and offline. Its own layer, after deps and
# before app code, because it is the slowest step and the least likely to change.
COPY prefetch_models.py server.py ./
RUN mkdir -p "$HF_HOME" "$NLTK_DATA" \
 && python prefetch_models.py

# --- App --------------------------------------------------------------------
COPY frontend ./frontend

# Run unprivileged. UID 1000 specifically because Hugging Face Spaces runs
# containers as that UID -- matching it avoids permission errors on the model
# cache there, and is good practice on every other host.
RUN useradd --create-home --uid 1000 appuser \
 && chown -R appuser:appuser /app /opt/models
USER appuser

# 7860 is the Hugging Face Spaces convention; PORT overrides it for hosts that
# inject their own (Railway, Render, Fly, Cloud Run).
ENV PORT=7860
EXPOSE 7860

# Generous start period: even with the models baked in, loading two
# transformers plus spaCy off disk takes a while on a small CPU instance.
HEALTHCHECK --interval=30s --timeout=10s --start-period=180s --retries=3 \
    CMD python -c "import os,urllib.request;urllib.request.urlopen('http://127.0.0.1:'+os.environ['PORT']+'/health')"

# Shell form so ${PORT} expands at container start rather than being passed
# through as a literal string.
CMD uvicorn server:app --host 0.0.0.0 --port ${PORT}
