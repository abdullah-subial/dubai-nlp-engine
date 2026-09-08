# Deploying Dine Dubai

The whole app is one process: FastAPI serves both HTML pages and the JSON API,
so there is no separate frontend to host and no database to provision.

---

## 1. Lock the dependency versions (do this first)

`requirements.txt` ships with pins resolved against Python 3.11, but the venv on
your Mac is the version set the app has actually been QA'd against. Make that
the source of truth:

```bash
source venv/bin/activate
pip freeze | grep -iE '^(fastapi|uvicorn|pandas|numpy|requests|torch|spacy|transformers|nltk)==' 
```

Copy those versions into `requirements.txt`, keeping `uvicorn[standard]`'s
extras marker. `torch` is deliberately left unpinned in the shipped file — pin
it to whatever `pip freeze` reports.

> The Dockerfile reads the torch spec straight off that line, so pinning it in
> `requirements.txt` is all that's needed — nothing in the Dockerfile changes.

---

## 2. Build and test locally

```bash
docker build -t dine-dubai .
```

First build takes a while: it installs the CPU-only torch wheel and downloads
both transformer models plus the spaCy model into the image. Later builds reuse
those layers unless `requirements.txt` changes.

Run it:

```bash
docker run --rm -p 8000:7860 \
  -e GOOGLE_PLACES_API_KEY="your_key" \
  -e HF_TOKEN="your_token" \
  dine-dubai
```

Then check:

```bash
curl localhost:8000/health
open http://localhost:8000
```

**What a good startup log looks like** — the models should load with no
download progress bars at all, since they're baked into the image:

```
[startup] Skipping CPU quantization ...   # only on some platforms; harmless
[prewarm] disabled -- set PREWARM_CACHE=1 to warm popular areas in the background
INFO:     Application startup complete.
```

If you see Hugging Face download bars, the prefetch layer didn't take effect —
check that `HF_HOME` is the same at build and run time.

---

## 3. Environment variables

| Variable | Required | Purpose |
|---|---|---|
| `GOOGLE_PLACES_API_KEY` | **Yes** — the app refuses to start without it | Places Text Search + Autocomplete |
| `HF_TOKEN` | No | Avoids Hugging Face rate limits. Models are baked in, so this matters less in the image than locally |
| `PORT` | No (defaults to 7860) | Most hosts inject this automatically |
| `PREWARM_CACHE` | No (defaults off) | **Set to `1` in production.** Warms popular areas in the background so a demo link returns instantly. It's off by default because it saturates a laptop's CPU — on a deployed box with idle capacity it's exactly what you want |

Never bake these into the image. Pass them at run time.

---

## 4. Deploy to Hugging Face Spaces (recommended)

Best fit here: the free CPU tier has enough memory for both transformer models,
it's built for ML demos, and it's a credible link for a portfolio post.

1. Create a new Space, SDK = **Docker**, hardware = CPU basic.
2. Spaces needs a `README.md` at the repo root with YAML frontmatter — it won't
   pick up the Dockerfile without it:

   ```yaml
   ---
   title: Dine Dubai
   emoji: 🍽️
   colorFrom: red
   colorTo: yellow
   sdk: docker
   app_port: 7860
   ---
   ```

3. Add your keys under **Settings → Variables and secrets** as *secrets*:
   `GOOGLE_PLACES_API_KEY`, `HF_TOKEN`, and `PREWARM_CACHE=1` as a variable.
4. Push:

   ```bash
   git remote add space https://huggingface.co/spaces/<username>/dine-dubai
   git push space claude/dubai-restaurant-recommendation-in2vu0:main
   ```

Spaces builds the Dockerfile and serves on 7860, which the image already
defaults to.

### Other hosts

Railway, Render, Fly.io and Cloud Run all work from the same Dockerfile — they
inject `PORT`, which the image respects. The one hard requirement is **memory**:
two transformer models plus spaCy and pandas will not fit a 512MB free tier.
Budget ~2GB.

---

## 5. Before you share the URL publicly

Worth doing before the link goes anywhere, not after:

- **Restrict the API key.** In Google Cloud Console, limit it by HTTP referrer
  or IP, and to just the Places APIs it needs.
- **Set a billing alert and a daily quota cap** on Places Text Search and
  Places Autocomplete.
- **Rate-limit `/api/area-suggest` and `/api/recommend`.** Autocomplete fires as
  people type, and both endpoints bill per request. The in-process suggestion
  cache helps with repeat prefixes but won't stop a bot or someone holding down
  a key in the Area field.
