# Deploying Find My Table

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
| `DAILY_SEARCH_CAP` | No (defaults 1000) | Searches per day across all visitors before the app stops calling Google. Cache hits don't count |
| `SEARCH_PER_HOUR` | No (defaults 12) | Searches per visitor per hour |
| `SUGGEST_PER_MINUTE` | No (defaults 60) | Area-typeahead calls per visitor per minute |
| `PREWARM_CACHE` | No (defaults off) | Warms popular areas in the background so a demo link returns instantly. Leave it off for a first deploy — it saturates a laptop's CPU, and a small shared instance isn't far off a laptop. Turn it on once you've confirmed the deployed app is responsive, and watch that it stays that way |

Never bake these into the image. Pass them at run time.

---

## 4. Deploy to Hugging Face Spaces (recommended)

Best fit here: it's built for ML demos and it's a credible link for a
portfolio post. Check the current pricing page before you count on a tier
being free — what Spaces offers on Docker SDK has changed before, and the
memory floor below still applies whichever tier you land on.

1. Create a new Space, SDK = **Docker**, hardware = the smallest CPU tier
   available to your account.
2. Spaces needs a `README.md` at the repo root with YAML frontmatter — it won't
   pick up the Dockerfile without it:

   ```yaml
   ---
   title: Find My Table
   emoji: 🍽️
   colorFrom: red
   colorTo: yellow
   sdk: docker
   app_port: 7860
   ---
   ```

3. Add your keys under **Settings → Variables and secrets** as *secrets*:
   `GOOGLE_PLACES_API_KEY` and `HF_TOKEN`. Leave `PREWARM_CACHE` unset for now
   (see the table above).
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

- **Restrict the API key to Places API (New)**, under *API restrictions*. That
  single entry covers both calls the app makes -- Text Search and Autocomplete.

  Leave **Application restrictions** set to **None**. This is deliberate, not
  an oversight: the key is used server-side, from the Python process inside the
  container, so an HTTP-referrer restriction has no referrer to match and
  would reject every call. IP allowlisting fails for a different reason --
  a Spaces container's egress address isn't stable enough to pin.

  That leaves the API restriction plus the app's own rate limits as the
  ceiling, which is the trade this deployment makes.
- **Set a billing alert** on the project. A daily quota cap on Places is worth
  setting too, but it isn't available on a free-tier billing account, so treat
  the alert as the thing you'll actually get.
- **Rate limiting is built in** (see the variables above) and is the real
  ceiling here: with no Google-side quota cap available, and spend-cap
  enforcement not covering Places, `DAILY_SEARCH_CAP` is what stands between a
  shared link and a surprise bill. Tune it to what you're willing to spend in a
  day.

### Rotating the key

Update the deployment before revoking the old key, or the live site 500s in the
gap between the two:

1. Create the new key and restrict it as above.
2. Update `GOOGLE_PLACES_API_KEY` in the host's secrets. On Spaces, changing a
   secret restarts the Space; if it doesn't, restart it from Settings.
3. Run one search against the live site to confirm the new key works.
4. **Then** delete the old key.
