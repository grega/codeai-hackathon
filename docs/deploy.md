# Deploying to Heroku

## Constraint: one web dyno

The store is in-memory (`store.py`), so **every request for a given avatar must
reach the same process**. Scale past one dyno and avatars will intermittently
404 as requests are routed to a dyno that has never seen them.

- `Procfile` pins `--workers 1` and gets concurrency from `--threads 32`
  instead. Threads share memory; processes do not.
- `app.json` pins `"quantity": 1`.
- Keep `heroku ps:scale web=1`.

Lifting this means giving `store.py` a real backend (Postgres, or Redis for the
run history). Until then, one dyno.

## Deploy

```bash
heroku create avatar-trainer          # also adds the `heroku` git remote
heroku ps:scale web=1
git push heroku HEAD:main
heroku open
```

You are on a feature branch, so `HEAD:main` is what pushes it — Heroku only
builds its default branch.

```bash
heroku config:set EPISODE_RATE=20     # optional
heroku logs --tail
```

## Dyno type

Use **Basic** or higher, not Eco. Eco dynos sleep after 30 minutes of
inactivity, and since all state is in memory, waking up means every avatar,
move and trained behaviour is gone. Mid-lesson that reads as the app losing a
child's work.

Note that **all dynos restart at least once every 24 hours** regardless of type.
Everything in memory goes with them. That is acceptable for a prototype — a
session is one lesson — but it is the first thing to fix if this is ever used
for work anyone expects to keep.

## What each deploy file does

| File | Purpose |
|---|---|
| `Procfile` | gunicorn with `gthread` workers — see below |
| `.python-version` | `3.13.9`, read by the `heroku/python` buildpack |
| `requirements.txt` | includes `gunicorn` |
| `app.json` | metadata, pinned formation, provider env vars |

`.tool-versions` is for local asdf; the buildpack reads `.python-version`. Keep
the two in step.

### Why `gthread`

Training is a long-lived SSE response, and jobs run on background threads.

- The default `sync` worker would tie up a whole worker per streaming client and
  kill it at the 30s timeout.
- `gevent` would need monkey-patching, which conflicts with the `threading`
  primitives in `jobs.py` and `training.py`.

`gthread` with 32 threads gives roughly 32 concurrent trainees on one dyno.
That is a classroom; it is not the internet.

## SSE and the Heroku router

The router applies two timeouts that a naive streaming endpoint fails:

| Rule | Limit | What handles it |
|---|---|---|
| First byte of the response (`H12`) | 30s | `app.py` flushes a `: open` comment immediately |
| Gap between bytes (`H15`) | 55s | `training.py` sends `: keepalive` every 15s |

The keepalive is what makes **pausing** work: a paused run produces no episodes,
so without it the connection sends nothing and the router hangs up after 55
seconds.

If a connection drops anyway, `EventSource` reconnects and the server replays
from episode 1 — `api.js` discards episodes it has already seen, so the learning
curve is not drawn on top of itself.

The router streams responses rather than buffering them, so SSE works without
extra configuration. The `X-Accel-Buffering: no` header in `app.py` is a no-op
here and matters only behind nginx.

## Ephemeral filesystem

Sketches are written to `DATA_DIR` (default `./data`) on the dyno and are lost
on every restart and deploy. That matches the in-memory store — the avatar
records go at the same moment — so nothing is left dangling and no add-on is
needed. Persisting uploads means S3 and a change to `store.add_avatar()`.

## External dependency to be aware of

Vue and three.js load from `unpkg.com` at runtime via the import map in
`static/index.html`. On a network that blocks unpkg — school filtering being the
realistic case — the page will not boot at all. To remove the risk, vendor the
files into `static/vendor/` and point the import map at local paths.

## Verifying a deploy

```bash
curl -s https://YOUR_APP.herokuapp.com/api/schema | head -c 200
heroku logs --tail
```

`/api/schema` exercises the app rather than just the web server: it reads the
bone tree and resolves the configured providers.

Then click through all four screens. The status bar shows which providers are
mock and which are real, so what the deploy is actually running is never in
doubt.

## Config

Locally, copy `.env.example` to `.env`. On Heroku use config vars:

```bash
heroku config:set PROVIDER_POSING=real AWS_DEFAULT_REGION=eu-west-1
heroku config                                  # what is currently set
```

`config.py` loads `.env` without overwriting anything already in the
environment, so the precedence is **real environment > `.env` > defaults**. That
is what makes one file work in both places: Heroku config vars arrive as real
environment variables and win, and `.env` is gitignored so it never reaches the
slug.

| Var | Default | Notes |
|---|---|---|
| `PROVIDER_RIGGING` | `mock` | `mock` or `real` |
| `PROVIDER_POSING` | `mock` | `mock` or `real` |
| `PROVIDER_TRAINING` | `mock` | `mock` or `real` |
| `RIGGING_SERVICE_URL` | — | supplied ngrok HTTPS base URL; server-side only |
| `RIGGING_SERVICE_TIMEOUT` | `300` | overall real-rigging deadline in seconds |
| `RIGGING_POLL_INTERVAL` | `5` | seconds between mesh/rig status checks |
| `EPISODE_RATE` | `20` | episodes/sec at speed 1; higher means more work per trainee |
| `DATA_DIR` | `./data` | upload location |
| `MAX_UPLOAD_BYTES` | `8388608` | 8MB |
| `AWS_DEFAULT_REGION` | — | read by boto3, not by this repo |
| `AWS_ACCESS_KEY_ID` | — | " |
| `AWS_SECRET_ACCESS_KEY` | — | " |
| `AWS_SESSION_TOKEN` | — | temporary credentials only; omit for long-lived IAM keys |
| `LLM_API_TOKEN` | — | **unset = the Bedrock endpoint returns 404.** Server-side only |
| `BEDROCK_ALLOWED_MODELS` | — | comma-separated; **empty = every model refused** |
| `BEDROCK_REGION` | `AWS_DEFAULT_REGION` | only if Bedrock is elsewhere |
| `BEDROCK_MAX_TOKENS` | `4096` | ceiling whatever the caller asks for |
| `BEDROCK_MAX_PROMPT_CHARS` | `20000` | prompt size limit |
| `LLM_RATE_PER_MINUTE` | `10` | per caller |
| `LLM_RATE_PER_DAY` | `500` | whole deployment — this bounds the bill |
| `PORT` | — | set by Heroku |

The Bedrock prompt endpoint is off unless `LLM_API_TOKEN` **and**
`BEDROCK_ALLOWED_MODELS` are both set — see [llm-endpoint.md](llm-endpoint.md).
Its rate limits are per-process and in memory, which is sound on one dyno and
would need Redis if that ever changes.

The AWS variables need no code here — boto3 reads them from the environment
itself. `.env` is loaded at `config` import, which happens before any provider
or Bedrock client is constructed, so anything creating a boto3 client at import
time still sees them.

`boto3` is in `requirements.txt` for the Bedrock endpoint. The client is built
lazily on first use, so the app starts and serves the avatar experience fine on
a machine with no AWS credentials at all.
