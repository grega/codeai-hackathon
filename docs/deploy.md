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

## Config reference

| Var | Default | Notes |
|---|---|---|
| `PROVIDER_RIGGING` | `mock` | `mock` or `real` |
| `PROVIDER_POSING` | `mock` | `mock` or `real` |
| `PROVIDER_TRAINING` | `mock` | `mock` or `real` |
| `EPISODE_RATE` | `20` | episodes/sec at speed 1; higher means more work per trainee |
| `DATA_DIR` | `./data` | upload location |
| `MAX_UPLOAD_BYTES` | `8388608` | 8MB |
| `PORT` | — | set by Heroku |
