We think engaging learners to create and train their own model is a compelling computational exercise and generative AI collaboration

Go to the wiki for more information: https://github.com/grega/codeai-hackathon/wiki

## Setup

Python is pinned with [asdf](https://asdf-vm.com).

```bash
asdf install                                  # python 3.13.9, per .tool-versions
python -m venv .venv
.venv/bin/pip install -r requirements.txt
```

## Run

```bash
.venv/bin/flask --app app run --debug         # http://localhost:5000
.venv/bin/pytest                              # contract suite
```

`--debug` restarts on file change and the store is in memory, so an edit
mid-session loses the current avatar. Drop the flag when demoing.

No npm install and no build step — Vue and three.js load from a CDN via the
import map in `static/index.html`.

## Deploy

Heroku, via the `Procfile` / `.python-version` / `app.json` in this repo:

```bash
heroku create avatar-trainer
heroku ps:scale web=1            # must stay 1 — the store is in memory
git push heroku HEAD:main        # Heroku only builds its default branch
```

Use a Basic dyno or higher: Eco dynos sleep, and sleeping loses every avatar.
Full notes, including why one dyno and how SSE survives the router timeouts:
[docs/deploy.md](docs/deploy.md).

## Providers

Each phase reads a `mock`/`real` env var, independently:

```bash
PROVIDER_POSING=real .venv/bin/flask --app app run
```

| Var | Phase | Interface | Mock |
|---|---|---|---|
| `PROVIDER_RIGGING` | 1 — sketch → rigged avatar | `Rigger` | returns the standard 16-bone figure |
| `PROVIDER_POSING` | 2 — prompt → pose | `Poser` | keyword-matches hand-authored clips |
| `PROVIDER_TRAINING` | 3–4 — train towards a pose | `Trainer` | hill-climber scored by `rewards.py` |

The UI status bar shows which are active. Adding one:
[docs/adding-a-provider.md](docs/adding-a-provider.md).

Other vars: `EPISODE_RATE` (episodes/sec at speed 1, default 20), `DATA_DIR`,
`MAX_UPLOAD_BYTES`.

## Layout

```
app.py              routes
schemas.py          the contract — bones, poses, clips, validation
rewards.py          reward function; the UI sliders map to its terms
training.py         run lifecycle, SSE pacing, pause/stop/speed
jobs.py             thread-backed job runner for slow providers
store.py            in-memory registries, wiped on restart
providers/
  base.py           the three ABCs
  mock/             stand-ins to copy
  real/             implementations go here
static/
  index.html        importmap: vue + three
  js/api.js         every fetch call
  js/viewport.js    three.js; renders procedural rigs and GLBs alike
  js/components/    one per screen
tests/test_contract.py
```

## Not built

Real providers, auth, persistence across restarts, mobile layout, and the
environments themselves — the Play screen is a stub holding the behaviour
library they will draw on.
