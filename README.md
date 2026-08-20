# Train Your Avatar

We think engaging learners to create and train their own model is a compelling
computational exercise and generative AI collaboration.

The experience now runs end to end:

```text
draw -> render image -> rig -> teach -> train -> play -> render a WebM video
```

## Background

Audience: 11-14

A prototype of an in-browser generative AI/ML learning experience for school kids.

Think a Tamagotchi that the young person, or user, can create and train, which can then be presented novel environments for it to navigate in some way. 

The input is freeform, an image supplied by the user, and the output is a 2D model. This would be the user's avatar. This can be offloaded to an SLM / LLM. 

This avatar would be rigged, and then could be trained to perform a bunch of actions. The training is an interactive part of the whole experience for the user - learning how training works, and understanding how this maps to the actual outcome. Animation should be mapped to behaviour. 

Phases:

1. User sketches (eg a stick person), we translate this into a 2D model using an LLM, the 2D model is auto-rigged. Output is rigged GLB.
2. Input is GLB, take a prompt eg "waves arms in air" and put it through an LLM, output is a pose / a set of poses (as a model)
3. Now we can send items to a reinforcement model: 1 neutral pose and 2. desired pose
4. Desired pose is given a reward function, points awarded for a pose which is close / the same as the desired one. This needs to be visualised as a learning activity.

For reference, we can start with @cindyloo's animation pipeline <a href="https://github.com/cindyloo/generative-ai-server">here</a>

We can also reference Faraz Faraqi's paper on InstructMesh <a href="https://groups.csail.mit.edu/hcie/files/research-projects/xspine/xspine.pdf">here</a>.

## Setup

Python is pinned with [asdf](https://asdf-vm.com).
Google Chrome is required for the final video recording step.

```bash
asdf install                                  # python 3.13.9, per .tool-versions
python -m venv .venv
.venv/bin/pip install -r requirements.txt
cp .env.example .env                          # optional; all vars have defaults
```

`.env` is gitignored and never overrides a real environment variable, so the
same config works locally and as Heroku config vars.

## Run

```bash
.venv/bin/flask --app app run --debug         # http://localhost:5000
.venv/bin/pytest                              # contract suite
```

`--debug` restarts on file change and the store is in memory, so an edit
mid-session loses the current avatar. Drop the flag when demoing.

No npm install and no build step — Vue and three.js load from a CDN via the
import map in `static/index.html`.

## Final video rendering

The Render step turns a saved behaviour into a customizable five-second WebM
video. It combines the current avatar rig with the behaviour's JSON animation
clip, so the server does not need to bake another GLB:

```text
Rig (procedural or GLB) + Clip -> Three.js preview -> 1280x720 WebM
```

The learner can choose the move, stage lighting, and camera motion before
recording. Rendering uses `canvas.captureStream()` and `MediaRecorder`; the
finished video remains in the browser and is downloaded directly rather than
uploaded to Flask.

Real GLB rigs must expose the 16 contract bone names from `schemas.py`. The same
viewport path applies the JSON clip to procedural and GLB rigs, which keeps the
mock and real provider workflows aligned.

To skip drawing and auto-rigging, use **Load GLB** on the Draw step. The upload
must be a skinned binary glTF with all 16 contract bones, either under their
contract names or the supported Mixamo aliases. GLB sideloads are not subject
to the sketch upload limit.

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
| `PROVIDER_RIGGING` | 1 — rendered image → rigged avatar | `Rigger` | returns the standard 16-bone figure |
| `PROVIDER_POSING` | 2 — prompt → pose | `Poser` | keyword-matches hand-authored clips |
| `PROVIDER_TRAINING` | 3–4 — train towards a pose | `Trainer` | hill-climber scored by `rewards.py` |

The UI status bar shows which are active. Adding one:
[docs/adding-a-provider.md](docs/adding-a-provider.md).

The real rigger calls the supplied ngrok service only from Flask:

```bash
RIGGING_SERVICE_URL=https://carie-spatterdashed-vella.ngrok-free.dev \
PROVIDER_RIGGING=real .venv/bin/flask --app app run
```

On a cache miss it classifies the rendered character image, generates two
T-pose augmentations, automatically confirms candidate A, and then runs mesh
generation, joint inference, and rigging. Existing completed rigs skip those
build steps.

`RIGGING_SERVICE_TIMEOUT` is the overall remote deadline (default 300 seconds)
including augmentation, and `RIGGING_POLL_INTERVAL` controls mesh/rig status
polling (default 5 seconds). Other vars: `EPISODE_RATE` (episodes/sec at speed
1, default 20), `DATA_DIR`, `MAX_UPLOAD_BYTES`.

## Bedrock prompt endpoint

`POST /api/llm/generate` runs a prompt against a Bedrock model for the teams
building the real providers. It costs money, so it is **off until configured**
and returns 404 to anyone without the token.

```bash
export LLM_API_TOKEN=$(python -c "import secrets; print(secrets.token_urlsafe(32))")
export BEDROCK_ALLOWED_MODELS="<model-id>,<model-id>"   # empty = refuse everything
export AWS_DEFAULT_REGION=eu-west-1

curl -s localhost:5000/api/llm/generate \
  -H "Authorization: Bearer $LLM_API_TOKEN" \
  -H 'Content-Type: application/json' \
  -d '{"model_id":"<model-id>","prompt":"Say hello","max_tokens":100}'
```

`GET /api/llm/models` lists what this deployment allows and your current
rate-limit usage. Find the model IDs for your account and region with
`aws bedrock list-inference-profiles --region "$AWS_DEFAULT_REGION"`.

**The token is server-side only — never ship it to the browser.** The frontend
does not call this endpoint. Details, including the threat model:
[docs/llm-endpoint.md](docs/llm-endpoint.md).

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
  js/viewport.js    preview + final Three.js rendering for both rig formats
  js/recorder.js    fixed-duration browser WebM capture
  js/components/    one per screen
  js/pipeline/      built line-extraction bundle (generated, see below)
  vendor/           browser-only WebM duration metadata helper
pipeline/           line-extraction source + tests (see below)
tests/test_contract.py
```

## Image pipeline

`static/js/components/StepSketch.js` runs every drawing/photo through a
browser-side OpenCV.js pipeline (`pipeline/src/pipeline.ts`'s
`extractLineDrawing()`) before sending it to `/api/renders` — perspective
correction, denoise, illumination normalization, binarize, stroke cleanup.
The rendered PNG returned by that endpoint is then sent to `/api/avatars`, so
the rigger receives the designed character shown in the preview.

This is the one piece of Node tooling in an otherwise build-free app,
kept isolated in `pipeline/` on purpose:

```bash
cd pipeline
npm install
npm test              # vitest — per-stage unit tests + a golden-image e2e test
npm run build         # bundles to ../static/js/pipeline/pipeline.js
```

`static/js/pipeline/pipeline.js` is a committed, vendored build artifact —
rebuild and commit it after editing anything under `pipeline/src/`.

## Not built

Real providers, user auth, persistence across restarts, and the environments
themselves. The Play screen currently previews the behaviour library that
future environments will draw on.
