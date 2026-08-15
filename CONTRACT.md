# Contract

Reference for the three provider integration points. Types in `schemas.py`,
signatures in `providers/base.py`, checks in `tests/test_contract.py`.

## Ownership

| Area | Owner | Touches | Leaves alone |
|---|---|---|---|
| Auto-rigging (phase 1) | rigging | `providers/real/rigging.py` | everything else |
| LLM pose generation (phase 2) | pose | `providers/real/posing.py` | everything else |
| Reinforcement learning (phases 3–4) | RL | `providers/real/training.py`, `rewards.py` (by proposal) | everything else |
| Glue, UI, contract | prototype | everything else | `providers/real/*` |

## Skeleton

16 bones. Defined in `schemas.BONE_TREE`, served at `GET /api/schema`.

```
root
└ hips ─┬ spine ─┬ neck ─ head
        │        ├ L_shoulder ─ L_elbow ─ L_hand
        │        └ R_shoulder ─ R_elbow ─ R_hand
        ├ L_hip ─ L_knee ─ L_foot
        └ R_hip ─ R_knee ─ R_foot
```

The set is closed: any other bone name raises `ContractError`.

11 of the 16 are articulated (`schemas.ARTICULATED_BONES`) — training moves only
these. Hands and feet follow their parents.

## Pose

Bone name → local rotation relative to rest, quaternion `[x, y, z, w]`.

```json
{ "L_shoulder": [0, 0, -0.64, 0.77], "R_shoulder": [0, 0, 0.64, 0.77] }
```

- Omitted bones are filled from rest.
- Quaternions are normalised on the way in.
- Build them with `rewards.quat_from_euler(x_deg, y_deg, z_deg)`.

### Rotation conventions

| Movement | Rotation |
|---|---|
| Raise left arm | `L_shoulder` z = -90 (straight up) |
| Raise right arm | `R_shoulder` z = +90 |
| Swing arms forward/back | about Y (+Y = left arm forward, right arm back) |
| Swing leg forward | negative X |
| Bend knee | positive X |
| Spread legs | `L_hip` z = -25, `R_hip` z = +25 |
| Lean forward | `spine` x = +30 |

Rotating an arm about X has no effect; the arm lies along the X axis.

## Clip

A single pose is a one-keyframe clip. Phases 2 and 3 use the same type.

```json
{
  "name": "Arms in the air",
  "fps": 24,
  "loop": true,
  "keyframes": [
    { "t": 0.0, "pose": {} },
    { "t": 0.5, "pose": {} }
  ]
}
```

`t` is seconds from start, strictly increasing. Limits: 240 keyframes, 30s.

## Rig

```json
{ "format": "procedural" | "glb", "skeleton": ["hips"], "glb_url": null }
```

- `procedural` — browser builds the figure from the bone tree; no asset needed.
- `glb` — browser loads a skinned mesh.

Both render through the same viewport path.

### GLB requirements

Verified against a synthetic fixture (`tests/fixtures/mixamo-style.glb`, 28KB,
committed) **and** a real Meshy export. Point the mock rigger at any GLB to
exercise the path:

```bash
MOCK_RIG_GLB=tests/fixtures/mixamo-style.glb flask --app app run
```

| Requirement | Why |
|---|---|
| **Bind in a T-pose** — arms out along ±X, legs down | A pose is defined as a rotation *away from bind*. An A-pose rig comes out with the arms wrong by the A/T difference. This is the one thing the browser cannot detect or correct. |
| Bone names either `schemas.BONES` or Mixamo convention | Resolved by contract name first, then `MIXAMO_BONE_MAP` |
| Any per-bone local axes | The browser reads each bone's bind rotation and retargets through world space, so `+Y`-down-the-limb rigs work unchanged |
| Any scale/units | Normalised to ~1.55 units tall on load; centimetre exports are fine |
| Baked animations | Ignored (no mixer is created). They'd fight the trainer for control of the bones |

### Two traps, both of which load cleanly and then move wrongly

**1. Mixamo's `LeftShoulder` is the clavicle.** The upper arm — what our
`L_shoulder` means — is `LeftArm`.

**2. The sides are mirrored, deliberately.** Our `L_*` bones sit at **−x**
(screen-left). Mixamo's `Left` is the character's **anatomical** left, which is
**+x** for a character facing +z — screen-right. So `L_*` maps to `Right*`.

Get this backwards and nothing looks broken: bones resolve, the mesh renders,
and then every arm raise drives the arm *downwards*, because a −90° z rotation
lifts a limb lying along −x and lowers one lying along +x.

We keep screen-relative sides because that is what the poses and rotation
conventions above are written against, and because a child reads "the left one"
as the one on the left of the screen. The cost: a pose meaning the character's
anatomical left plays on its right — invisible for symmetric moves like waving,
wrong for "raise your right hand".

The mapping is in `schemas.MIXAMO_BONE_MAP` and ships to the browser on both
`/api/schema` and every rig object.

**Bone names arrive mangled, variously.** A bone authored as `mixamorig:LeftArm`
has been seen as `mixamorigLeftArm` (three's `sanitizeNodeName` strips `:`),
`mixamorig_LeftArm` (FBX conversion), and `mixamorig_LeftArm_011` (exporter
index). The viewport normalises all of these; the real Meshy export tested was
the last form.

A Mixamo rig has ~65 bones to our 16. Unmapped bones (fingers, toes, Spine1/2)
stay at bind — fine, though torso bends read as stiffer since our single `spine`
drives only the first of three spine joints.

**Proportions are still shared, not per-rig.** `rewards.py` scores `arm_height`
and `symmetry` using `BONE_TREE` offsets, so a GLB with different limb lengths is
scored against a differently-proportioned body than the one on screen. Poses
themselves are rotation-only and unaffected. Fixing it means letting `Rig` carry
its own offsets — worth doing once real rigs derive proportions from the drawing.

## Episode

One per training episode, streamed over SSE.

```json
{ "episode": 47, "reward": 0.68, "best_reward": 0.71, "match": 0.55,
  "exploration": 0.32, "pose": {}, "per_joint_error": { "L_shoulder": 0.12 },
  "done": false, "note": "New best!" }
```

| Field | Meaning |
|---|---|
| `reward` | score of the pose tried this episode; depends on the slider weights |
| `best_reward` | high-water mark; never decreases |
| `match` | distance to target, 0 at the starting pose, 1 at the target |
| `per_joint_error` | 0..1 per articulated bone; drives the joint heat colours |
| `exploration` | 0..1, how large the random changes currently are |

`reward` and `match` differ whenever the weights reward something other than
copying the target.

## Training output

A finished run exports two artifacts. They have different audiences, and the
split matters — see the warning below.

| Endpoint | What it is | For |
|---|---|---|
| `GET /api/training/runs/<id>/export.glb` | The avatar's own model, node rotations baked to the learned end pose, with everything below embedded in `asset.extras.avatarTrainer` | Rendering, 3D tools, humans |
| `GET /api/training/runs/<id>/export.json` | Bones, start pose, end pose, provenance. Nothing else | The animation step |

Built by `export.py`; GLB surgery and the server-side retarget live in
`gltf.py`. GLB-format rigs only — a procedural avatar returns 409 with
"there's no model to export".

### The two poses

**Start is the T-pose**: every bone at identity. That is our rest pose *and* the
GLB's bind pose, so both skeletons agree at frame zero of any animation. It
needs no baking — it is the file's natural state.

**End is what the learner reached** — the last episode's pose. The hill-climber
only accepts improvements, so the last pose is also the best one; taking it from
the end of history means a run stopped early still exports what was on screen.

A GLB can hold only one static pose in its node rotations, so the file *is* the
end state and the start pose rides in `extras`.

### The document

```json
{
  "schema_version": 1,
  "run_id": "run_…", "avatar_id": "av_…",
  "skeleton": { "bones": [...], "articulated": [...],
                "hierarchy": {...}, "rest_offsets": {...},
                "pose_format": "bone name -> local rotation relative to rest, quaternion [x, y, z, w]" },
  "poses":   { "start": { "hips": [0,0,0,1], ... },
               "end":   { "hips": [0,0,0,1], "L_shoulder": [...], ... } },
  "target":  { "name": "Arms in the air", "prompt": "waves arms in the air" },
  "training": { "episodes_run": 200, "best_reward": 0.9, "match": 0.7,
                "reward_weights": {...} }
}
```

`export.glb` adds `node_map` (contract bone → the GLB node actually driven),
`unmapped_bones`, `bone_aliases` and `posed: "end"`. It is self-describing on
purpose: a consumer needs no other file to interpret the poses.

### ⚠️ Do not send the GLB to a language model

A real Meshy export is ~3.6MB — roughly 5M characters base64'd, essentially all
mesh, textures and skin weights that no model can act on. `export.json` is ~3KB
and carries every fact the animation step needs. A test pins it under 20KB.

### Next step: animation (not built)

Turning start → end into in-between keyframes is the same shape as an existing
interface:

```
(start pose, end pose, intent) -> Clip
```

`Clip` already exists and the viewport already plays it, so the natural home is
a fourth provider alongside `Rigger`/`Poser`/`Trainer` rather than a new
pipeline. Worth knowing before building it:

- **Feed it `export.json`, not the GLB.** The bone list, `pose_format` and
  `rest_offsets` in that document are what make a pose interpretable.
- **The rotation conventions above are not guessable.** Put them in the prompt,
  or the model will invent a sign convention and limbs will bend backwards.
- **Ask for a `Clip`.** `validate_clip()` then enforces the bone names,
  normalises the quaternions and rejects out-of-order keyframes for free — the
  same guard rail the LLM poser gets.
- **Baking that clip into the GLB as a glTF `animation`** (channels + samplers)
  is a separate, later job. Nothing currently writes one, and the viewport
  deliberately ignores baked animations because they fight `applyPose`.

### Known duplication

The world-space retarget exists **twice**: `gltf.pose_glb` (Python, for export)
and `#applyPoseGlb` in `static/js/viewport.js` (browser, for display). Same
maths, two implementations — change one and you must change the other.
`tests/test_export.py` pins the Python side and asserts the two normalisers
agree, but it cannot see the browser. The eventual fix is to compute poses
once, server-side, and have the viewport apply what it is handed.

## Interfaces

```python
class Rigger(ABC):                                    # phase 1
    def rig(self, image_bytes: bytes, mime: str, progress) -> Rig: ...

class Poser(ABC):                                     # phase 2
    def pose(self, prompt: str, rig: Rig, progress) -> Clip: ...

class Trainer(ABC):                                   # phases 3 & 4
    def train(self, rig: Rig, target: Clip, cfg: TrainConfig,
              stop: threading.Event) -> Iterator[Episode]: ...
```

## Rules

All checked by `tests/test_contract.py`.

1. Accept and return only `schemas.py` types.
2. Import from `schemas` and `rewards` only.
3. Fail with `ProviderError(user_message=...)`. `user_message` is shown verbatim
   to an 11-year-old; everything else stays on the server.
4. Call `progress(fraction, message)`.
5. Trainers check `stop.is_set()` between episodes and return promptly.
6. Trainers do not sleep — the server paces the stream.
7. Posers are deterministic: same prompt, same clip. Pin temperature to 0 or
   cache by prompt.
8. Score with `rewards.reward()`; the UI sliders are labelled with its terms.

## Swapping in a real provider

```bash
cp providers/mock/posing.py providers/real/posing.py   # rename MockPoser -> RealPoser
export PROVIDER_POSING=real
pytest                                                  # same suite, your code
```

The three switches are independent:

```bash
PROVIDER_POSING=real flask --app app run
```

`PROVIDER_RIGGING`, `PROVIDER_POSING`, `PROVIDER_TRAINING` each take `mock` or
`real`. The UI status bar shows which are active. Worked example in
`docs/adding-a-provider.md`.

## HTTP API

| Method | Path | Purpose |
|---|---|---|
| GET | `/api/schema` | bone tree, rest pose, reward labels, active providers |
| POST | `/api/avatars` | multipart `image` → job |
| POST | `/api/renders` | multipart `image`, `prompt` → job → rendered PNG |
| GET | `/api/avatars/<id>` | avatar record including rig |
| GET | `/api/avatars/<id>/image` | the original drawing |
| GET | `/api/avatars/<id>/glb` | GLB bytes when `format == "glb"` |
| POST | `/api/avatars/<id>/tpose` | job → forward-facing, T-pose, transparent PNG from the avatar's source image |
| POST | `/api/avatars/<id>/poses` | `{prompt}` → job → clip |
| GET | `/api/clips/<id>` | a clip |
| GET | `/api/jobs/<id>` | `{status, progress, message, result, error}` |
| POST | `/api/training/runs` | `{avatar_id, target_clip_id, config, speed}` |
| GET | `/api/training/runs/<id>/events` | SSE: `episode` events, then `end` |
| POST | `/api/training/runs/<id>/{stop,pause,resume,speed}` | control |
| GET | `/api/training/runs/<id>/export.glb` | posed GLB + embedded metadata |
| GET | `/api/training/runs/<id>/export.json` | poses + provenance, for the animation step |
| GET/POST | `/api/behaviours` | named behaviours |

- Rigging and pose generation return a job immediately; poll `/api/jobs/<id>`.
- Training uses SSE and replays from episode 1 on connect.
- Errors are always `{"error": "message safe to show a child"}`.
