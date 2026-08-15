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
- `glb` — browser loads a skinned mesh; bone names must match `schemas.BONES`.

Both render through the same viewport path.

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
| GET | `/api/avatars/<id>` | avatar record including rig |
| GET | `/api/avatars/<id>/image` | the original drawing |
| GET | `/api/avatars/<id>/glb` | GLB bytes when `format == "glb"` |
| POST | `/api/avatars/<id>/tpose` | job → forward-facing, T-pose, transparent PNG, built from the last `/render` output (400 if nothing's been rendered yet) |
| POST | `/api/avatars/<id>/poses` | `{prompt}` → job → clip |
| GET | `/api/clips/<id>` | a clip |
| GET | `/api/jobs/<id>` | `{status, progress, message, result, error}` |
| POST | `/api/training/runs` | `{avatar_id, target_clip_id, config, speed}` |
| GET | `/api/training/runs/<id>/events` | SSE: `episode` events, then `end` |
| POST | `/api/training/runs/<id>/{stop,pause,resume,speed}` | control |
| GET/POST | `/api/behaviours` | named behaviours |

- Rigging and pose generation return a job immediately; poll `/api/jobs/<id>`.
- Training uses SSE and replays from episode 1 on connect.
- Errors are always `{"error": "message safe to show a child"}`.
