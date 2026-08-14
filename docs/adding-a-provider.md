# Adding a provider

Worked example: replacing the mock poser with one that calls an LLM. Rigging and
training follow the same three steps.

See `CONTRACT.md` for the types and rules.

## 1. Copy the mock

```bash
mkdir -p providers/real && touch providers/real/__init__.py
cp providers/mock/posing.py providers/real/posing.py
```

## 2. Rename the class and replace the body

The factory resolves `PROVIDER_POSING=real` to `providers.real.posing.RealPoser`,
so the class name must be `RealPoser`.

```python
# providers/real/posing.py
import json

from anthropic import Anthropic

from providers.base import Poser, Progress
from schemas import BONES, Clip, Keyframe, ProviderError, Rig

SYSTEM = f"""You pose a 3D stick figure for a children's learning app.
Reply with JSON only: {{"keyframes": [{{"t": 0.0, "bones": {{"BONE": [x,y,z,w]}}}}]}}

Valid bone names, and no others: {", ".join(BONES)}
Rotations are LOCAL quaternions relative to a rest pose where the figure stands
with arms out to the sides. Omit bones that do not move.
Left arm lies along -X, so raising it is a NEGATIVE z rotation (-90 is straight
up). Right arm is the mirror. Legs swing forward with a negative x rotation and
knees bend with a positive one.
Use 2-5 keyframes over 1-2 seconds."""


class RealPoser(Poser):
    def __init__(self):
        self.client = Anthropic()

    def pose(self, prompt: str, rig: Rig, progress: Progress) -> Clip:
        progress(0.2, "Reading your words...")

        try:
            message = self.client.messages.create(
                model="claude-sonnet-5",
                max_tokens=2000,
                temperature=0,          # rule 7: same prompt, same clip
                system=SYSTEM,
                messages=[{"role": "user", "content": prompt}],
            )
        except Exception as exc:
            # Rule 3: the child sees user_message, the terminal sees detail.
            raise ProviderError(
                "I couldn't think of that move right now. Try again?",
                detail=f"{type(exc).__name__}: {exc}") from exc

        progress(0.8, "Moving the joints...")

        try:
            data = json.loads(message.content[0].text)
            keyframes = [Keyframe(t=float(kf["t"]), pose=kf["bones"])
                         for kf in data["keyframes"]]
        except (KeyError, ValueError, TypeError) as exc:
            raise ProviderError(
                "That move came out muddled. Try describing it differently?",
                detail=f"unparseable model output: {exc}") from exc

        return Clip(name=prompt[:40], prompt=prompt, keyframes=keyframes,
                    loop=True)
```

You do not need to validate the pose yourself. `app.py` runs `validate_clip()`
on whatever you return, which normalises the quaternions, fills omitted bones
from rest, and raises `ContractError` naming the offending bone if the model
invents a name. That error is logged as a contract violation and shown to the
user as a generic message.

## 3. Switch it on and test

```bash
export PROVIDER_POSING=real
pytest tests/test_contract.py -k Poser
```

The suite is parameterised over whichever provider is configured, so this is the
same file that checks the mock. Expect to have to satisfy:

| Test | Requirement |
|---|---|
| `test_only_uses_contract_bones` | no invented bone names |
| `test_is_deterministic` | `temperature=0`, or cache by prompt |
| `test_keyframe_times_strictly_increase` | sort your keyframes |
| `test_unknown_prompt_still_returns_something` | handle nonsense without raising |
| `test_moves_the_avatar` | the pose differs from rest |

Then run the app and click through:

```bash
flask --app app run --debug
```

The status bar shows `posing: REAL` while rigging and training stay on mocks.

## Notes per provider

**Rigger.** Start by returning `Rig(format="procedural")` and ignoring the image
— that alone makes the pipeline run. Move to `Rig(format="glb", glb_bytes=...)`
when you have geometry; bone names in the GLB must match `schemas.BONES` and the
frontend needs no change.

**Trainer.** Yield episodes as fast as you compute them and do not sleep; the
server paces the stream so the speed control works. Check `stop.is_set()` between
episodes. Score with `rewards.reward()` so the UI sliders keep their meaning.
`rewards.world_positions()` gives you joint positions in world space if your
policy needs them.

## Failure modes

| Symptom | Cause |
|---|---|
| `RuntimeError` naming a missing file at startup | `PROVIDER_X=real` with no `providers/real/<name>.py` |
| `RuntimeError` about a missing class | class not named `Real<Rigger\|Poser\|Trainer>` |
| `CONTRACT VIOLATION` in the server log | return value failed `validate_*`; the message names the field |
| UI shows a friendly error, terminal shows detail | your `ProviderError` — working as intended |
