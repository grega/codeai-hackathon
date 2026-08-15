"""Generate the before/after export fixtures.

Produces, from a fixed seed so reruns are byte-identical and diffable:

    tests/fixtures/mixamo-style.glb            input  — T-pose, untouched
    tests/fixtures/mixamo-style-trained.glb    output — posed to the learned end
    tests/fixtures/mixamo-style-trained.json   the document the animation step gets

Run from the repo root:

    .venv/bin/python tools/make_export_fixtures.py

Point it at a different rig to make a local pair from a real Meshy export
(don't commit those — they run to megabytes):

    .venv/bin/python tools/make_export_fixtures.py tests/fixtures/spin-test.glb
"""

from __future__ import annotations

import json
import sys
import threading
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import config  # noqa: E402

#: Fixed so the fixture is reproducible. A different seed gives a different
#: (equally valid) learned pose, which would make every regeneration a diff.
SEED = 7
EPISODES = 300
PROMPT = "waves arms in the air"


def main(source: str) -> None:
    source_path = Path(source)
    if not source_path.is_absolute():
        source_path = ROOT / source_path
    if not source_path.exists():
        raise SystemExit(f"no such rig: {source_path}")

    # The mock rigger serves whichever GLB this points at.
    config.MOCK_RIG_GLB = str(source_path)

    import export
    import providers
    from schemas import TrainConfig, validate_clip, validate_rig
    from store import store
    from training import TrainingRun

    def quiet(fraction: float, message: str = "") -> None:
        pass

    rig = validate_rig(providers.get_rigger().rig(b"fixture", "image/png", quiet))
    avatar = store.add_avatar(rig)
    clip = store.add_clip(validate_clip(
        providers.get_poser().pose(PROMPT, rig, quiet)))

    # Train synchronously — no server, no threads, no SSE pacing.
    run = TrainingRun(avatar.id, clip, TrainConfig(episodes=EPISODES, seed=SEED))
    for episode in providers.get_trainer().train(
            rig, clip, run.cfg, threading.Event()):
        run._history.append(episode)
        run.best_reward = max(run.best_reward, episode.best_reward)

    # Deterministic ids, or every regeneration churns the metadata.
    run.id = "run_fixture"
    avatar.id = "av_fixture"

    data, document = export.build_glb(run, avatar, clip)

    stem = source_path.with_suffix("")
    glb_out = stem.with_name(stem.name + "-trained.glb")
    json_out = stem.with_name(stem.name + "-trained.json")
    glb_out.write_bytes(data)
    json_out.write_text(json.dumps(document, indent=2) + "\n")

    last = run._history[-1]
    print(f"input   {source_path.relative_to(ROOT)}  "
          f"{source_path.stat().st_size:,} bytes  (T-pose)")
    print(f"output  {glb_out.relative_to(ROOT)}  "
          f"{glb_out.stat().st_size:,} bytes  (posed: end)")
    print(f"doc     {json_out.relative_to(ROOT)}  "
          f"{json_out.stat().st_size:,} bytes")
    print(f"        {last.episode} episodes, best reward "
          f"{run.best_reward:.3f}, match {last.match:.1%}")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "tests/fixtures/mixamo-style.glb")
