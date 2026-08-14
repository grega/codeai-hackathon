"""Mock rigger: ignores the drawing and returns the canonical procedural rig.

TODO(real): implement providers/real/rigging.py with a class `RealRigger`
subclassing providers.base.Rigger, satisfying `Rigger.rig()`. That is the only
function you need to provide. Return Rig(format="glb", glb_bytes=...) once you
have real geometry — the frontend already handles it.
"""

from __future__ import annotations

import time

from providers.base import Progress, Rigger
from schemas import BONES, ProviderError, Rig

#: Staged progress so the browser's loading states get exercised properly.
#: A real rigger's stages will differ; the point is that it reports *something*.
_STAGES = [
    (0.15, "Looking at your drawing..."),
    (0.40, "Finding the head, arms and legs..."),
    (0.70, "Building a skeleton..."),
    (0.92, "Connecting the joints..."),
]


class MockRigger(Rigger):
    def rig(self, image_bytes: bytes, mime: str, progress: Progress) -> Rig:
        if not image_bytes:
            raise ProviderError(
                "I didn't get a drawing. Try sketching something first!",
                detail="empty image payload")

        for fraction, message in _STAGES:
            progress(fraction, message)
            time.sleep(0.35)  # stand in for real work, keeps the UI honest

        return Rig(
            format="procedural",
            skeleton=list(BONES),
            notes=("Mock rig: your drawing was saved but not analysed. "
                   "Swap in a real rigger to build a mesh from it."),
        )
