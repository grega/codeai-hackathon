"""One-line server logging.

Everything server-side goes through `log()` rather than `print()` so it is
flushed immediately. Python block-buffers stdout when it isn't a terminal, so
a plain print on a dyno sits in a buffer that a restart discards — which loses
precisely the lines worth having: rejected tokens, contract violations, AWS
errors. Flushing costs nothing at this volume.
"""

from __future__ import annotations

import sys


def log(message: str) -> None:
    print(message, file=sys.stdout, flush=True)
