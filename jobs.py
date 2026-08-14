"""A tiny thread-backed job runner.

Rigging and LLM pose generation are slow in reality, so both go through the
same job+poll pattern: POST returns a job id immediately, the browser polls
GET /api/jobs/<id> until status is "done" or "error". Building the UI against
this from day one means a real provider taking 20 seconds needs no UI change.

Training does NOT use this — it is a stream, not a job, so it uses SSE.
"""

from __future__ import annotations

import threading
import traceback
import uuid
from dataclasses import dataclass, field
from typing import Any, Callable

from schemas import ContractError, ProviderError

_JOB_LIMIT = 200


@dataclass
class Job:
    id: str
    status: str = "queued"        # queued | running | done | error
    progress: float = 0.0         # 0..1
    message: str = ""
    result: Any = None
    error: str | None = None      # safe to show a child
    detail: str | None = None     # server-side only, for the terminal

    def to_json(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "status": self.status,
            "progress": round(self.progress, 3),
            "message": self.message,
            "result": self.result,
            "error": self.error,
        }


class JobRunner:
    def __init__(self) -> None:
        self._jobs: dict[str, Job] = {}
        self._order: list[str] = []
        self._lock = threading.RLock()

    def get(self, job_id: str) -> Job | None:
        return self._jobs.get(job_id)

    def submit(self, fn: Callable[[Callable[[float, str], None]], Any], *,
               message: str = "Working...") -> Job:
        """Run ``fn(progress)`` on a worker thread.

        ``fn`` receives a ``progress(fraction, message)`` callback and returns a
        JSON-serialisable result. Raising ProviderError produces a child-safe
        error; anything else is caught and reported generically with the real
        traceback logged server-side.
        """
        job = Job(id=f"job_{uuid.uuid4().hex[:10]}", message=message)
        with self._lock:
            self._jobs[job.id] = job
            self._order.append(job.id)
            self._evict()

        def report(fraction: float, msg: str = "") -> None:
            job.progress = max(0.0, min(1.0, float(fraction)))
            if msg:
                job.message = msg

        def run() -> None:
            job.status = "running"
            try:
                job.result = fn(report)
                job.progress = 1.0
                job.status = "done"
            except ProviderError as exc:
                job.status = "error"
                job.error = exc.user_message
                job.detail = exc.detail
                print(f"[job {job.id}] provider error: {exc.detail or exc}")
            except ContractError as exc:
                # A provider returned something the contract forbids. This is a
                # bug in that provider, so name it loudly in the server log.
                job.status = "error"
                job.error = ("The avatar service sent back something I didn't "
                             "understand. Check the server log.")
                job.detail = str(exc)
                print(f"[job {job.id}] CONTRACT VIOLATION: {exc}")
            except Exception as exc:  # noqa: BLE001 - last line of defence
                job.status = "error"
                job.error = "Something went wrong. Try again?"
                job.detail = str(exc)
                print(f"[job {job.id}] unexpected error:\n{traceback.format_exc()}")

        threading.Thread(target=run, name=f"job-{job.id}", daemon=True).start()
        return job

    def _evict(self) -> None:
        while len(self._order) > _JOB_LIMIT:
            self._jobs.pop(self._order.pop(0), None)


runner = JobRunner()
