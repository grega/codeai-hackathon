// Every call to the backend lives here, mirroring CONTRACT.md one-to-one.
// Nothing else in the frontend calls fetch().

class ApiError extends Error {}

async function request(url, options = {}) {
  let response;
  try {
    response = await fetch(url, options);
  } catch (cause) {
    throw new ApiError("Can't reach the server. Is it still running?", { cause });
  }

  let body = null;
  try {
    body = await response.json();
  } catch {
    // Non-JSON response (a proxy error page, say) — fall through to the status.
  }

  if (!response.ok) {
    throw new ApiError(body?.error || `Something went wrong (${response.status}).`);
  }
  return body;
}

const json = (body) => ({
  method: "POST",
  headers: { "Content-Type": "application/json" },
  body: JSON.stringify(body),
});

/**
 * Poll a job until it finishes. Rigging and pose generation both use this, so
 * a real provider that takes 30 seconds needs no UI change — only the progress
 * messages it reports will differ.
 */
async function waitForJob(job, onProgress, { interval = 350 } = {}) {
  let current = job;
  while (current.status === "queued" || current.status === "running") {
    onProgress?.(current.progress, current.message);
    await new Promise((resolve) => setTimeout(resolve, interval));
    current = await request(`/api/jobs/${current.id}`);
  }
  if (current.status === "error") throw new ApiError(current.error);
  onProgress?.(1, "Done!");
  return current.result;
}

export const api = {
  ApiError,

  schema: () => request("/api/schema"),

  // -- phase 1 --------------------------------------------------------
  async createAvatar(imageBlob, onProgress) {
    const form = new FormData();
    form.append("image", imageBlob, "sketch.png");
    const job = await request("/api/avatars", { method: "POST", body: form });
    return waitForJob(job, onProgress);
  },

  getAvatar: (id) => request(`/api/avatars/${id}`),

  // -- phase 2 --------------------------------------------------------
  async createPose(avatarId, prompt, onProgress) {
    const job = await request(`/api/avatars/${avatarId}/poses`, json({ prompt }));
    return waitForJob(job, onProgress);
  },

  getClip: (id) => request(`/api/clips/${id}`),

  // -- phases 3 & 4 ---------------------------------------------------
  startRun: (avatarId, targetClipId, config, speed = 1) =>
    request("/api/training/runs",
      json({ avatar_id: avatarId, target_clip_id: targetClipId, config, speed })),

  runControl: (runId, action, body = {}) =>
    request(`/api/training/runs/${runId}/${action}`, json(body)),

  /**
   * Subscribe to a run's episode stream. Returns a close() function.
   *
   * The stream always replays from episode 1, so attaching a moment after the
   * run starts still draws a complete learning curve.
   */
  streamRun(runId, { onEpisode, onEnd, onError }) {
    const source = new EventSource(`/api/training/runs/${runId}/events`);

    // EventSource reconnects on its own, and the server replays from episode 1,
    // so without this guard a dropped connection redraws the whole curve on top
    // of itself. Only shows up behind a proxy that closes idle connections.
    let lastEpisode = 0;
    source.addEventListener("episode", (e) => {
      const episode = JSON.parse(e.data);
      if (episode.episode <= lastEpisode) return;
      lastEpisode = episode.episode;
      onEpisode?.(episode);
    });
    source.addEventListener("end", (e) => {
      onEnd?.(JSON.parse(e.data));
      source.close();
    });
    source.onerror = () => {
      // EventSource retries by itself; only surface a failure once it has
      // actually given up, otherwise a normal reconnect looks like an error.
      if (source.readyState === EventSource.CLOSED) {
        onError?.(new ApiError("Lost contact with the training run."));
      }
    };
    return () => source.close();
  },

  // -- behaviours -----------------------------------------------------
  listBehaviours: (avatarId) =>
    request(`/api/behaviours?avatar_id=${encodeURIComponent(avatarId)}`),

  saveBehaviour: (body) => request("/api/behaviours", json(body)),
};
