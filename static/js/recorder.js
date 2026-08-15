import {
  VIDEO_DURATION_MS,
  VIDEO_FPS,
} from "./viewport.js";

function chooseMimeType() {
  const candidates = [
    "video/webm;codecs=vp9",
    "video/webm;codecs=vp8",
    "video/webm",
  ];
  const match = candidates.find((type) => MediaRecorder.isTypeSupported(type));
  if (!match) {
    throw new Error(
      "This browser cannot record WebM video. Open the app in Chrome.",
    );
  }
  return match;
}

export function recordCanvas(canvas, viewport, onProgress) {
  if (!canvas.captureStream || typeof MediaRecorder === "undefined") {
    throw new Error(
      "This browser cannot record the preview. Open the app in Chrome.",
    );
  }

  const stream = canvas.captureStream(VIDEO_FPS);
  const mimeType = chooseMimeType();
  const recorder = new MediaRecorder(stream, {
    mimeType,
    videoBitsPerSecond: 8_000_000,
  });
  const chunks = [];

  return new Promise((resolve, reject) => {
    let progressFrame = 0;
    let stopTimer = 0;

    const stopTracks = () => {
      stream.getTracks().forEach((track) => track.stop());
    };

    const finishProgress = () => {
      cancelAnimationFrame(progressFrame);
      onProgress(1);
    };

    recorder.addEventListener("dataavailable", (event) => {
      if (event.data.size > 0) chunks.push(event.data);
    });

    recorder.addEventListener("error", () => {
      window.clearTimeout(stopTimer);
      finishProgress();
      viewport.stopCapture();
      stopTracks();
      reject(new Error("The recording stopped unexpectedly. Please try again."));
    });

    recorder.addEventListener("start", () => {
      const startedAt = performance.now();
      viewport.startCapture(startedAt);
      const updateProgress = () => {
        onProgress(
          Math.min((performance.now() - startedAt) / VIDEO_DURATION_MS, 1),
        );
        progressFrame = requestAnimationFrame(updateProgress);
      };
      updateProgress();
      stopTimer = window.setTimeout(() => recorder.stop(), VIDEO_DURATION_MS);
    }, { once: true });

    recorder.addEventListener("stop", () => {
      finishProgress();
      viewport.stopCapture();
      stopTracks();
      if (chunks.length === 0) {
        reject(new Error("No video was captured. Please try again."));
        return;
      }

      const rawBlob = new Blob(chunks, { type: mimeType });
      const fixDuration = window.ysFixWebmDuration;
      if (typeof fixDuration !== "function") {
        resolve(rawBlob);
        return;
      }
      fixDuration(rawBlob, VIDEO_DURATION_MS, { logger: false })
        .then(resolve, reject);
    }, { once: true });

    recorder.start(250);
  });
}
