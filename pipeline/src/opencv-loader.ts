// @techstark/opencv-js can resolve to a ready module, an unready module that
// still needs `onRuntimeInitialized`, or a Promise, depending on environment.
// This mirrors the package's own documented "Basic Usage" pattern.
import cvModule from "@techstark/opencv-js";

let cvPromise: Promise<any> | null = null;

export function loadOpenCv(): Promise<any> {
  if (!cvPromise) {
    cvPromise = resolveCv();
  }
  return cvPromise;
}

async function resolveCv(): Promise<any> {
  const mod: any = cvModule;

  if (mod instanceof Promise) {
    return mod;
  }
  if (mod.Mat) {
    return mod;
  }
  await new Promise<void>((resolve) => {
    mod.onRuntimeInitialized = () => resolve();
  });
  return mod;
}
