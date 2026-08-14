import type { PipelineOptions } from "../types";

/**
 * Converts to grayscale and smooths paper texture / sensor noise with a
 * bilateral filter, which (unlike a Gaussian blur) preserves the drawn
 * line edges while smoothing flat regions.
 */
export function denoise(cv: any, src: any, options: Required<PipelineOptions>): any {
  const gray = new cv.Mat();
  if (src.channels() === 1) {
    src.copyTo(gray);
  } else {
    cv.cvtColor(src, gray, cv.COLOR_RGBA2GRAY);
  }

  const { d, sigmaColor, sigmaSpace } = options.bilateralParams;
  const filtered = new cv.Mat();
  cv.bilateralFilter(gray, filtered, d, sigmaColor, sigmaSpace, cv.BORDER_DEFAULT);

  gray.delete();
  return filtered;
}
