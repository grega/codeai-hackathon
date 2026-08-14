import type { PipelineOptions } from "../types";

/**
 * Global Otsu threshold on the already illumination-normalized image.
 * Otsu (rather than adaptive thresholding) avoids the speckle artifacts
 * adaptive thresholding tends to introduce in large uniform regions, which
 * is safe here because normalizeIllumination already flattened lighting.
 * Output: 255 = line (foreground), 0 = background.
 */
export function binarize(cv: any, src: any, options: Required<PipelineOptions>): any {
  const otsuProbe = new cv.Mat();
  const otsuThreshold = cv.threshold(src, otsuProbe, 0, 255, cv.THRESH_BINARY_INV + cv.THRESH_OTSU);
  otsuProbe.delete();

  const finalThreshold = Math.min(255, Math.max(0, otsuThreshold + options.otsuThresholdOffset));
  const result = new cv.Mat();
  cv.threshold(src, result, finalThreshold, 255, cv.THRESH_BINARY_INV);

  return result;
}
