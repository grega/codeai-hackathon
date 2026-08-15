import type { PipelineOptions } from "../types";

/**
 * Flattens shadows/uneven lighting by estimating the slow-varying background
 * illumination (a heavy morphological close, which blurs out thin dark lines
 * but preserves the lighting gradient) and dividing it out. This is what
 * makes a single global threshold viable downstream.
 */
export function normalizeIllumination(cv: any, src: any, options: Required<PipelineOptions>): any {
  const kernelSize = toOdd(Math.max(9, src.cols * options.illuminationKernelFraction));
  const kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, new cv.Size(kernelSize, kernelSize));

  const background = new cv.Mat();
  cv.morphologyEx(src, background, cv.MORPH_CLOSE, kernel);

  const srcFloat = new cv.Mat();
  const backgroundFloat = new cv.Mat();
  src.convertTo(srcFloat, cv.CV_32F);
  background.convertTo(backgroundFloat, cv.CV_32F);

  // Avoid divide-by-zero on pure-black background estimates.
  const epsilon = new cv.Mat(backgroundFloat.rows, backgroundFloat.cols, backgroundFloat.type(), new cv.Scalar(1));
  cv.add(backgroundFloat, epsilon, backgroundFloat);

  const normalizedFloat = new cv.Mat();
  cv.divide(srcFloat, backgroundFloat, normalizedFloat, 255);

  const normalized = new cv.Mat();
  normalizedFloat.convertTo(normalized, cv.CV_8U);

  kernel.delete();
  background.delete();
  srcFloat.delete();
  backgroundFloat.delete();
  epsilon.delete();
  normalizedFloat.delete();

  return normalized;
}

function toOdd(value: number): number {
  const rounded = Math.round(value);
  return rounded % 2 === 0 ? rounded + 1 : rounded;
}
