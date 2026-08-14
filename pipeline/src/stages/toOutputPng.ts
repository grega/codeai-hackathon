import type { PipelineOptions } from "../types";

/**
 * Renders the final line mask (255 = line, 0 = background) to a PNG Blob.
 * Default: solid white background with dark lines (deterministic, avoids
 * anti-aliasing halo risk from compositing a transparent PNG in an unknown
 * downstream tool). Switching to a transparent-alpha output is isolated to
 * this one stage.
 */
export async function toOutputPng(cv: any, src: any, options: Required<PipelineOptions>): Promise<Blob> {
  let rgba: any;

  if (options.outputBackground === "transparent") {
    const zeros = cv.Mat.zeros(src.rows, src.cols, cv.CV_8UC1);
    const channels = new cv.MatVector();
    channels.push_back(zeros);
    channels.push_back(zeros);
    channels.push_back(zeros);
    channels.push_back(src);
    rgba = new cv.Mat();
    cv.merge(channels, rgba);
    zeros.delete();
    channels.delete();
  } else {
    const inverted = new cv.Mat();
    cv.bitwise_not(src, inverted);
    rgba = new cv.Mat();
    cv.cvtColor(inverted, rgba, cv.COLOR_GRAY2RGBA);
    inverted.delete();
  }

  const canvas = document.createElement("canvas");
  canvas.width = rgba.cols;
  canvas.height = rgba.rows;
  cv.imshow(canvas, rgba);
  rgba.delete();

  return new Promise<Blob>((resolve, reject) => {
    canvas.toBlob((blob) => {
      if (blob) {
        resolve(blob);
      } else {
        reject(new Error("Failed to encode output PNG"));
      }
    }, "image/png");
  });
}
