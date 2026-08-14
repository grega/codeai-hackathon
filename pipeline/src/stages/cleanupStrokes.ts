import type { PipelineOptions } from "../types";

/**
 * Drops small noise specks (dust, JPEG artifacts) via connected-component
 * area filtering, then lightly closes/opens to
 * reconnect broken strokes and knock off single-pixel jaggies without
 * eroding line thickness. Input/output convention: 255 = line, 0 = background.
 */
export function cleanupStrokes(cv: any, src: any, options: Required<PipelineOptions>): any {
  const labels = new cv.Mat();
  const stats = new cv.Mat();
  const centroids = new cv.Mat();
  const numLabels = cv.connectedComponentsWithStats(src, labels, stats, centroids, 8, cv.CV_32S);

  const keep = new Uint8Array(numLabels);
  for (let label = 1; label < numLabels; label++) {
    const area = stats.intPtr(label, cv.CC_STAT_AREA)[0];
    keep[label] = area >= options.minComponentArea ? 1 : 0;
  }

  const despeckled = new cv.Mat(src.rows, src.cols, cv.CV_8UC1, new cv.Scalar(0));
  const labelData = labels.data32S as Int32Array;
  const outData = despeckled.data as Uint8Array;
  for (let i = 0; i < labelData.length; i++) {
    const label = labelData[i];
    if (label > 0 && keep[label]) {
      outData[i] = 255;
    }
  }

  labels.delete();
  stats.delete();
  centroids.delete();

  const closeKernel = cv.getStructuringElement(cv.MORPH_RECT, new cv.Size(3, 3));
  const closed = new cv.Mat();
  cv.morphologyEx(despeckled, closed, cv.MORPH_CLOSE, closeKernel);
  despeckled.delete();
  closeKernel.delete();

  const openKernel = cv.getStructuringElement(cv.MORPH_RECT, new cv.Size(2, 2));
  const opened = new cv.Mat();
  cv.morphologyEx(closed, opened, cv.MORPH_OPEN, openKernel);
  closed.delete();
  openKernel.delete();

  return opened;
}
