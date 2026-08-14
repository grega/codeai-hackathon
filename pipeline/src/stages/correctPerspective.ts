import type { PipelineOptions } from "../types";

/**
 * Detects the paper's quadrilateral outline and warps it fronto-parallel.
 * If no confident quad is found, returns the input unchanged rather than
 * forcing a bad warp.
 */
export function correctPerspective(cv: any, src: any, options: Required<PipelineOptions>): any {
  if (!options.attemptPerspectiveCorrection) {
    return src.clone();
  }

  const gray = new cv.Mat();
  cv.cvtColor(src, gray, cv.COLOR_RGBA2GRAY);

  const blurred = new cv.Mat();
  cv.GaussianBlur(gray, blurred, new cv.Size(5, 5), 0);

  const edges = new cv.Mat();
  cv.Canny(blurred, edges, 50, 150);

  const dilateKernel = cv.getStructuringElement(cv.MORPH_RECT, new cv.Size(3, 3));
  cv.dilate(edges, edges, dilateKernel);

  const contours = new cv.MatVector();
  const hierarchy = new cv.Mat();
  cv.findContours(edges, contours, hierarchy, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE);

  const imageArea = src.rows * src.cols;
  let bestQuad: any = null;
  let bestArea = 0;

  for (let i = 0; i < contours.size(); i++) {
    const contour = contours.get(i);
    const area = cv.contourArea(contour);
    if (area < imageArea * 0.6 || area <= bestArea) {
      contour.delete();
      continue;
    }

    const perimeter = cv.arcLength(contour, true);
    const approx = new cv.Mat();
    cv.approxPolyDP(contour, approx, 0.02 * perimeter, true);
    contour.delete();

    if (approx.rows === 4 && cv.isContourConvex(approx)) {
      if (bestQuad) bestQuad.delete();
      bestQuad = approx;
      bestArea = area;
    } else {
      approx.delete();
    }
  }

  gray.delete();
  blurred.delete();
  edges.delete();
  dilateKernel.delete();
  contours.delete();
  hierarchy.delete();

  if (!bestQuad) {
    return src.clone();
  }

  const corners = orderQuadCorners(bestQuad);
  bestQuad.delete();

  const outWidth = Math.round(Math.max(distance(corners[0], corners[1]), distance(corners[3], corners[2])));
  const outHeight = Math.round(Math.max(distance(corners[0], corners[3]), distance(corners[1], corners[2])));

  if (outWidth < 10 || outHeight < 10) {
    return src.clone();
  }

  const srcQuad = cv.matFromArray(4, 1, cv.CV_32FC2, corners.flat());
  const dstQuad = cv.matFromArray(4, 1, cv.CV_32FC2, [
    0, 0,
    outWidth - 1, 0,
    outWidth - 1, outHeight - 1,
    0, outHeight - 1,
  ]);

  const transform = cv.getPerspectiveTransform(srcQuad, dstQuad);
  const warped = new cv.Mat();
  cv.warpPerspective(src, warped, transform, new cv.Size(outWidth, outHeight));

  srcQuad.delete();
  dstQuad.delete();
  transform.delete();

  return warped;
}

function distance(a: number[], b: number[]): number {
  return Math.hypot(a[0] - b[0], a[1] - b[1]);
}

/** Orders the 4 corners of an approxPolyDP quad as top-left, top-right, bottom-right, bottom-left. */
function orderQuadCorners(approx: any): number[][] {
  const points: number[][] = [];
  for (let i = 0; i < 4; i++) {
    points.push([approx.data32S[i * 2], approx.data32S[i * 2 + 1]]);
  }
  const sums = points.map((p) => p[0] + p[1]);
  const diffs = points.map((p) => p[0] - p[1]);
  return [
    points[sums.indexOf(Math.min(...sums))],
    points[diffs.indexOf(Math.max(...diffs))],
    points[sums.indexOf(Math.max(...sums))],
    points[diffs.indexOf(Math.min(...diffs))],
  ];
}
