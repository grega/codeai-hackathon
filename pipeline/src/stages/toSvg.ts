/**
 * Traces the binary line mask (255 = line, 0 = background) into an SVG
 * string via contour detection, so the drawing can be exported as a
 * resolution-independent vector alongside the raster PNG.
 *
 * Uses RETR_CCOMP so ring-shaped strokes (a hand-drawn circle is a thin
 * annulus, not a filled disk) produce both an outer and an inner contour;
 * combining all contours into one path with fill-rule="evenodd" renders
 * the hollow centers correctly instead of solid blobs. approxPolyDP
 * simplifies each contour so the path isn't one point per pixel.
 */
export function toSvg(cv: any, src: any): string {
  const contours = new cv.MatVector();
  const hierarchy = new cv.Mat();
  cv.findContours(src, contours, hierarchy, cv.RETR_CCOMP, cv.CHAIN_APPROX_SIMPLE);

  const subpaths: string[] = [];
  for (let i = 0; i < contours.size(); i++) {
    const contour = contours.get(i);
    if (contour.rows < 3) {
      contour.delete();
      continue;
    }

    const perimeter = cv.arcLength(contour, true);
    const approx = new cv.Mat();
    cv.approxPolyDP(contour, approx, Math.max(1, perimeter * 0.002), true);
    contour.delete();

    if (approx.rows >= 2) {
      const commands: string[] = [];
      for (let p = 0; p < approx.rows; p++) {
        const x = approx.data32S[p * 2];
        const y = approx.data32S[p * 2 + 1];
        commands.push(`${p === 0 ? "M" : "L"}${x} ${y}`);
      }
      subpaths.push(`${commands.join(" ")} Z`);
    }
    approx.delete();
  }

  contours.delete();
  hierarchy.delete();

  const pathData = subpaths.join(" ");
  return (
    `<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 ${src.cols} ${src.rows}" ` +
    `width="${src.cols}" height="${src.rows}"><path d="${pathData}" fill="#000000" fill-rule="evenodd"/></svg>`
  );
}
