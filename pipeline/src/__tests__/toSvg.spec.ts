import { describe, expect, it } from "vitest";
import { loadOpenCv } from "../opencv-loader";
import { toSvg } from "../stages/toSvg";

describe("toSvg", () => {
  it("produces an empty-path SVG for a blank mask", async () => {
    const cv = await loadOpenCv();
    const mask = cv.Mat.zeros(50, 50, cv.CV_8UC1);

    const svg = toSvg(cv, mask);

    expect(svg).toContain('viewBox="0 0 50 50"');
    expect(svg).toContain('d=""');

    mask.delete();
  });

  it("traces a filled square into a single closed subpath", async () => {
    const cv = await loadOpenCv();
    const mask = cv.Mat.zeros(50, 50, cv.CV_8UC1);
    const rect = new cv.Rect(10, 10, 20, 20);
    const roi = mask.roi(rect);
    roi.setTo(new cv.Scalar(255));
    roi.delete();

    const svg = toSvg(cv, mask);

    expect(svg).toContain("<path");
    expect(svg).toContain("fill-rule=\"evenodd\"");
    // Exactly one closed subpath for a single filled blob.
    expect(svg.match(/Z/g)?.length).toBe(1);
    expect(svg).toMatch(/M\d+ \d+/);

    mask.delete();
  });

  it("traces a ring (annulus) as two subpaths so the hole renders via evenodd", async () => {
    const cv = await loadOpenCv();
    const mask = cv.Mat.zeros(60, 60, cv.CV_8UC1);
    cv.circle(mask, new cv.Point(30, 30), 20, new cv.Scalar(255), 6);

    const svg = toSvg(cv, mask);

    // A stroked circle has an outer and inner contour -> two subpaths.
    expect(svg.match(/Z/g)?.length).toBe(2);

    mask.delete();
  });
});
