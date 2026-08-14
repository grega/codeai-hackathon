// @vitest-environment jsdom
import { describe, expect, it } from "vitest";
import { PNG } from "pngjs";
import { loadOpenCv } from "../opencv-loader";
import { toOutputPng } from "../stages/toOutputPng";
import { DEFAULT_PIPELINE_OPTIONS } from "../types";

async function decodePng(blob: Blob) {
  const buffer = Buffer.from(await blob.arrayBuffer());
  return PNG.sync.read(buffer);
}

describe("toOutputPng", () => {
  it("renders a white background with dark lines when outputBackground is white", async () => {
    const cv = await loadOpenCv();
    const mask = cv.Mat.zeros(20, 20, cv.CV_8UC1);
    // A 255 (line) pixel at (5, 5); everywhere else is 0 (background).
    mask.ucharPtr(5, 5)[0] = 255;

    const blob = await toOutputPng(cv, mask, { ...DEFAULT_PIPELINE_OPTIONS, outputBackground: "white" });
    expect(blob.type).toBe("image/png");

    const png = await decodePng(blob);
    expect(png.width).toBe(20);
    expect(png.height).toBe(20);

    const idx = (y: number, x: number) => (y * png.width + x) * 4;
    const bgOffset = idx(0, 0);
    const lineOffset = idx(5, 5);

    expect(png.data[bgOffset]).toBe(255); // white background
    expect(png.data[lineOffset]).toBe(0); // dark line

    mask.delete();
  });

  it("renders a transparent background by default", async () => {
    const cv = await loadOpenCv();
    const mask = cv.Mat.zeros(10, 10, cv.CV_8UC1);
    mask.ucharPtr(2, 2)[0] = 255;

    expect(DEFAULT_PIPELINE_OPTIONS.outputBackground).toBe("transparent");
    const blob = await toOutputPng(cv, mask, DEFAULT_PIPELINE_OPTIONS);
    const png = await decodePng(blob);

    const idx = (y: number, x: number) => (y * png.width + x) * 4;
    const bgAlpha = png.data[idx(0, 0) + 3];
    const lineAlpha = png.data[idx(2, 2) + 3];

    expect(bgAlpha).toBe(0);
    expect(lineAlpha).toBe(255);

    mask.delete();
  });
});
