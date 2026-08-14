import { describe, expect, it } from "vitest";
import { loadOpenCv } from "../opencv-loader";
import { cleanupStrokes } from "../stages/cleanupStrokes";
import { DEFAULT_PIPELINE_OPTIONS } from "../types";

describe("cleanupStrokes", () => {
  it("removes small noise specks below minComponentArea", async () => {
    const cv = await loadOpenCv();
    const src = new cv.Mat(100, 100, cv.CV_8UC1, new cv.Scalar(0));

    // A single 1x1 speck (well below the default minComponentArea).
    src.ucharPtr(10, 10)[0] = 255;
    // A real 8x8 stroke blob (well above it).
    for (let y = 40; y < 48; y++) {
      for (let x = 40; x < 48; x++) {
        src.ucharPtr(y, x)[0] = 255;
      }
    }

    const result = cleanupStrokes(cv, src, DEFAULT_PIPELINE_OPTIONS);

    expect(result.ucharPtr(10, 10)[0]).toBe(0);
    expect(result.ucharPtr(44, 44)[0]).toBe(255);

    src.delete();
    result.delete();
  });

  it("preserves output dimensions", async () => {
    const cv = await loadOpenCv();
    const src = cv.Mat.zeros(60, 80, cv.CV_8UC1);

    const result = cleanupStrokes(cv, src, DEFAULT_PIPELINE_OPTIONS);

    expect(result.rows).toBe(60);
    expect(result.cols).toBe(80);

    src.delete();
    result.delete();
  });
});
