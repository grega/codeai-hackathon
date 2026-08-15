import { describe, expect, it } from "vitest";
import { loadOpenCv } from "../opencv-loader";
import { denoise } from "../stages/denoise";
import { loadFixtureMat } from "./helpers/loadFixtureMat";
import { DEFAULT_PIPELINE_OPTIONS } from "../types";

describe("denoise", () => {
  it("converts to single-channel grayscale of the same dimensions", async () => {
    const cv = await loadOpenCv();
    const src = loadFixtureMat(cv, "plain-paper-good-light.png");

    const result = denoise(cv, src, DEFAULT_PIPELINE_OPTIONS);

    expect(result.channels()).toBe(1);
    expect(result.rows).toBe(src.rows);
    expect(result.cols).toBe(src.cols);

    src.delete();
    result.delete();
  });

  it("reduces pixel-to-pixel noise while preserving strong edges", async () => {
    const cv = await loadOpenCv();
    // Synthetic 40x40 noisy patch: flat gray with random speckle plus a hard
    // vertical edge down the middle.
    const size = 40;
    const src = new cv.Mat(size, size, cv.CV_8UC1);
    for (let y = 0; y < size; y++) {
      for (let x = 0; x < size; x++) {
        const base = x < size / 2 ? 200 : 40;
        const noise = (x * 7 + y * 13) % 25;
        src.ucharPtr(y, x)[0] = Math.min(255, Math.max(0, base + noise - 12));
      }
    }
    const srcRgba = new cv.Mat();
    cv.cvtColor(src, srcRgba, cv.COLOR_GRAY2RGBA);

    const result = denoise(cv, srcRgba, DEFAULT_PIPELINE_OPTIONS);

    // Edge should still be sharp: left half stays bright, right half stays dark.
    const leftRoi = result.roi(new cv.Rect(0, 0, size / 2, size));
    const rightRoi = result.roi(new cv.Rect(size / 2, 0, size / 2, size));
    const leftMean = cv.mean(leftRoi)[0];
    const rightMean = cv.mean(rightRoi)[0];
    expect(leftMean - rightMean).toBeGreaterThan(80);

    leftRoi.delete();
    rightRoi.delete();
    src.delete();
    srcRgba.delete();
    result.delete();
  });
});
