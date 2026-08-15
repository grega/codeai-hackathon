import { describe, expect, it } from "vitest";
import { loadOpenCv } from "../opencv-loader";
import { denoise } from "../stages/denoise";
import { normalizeIllumination } from "../stages/normalizeIllumination";
import { loadFixtureMat } from "./helpers/loadFixtureMat";
import { DEFAULT_PIPELINE_OPTIONS } from "../types";

function halfMeans(cv: any, mat: any): [number, number] {
  const left = mat.roi(new cv.Rect(0, 0, Math.floor(mat.cols / 2), mat.rows));
  const right = mat.roi(new cv.Rect(Math.floor(mat.cols / 2), 0, Math.floor(mat.cols / 2), mat.rows));
  const leftMean = cv.mean(left)[0];
  const rightMean = cv.mean(right)[0];
  left.delete();
  right.delete();
  return [leftMean, rightMean];
}

describe("normalizeIllumination", () => {
  it("flattens a strong diagonal shadow gradient", async () => {
    const cv = await loadOpenCv();
    const src = loadFixtureMat(cv, "shadow-gradient.png");
    const gray = denoise(cv, src, DEFAULT_PIPELINE_OPTIONS);

    const [beforeLeft, beforeRight] = halfMeans(cv, gray);
    const beforeGap = Math.abs(beforeLeft - beforeRight);

    const normalized = normalizeIllumination(cv, gray, DEFAULT_PIPELINE_OPTIONS);
    const [afterLeft, afterRight] = halfMeans(cv, normalized);
    const afterGap = Math.abs(afterLeft - afterRight);

    expect(beforeGap).toBeGreaterThan(30);
    expect(afterGap).toBeLessThan(beforeGap * 0.5);

    src.delete();
    gray.delete();
    normalized.delete();
  });

  it("preserves dimensions and channel count", async () => {
    const cv = await loadOpenCv();
    const src = loadFixtureMat(cv, "plain-paper-good-light.png");
    const gray = denoise(cv, src, DEFAULT_PIPELINE_OPTIONS);

    const normalized = normalizeIllumination(cv, gray, DEFAULT_PIPELINE_OPTIONS);

    expect(normalized.rows).toBe(gray.rows);
    expect(normalized.cols).toBe(gray.cols);
    expect(normalized.channels()).toBe(1);

    src.delete();
    gray.delete();
    normalized.delete();
  });
});
