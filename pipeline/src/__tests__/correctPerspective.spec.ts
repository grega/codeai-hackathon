import { describe, expect, it } from "vitest";
import { loadOpenCv } from "../opencv-loader";
import { correctPerspective } from "../stages/correctPerspective";
import { loadFixtureMat } from "./helpers/loadFixtureMat";
import { DEFAULT_PIPELINE_OPTIONS } from "../types";

describe("correctPerspective", () => {
  it("skips gracefully (returns input unchanged) when disabled", async () => {
    const cv = await loadOpenCv();
    const src = loadFixtureMat(cv, "plain-paper-good-light.png");

    const result = correctPerspective(cv, src, { ...DEFAULT_PIPELINE_OPTIONS, attemptPerspectiveCorrection: false });

    expect(result.rows).toBe(src.rows);
    expect(result.cols).toBe(src.cols);

    src.delete();
    result.delete();
  });

  it("warps a paper photographed at an angle against a contrasting desk background", async () => {
    const cv = await loadOpenCv();
    const src = loadFixtureMat(cv, "angled-crop.png");

    const result = correctPerspective(cv, src, DEFAULT_PIPELINE_OPTIONS);

    // The desk margin (visible border around the rotated paper) should be
    // cropped away, so the warped result should be smaller than the padded
    // input frame.
    expect(result.rows * result.cols).toBeLessThan(src.rows * src.cols);

    src.delete();
    result.delete();
  });
});
