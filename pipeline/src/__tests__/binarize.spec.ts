import { describe, expect, it } from "vitest";
import { loadOpenCv } from "../opencv-loader";
import { denoise } from "../stages/denoise";
import { normalizeIllumination } from "../stages/normalizeIllumination";
import { binarize } from "../stages/binarize";
import { loadFixtureMat, nonZeroFraction } from "./helpers/loadFixtureMat";
import { DEFAULT_PIPELINE_OPTIONS } from "../types";

describe("binarize", () => {
  it("produces a plausible dark-pixel fraction for a simple line drawing", async () => {
    const cv = await loadOpenCv();
    const src = loadFixtureMat(cv, "plain-paper-good-light.png");
    const gray = denoise(cv, src, DEFAULT_PIPELINE_OPTIONS);
    const normalized = normalizeIllumination(cv, gray, DEFAULT_PIPELINE_OPTIONS);

    const result = binarize(cv, normalized, DEFAULT_PIPELINE_OPTIONS);
    const fraction = nonZeroFraction(cv, result);

    expect(fraction).toBeGreaterThan(0.005);
    expect(fraction).toBeLessThan(0.2);

    src.delete();
    gray.delete();
    normalized.delete();
    result.delete();
  });

  it("keeps more foreground when otsuThresholdOffset is increased", async () => {
    const cv = await loadOpenCv();
    const src = loadFixtureMat(cv, "plain-paper-good-light.png");
    const gray = denoise(cv, src, DEFAULT_PIPELINE_OPTIONS);
    const normalized = normalizeIllumination(cv, gray, DEFAULT_PIPELINE_OPTIONS);

    const strict = binarize(cv, normalized, DEFAULT_PIPELINE_OPTIONS);
    const lenient = binarize(cv, normalized, { ...DEFAULT_PIPELINE_OPTIONS, otsuThresholdOffset: 40 });

    expect(nonZeroFraction(cv, lenient)).toBeGreaterThanOrEqual(nonZeroFraction(cv, strict));

    src.delete();
    gray.delete();
    normalized.delete();
    strict.delete();
    lenient.delete();
  });
});
