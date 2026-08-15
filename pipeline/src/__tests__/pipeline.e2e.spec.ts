// @vitest-environment jsdom
import { readFileSync, writeFileSync, mkdirSync, existsSync } from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";
import { describe, expect, it } from "vitest";
import { PNG } from "pngjs";
import pixelmatch from "pixelmatch";
import { extractLineDrawing } from "../pipeline";
import type { PipelineOptions } from "../pipeline";
import { loadFixtureImageData } from "./helpers/loadFixtureMat";

const FIXTURES_DIR = path.join(path.dirname(fileURLToPath(import.meta.url)), "fixtures");
const GOLDEN_DIR = path.join(FIXTURES_DIR, "golden");
const UPDATE_GOLDENS = process.env.UPDATE_GOLDENS === "1";

const FIXTURES = [
  "plain-paper-good-light.png",
  "shadow-gradient.png",
  "graph-paper.png",
  "lined-paper.png",
  "angled-crop.png",
];

// Feeds an ImageData (rather than a Blob/HTMLImageElement) so this test
// exercises the full stage pipeline without depending on jsdom's <img>
// decoding, which jsdom does not implement.
async function runPipelineOnFixture(name: string, options: PipelineOptions = {}) {
  const decoded = loadFixtureImageData(name);
  const imageData = new ImageData(new Uint8ClampedArray(decoded.data), decoded.width, decoded.height);
  const { outputBlob, outputSvg, stages } = await extractLineDrawing(imageData, options);
  stages.forEach((stage) => stage.mat.delete());
  return { outputBlob, outputSvg };
}

describe("extractLineDrawing (full pipeline)", () => {
  it.each(FIXTURES)("produces a stable output for %s", async (name) => {
    const { outputBlob } = await runPipelineOnFixture(name);
    const outputBuffer = Buffer.from(await outputBlob.arrayBuffer());
    const outputPng = PNG.sync.read(outputBuffer);

    const goldenPath = path.join(GOLDEN_DIR, name);

    if (UPDATE_GOLDENS || !existsSync(goldenPath)) {
      mkdirSync(GOLDEN_DIR, { recursive: true });
      writeFileSync(goldenPath, outputBuffer);
      return;
    }

    const goldenPng = PNG.sync.read(readFileSync(goldenPath));
    expect(outputPng.width).toBe(goldenPng.width);
    expect(outputPng.height).toBe(goldenPng.height);

    const diff = new PNG({ width: outputPng.width, height: outputPng.height });
    const differingPixels = pixelmatch(
      outputPng.data,
      goldenPng.data,
      diff.data,
      outputPng.width,
      outputPng.height,
      { threshold: 0.1 }
    );

    const totalPixels = outputPng.width * outputPng.height;
    expect(differingPixels / totalPixels).toBeLessThan(0.01);
  });

  it("produces a mostly-white PNG with a plausible fraction of dark line pixels when outputBackground is white", async () => {
    const { outputBlob } = await runPipelineOnFixture("plain-paper-good-light.png", { outputBackground: "white" });
    const outputBuffer = Buffer.from(await outputBlob.arrayBuffer());
    const outputPng = PNG.sync.read(outputBuffer);

    let darkPixels = 0;
    const totalPixels = outputPng.width * outputPng.height;
    for (let i = 0; i < outputPng.data.length; i += 4) {
      if (outputPng.data[i] < 128) darkPixels++;
    }

    const darkFraction = darkPixels / totalPixels;
    expect(darkFraction).toBeGreaterThan(0.005);
    expect(darkFraction).toBeLessThan(0.2);
  });

  it("defaults to a transparent PNG background with a plausible fraction of opaque line pixels", async () => {
    const { outputBlob } = await runPipelineOnFixture("plain-paper-good-light.png");
    const outputBuffer = Buffer.from(await outputBlob.arrayBuffer());
    const outputPng = PNG.sync.read(outputBuffer);

    let opaquePixels = 0;
    const totalPixels = outputPng.width * outputPng.height;
    for (let i = 0; i < outputPng.data.length; i += 4) {
      if (outputPng.data[i + 3] > 128) opaquePixels++;
    }

    const opaqueFraction = opaquePixels / totalPixels;
    expect(opaqueFraction).toBeGreaterThan(0.005);
    expect(opaqueFraction).toBeLessThan(0.2);
  });

  it("produces a non-empty SVG with path data", async () => {
    const { outputSvg } = await runPipelineOnFixture("plain-paper-good-light.png");

    expect(outputSvg).toContain("<svg");
    expect(outputSvg).toContain("<path");
    const match = outputSvg.match(/d="([^"]*)"/);
    expect(match?.[1]?.length ?? 0).toBeGreaterThan(0);
  });
});
