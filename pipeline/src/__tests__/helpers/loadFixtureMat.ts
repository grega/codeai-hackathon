import { readFileSync } from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";
import { PNG } from "pngjs";

const FIXTURES_DIR = path.join(path.dirname(fileURLToPath(import.meta.url)), "..", "fixtures");

export interface DecodedFixture {
  width: number;
  height: number;
  data: Uint8Array;
}

/** Decodes a fixture PNG into RGBA pixel data, bypassing any canvas/DOM APIs. */
export function loadFixtureImageData(name: string): DecodedFixture {
  const buffer = readFileSync(path.join(FIXTURES_DIR, name));
  const png = PNG.sync.read(buffer);
  return { width: png.width, height: png.height, data: png.data };
}

/** Builds a cv.Mat (RGBA) directly from a fixture, with no canvas involved. */
export function loadFixtureMat(cv: any, name: string): any {
  return cv.matFromImageData(loadFixtureImageData(name));
}

/** Fraction of pixels in a single-channel 8U Mat that are non-zero. */
export function nonZeroFraction(cv: any, mat: any): number {
  return cv.countNonZero(mat) / (mat.rows * mat.cols);
}
