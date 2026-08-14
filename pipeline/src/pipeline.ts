import { loadOpenCv } from "./opencv-loader";
import { decodeToMat, resizeIfNeeded } from "./imageIO";
import type { PipelineInput } from "./imageIO";
import { correctPerspective } from "./stages/correctPerspective";
import { denoise } from "./stages/denoise";
import { normalizeIllumination } from "./stages/normalizeIllumination";
import { binarize } from "./stages/binarize";
import { cleanupStrokes } from "./stages/cleanupStrokes";
import { toOutputPng } from "./stages/toOutputPng";
import { toSvg } from "./stages/toSvg";
import { resolveOptions } from "./types";
import type { PipelineOptions, PipelineResult, StageResult } from "./types";

export type { PipelineInput } from "./imageIO";
export type { PipelineOptions, PipelineResult, StageResult } from "./types";

/**
 * Extracts a clean line drawing from a photo of paper (or a canvas drawing).
 *
 * Memory convention: each stage function receives a Mat it does not own
 * (the caller deletes it) and returns a new Mat the caller now owns.
 */
export async function extractLineDrawing(
  input: PipelineInput,
  options: PipelineOptions = {}
): Promise<PipelineResult> {
  const cv = await loadOpenCv();
  const resolved = resolveOptions(options);
  const stages: StageResult[] = [];

  const decoded = await decodeToMat(cv, input);
  let current = resizeIfNeeded(cv, decoded, resolved.maxDimension);
  decoded.delete();
  stages.push({ debugLabel: "resized", mat: current.clone() });

  current = advance(current, correctPerspective(cv, current, resolved));
  stages.push({ debugLabel: "perspective", mat: current.clone() });

  current = advance(current, denoise(cv, current, resolved));
  stages.push({ debugLabel: "denoise", mat: current.clone() });

  current = advance(current, normalizeIllumination(cv, current, resolved));
  stages.push({ debugLabel: "illumination", mat: current.clone() });

  current = advance(current, binarize(cv, current, resolved));
  stages.push({ debugLabel: "binarized", mat: current.clone() });

  current = advance(current, cleanupStrokes(cv, current, resolved));
  stages.push({ debugLabel: "cleaned", mat: current.clone() });

  const outputSvg = toSvg(cv, current);
  const outputBlob = await toOutputPng(cv, current, resolved);
  current.delete();

  return { outputBlob, outputSvg, stages };
}

function advance(previous: any, next: any): any {
  previous.delete();
  return next;
}
