export interface PipelineOptions {
  /** Downscale the input if its longest edge exceeds this, in pixels. */
  maxDimension?: number;
  bilateralParams?: { d: number; sigmaColor: number; sigmaSpace: number };
  /** Fraction of image width used as the illumination-background kernel size. */
  illuminationKernelFraction?: number;
  /** Shift applied to the Otsu threshold; increase to keep fainter marks. */
  otsuThresholdOffset?: number;
  /** Connected components smaller than this (px^2) are dropped as noise. */
  minComponentArea?: number;
  /** Attempt to detect the paper's quadrilateral and warp it fronto-parallel. */
  attemptPerspectiveCorrection?: boolean;
  outputBackground?: "white" | "transparent";
}

export const DEFAULT_PIPELINE_OPTIONS: Required<PipelineOptions> = {
  maxDimension: 1600,
  bilateralParams: { d: 9, sigmaColor: 75, sigmaSpace: 75 },
  illuminationKernelFraction: 1 / 15,
  otsuThresholdOffset: 0,
  minComponentArea: 12,
  attemptPerspectiveCorrection: true,
  outputBackground: "transparent",
};

export function resolveOptions(options: PipelineOptions = {}): Required<PipelineOptions> {
  return {
    ...DEFAULT_PIPELINE_OPTIONS,
    ...options,
    bilateralParams: {
      ...DEFAULT_PIPELINE_OPTIONS.bilateralParams,
      ...options.bilateralParams,
    },
  };
}

export interface StageResult {
  debugLabel: string;
  mat: any;
}

export interface PipelineResult {
  outputBlob: Blob;
  outputSvg: string;
  stages: StageResult[];
}
