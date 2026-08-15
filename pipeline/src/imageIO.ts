export type PipelineInput = HTMLImageElement | ImageData | Blob;

export async function decodeToMat(cv: any, input: PipelineInput): Promise<any> {
  if (typeof ImageData !== "undefined" && input instanceof ImageData) {
    return cv.matFromImageData(input);
  }
  if (typeof Blob !== "undefined" && input instanceof Blob) {
    const imageElement = await blobToImageElement(input);
    return cv.imread(imageElement);
  }
  return cv.imread(input as HTMLImageElement);
}

async function blobToImageElement(blob: Blob): Promise<HTMLImageElement> {
  const url = URL.createObjectURL(blob);
  try {
    return await new Promise<HTMLImageElement>((resolve, reject) => {
      const img = new Image();
      img.onload = () => resolve(img);
      img.onerror = () => reject(new Error("Failed to decode image"));
      img.src = url;
    });
  } finally {
    URL.revokeObjectURL(url);
  }
}

/** Downscale `src` if its longest edge exceeds `maxDimension`. Always returns a new Mat. */
export function resizeIfNeeded(cv: any, src: any, maxDimension: number): any {
  const longestEdge = Math.max(src.rows, src.cols);
  if (longestEdge <= maxDimension) {
    return src.clone();
  }
  const scale = maxDimension / longestEdge;
  const dsize = new cv.Size(Math.round(src.cols * scale), Math.round(src.rows * scale));
  const resized = new cv.Mat();
  cv.resize(src, resized, dsize, 0, 0, cv.INTER_AREA);
  return resized;
}
