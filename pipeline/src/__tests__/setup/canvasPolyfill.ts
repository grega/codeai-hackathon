// jsdom does not implement `ImageData`, but @techstark/opencv-js's `cv.imshow`
// constructs one internally when writing pixels to a canvas. Polyfill it with
// node-canvas's implementation for tests that exercise cv.imshow under jsdom.
import { ImageData as NodeCanvasImageData } from "canvas";

if (typeof globalThis.ImageData === "undefined") {
  // @ts-expect-error -- node-canvas's ImageData is structurally compatible with the DOM's.
  globalThis.ImageData = NodeCanvasImageData;
}
