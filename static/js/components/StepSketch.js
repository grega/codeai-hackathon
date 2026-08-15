// Phase 1 — draw a character, get a rigged avatar back.

import { computed, onMounted, ref } from "vue";
import { api } from "../api.js";
import { generatePosedAvatar, setAvatar, goTo } from "../store.js";

export const StepSketch = {
  setup() {
    const canvas = ref(null);
    const busy = ref(false);
    const progress = ref(0);
    const message = ref("");
    const error = ref(null);
    const glbBusy = ref(false);
    const hasDrawing = ref(false);
    const uploadCleaned = ref(false);
    const renderPrompt = ref("");
    const renderBusy = ref(false);
    const renderProgress = ref(0);
    const renderMessage = ref("");
    const renderError = ref(null);
    const renderedImage = ref(null);
    const tool = ref("pen"); // "pen" | "eraser"
    const undoStack = ref([]);
    let ctx = null;
    let drawing = false;

    const PEN_WIDTH = 7;
    const ERASER_WIDTH = 28;
    const UNDO_HISTORY_LIMIT = 20;

    onMounted(() => {
      const el = canvas.value;
      // Fixed backing size keeps the exported image consistent whatever the
      // screen does; CSS scales it to fit.
      el.width = 640;
      el.height = 640;
      ctx = el.getContext("2d");
      ctx.fillStyle = "#ffffff";
      ctx.fillRect(0, 0, el.width, el.height);
      ctx.lineCap = "round";
      ctx.lineJoin = "round";
    });

    function pushUndoSnapshot() {
      const el = canvas.value;
      undoStack.value.push({
        imageData: ctx.getImageData(0, 0, el.width, el.height),
        hadDrawing: hasDrawing.value,
      });
      if (undoStack.value.length > UNDO_HISTORY_LIMIT) undoStack.value.shift();
    }

    function undo() {
      const entry = undoStack.value.pop();
      if (!entry) return;
      ctx.putImageData(entry.imageData, 0, 0);
      hasDrawing.value = entry.hadDrawing;
      uploadCleaned.value = false;
      clearRenderedImage();
    }

    function positionOf(event) {
      const rect = canvas.value.getBoundingClientRect();
      return {
        x: (event.clientX - rect.left) * (canvas.value.width / rect.width),
        y: (event.clientY - rect.top) * (canvas.value.height / rect.height),
      };
    }

    function start(event) {
      if (busy.value || renderBusy.value || glbBusy.value) return;
      pushUndoSnapshot();
      clearRenderedImage();
      drawing = true;
      hasDrawing.value = true;
      uploadCleaned.value = false;
      ctx.strokeStyle = tool.value === "eraser" ? "#ffffff" : "#1e293b";
      ctx.lineWidth = tool.value === "eraser" ? ERASER_WIDTH : PEN_WIDTH;
      const { x, y } = positionOf(event);
      ctx.beginPath();
      ctx.moveTo(x, y);
      canvas.value.setPointerCapture(event.pointerId);
    }

    function move(event) {
      if (!drawing) return;
      const { x, y } = positionOf(event);
      ctx.lineTo(x, y);
      ctx.stroke();
    }

    function end() { drawing = false; }

    function clear() {
      ctx.fillStyle = "#ffffff";
      ctx.fillRect(0, 0, canvas.value.width, canvas.value.height);
      hasDrawing.value = false;
      uploadCleaned.value = false;
      clearRenderedImage();
    }

    function clearCanvas() {
      pushUndoSnapshot();
      clear();
    }

    function drawBlobToCanvas(blob) {
      return new Promise((resolve, reject) => {
        const image = new Image();
        image.onload = () => {
          clear();
          // Fit the upload inside the square without distorting it.
          const scale = Math.min(canvas.value.width / image.width,
                                 canvas.value.height / image.height);
          const w = image.width * scale;
          const h = image.height * scale;
          ctx.drawImage(image, (canvas.value.width - w) / 2,
                        (canvas.value.height - h) / 2, w, h);
          hasDrawing.value = true;
          URL.revokeObjectURL(image.src);
          resolve();
        };
        image.onerror = () => reject(new Error("Failed to load the selected file"));
        image.src = URL.createObjectURL(blob);
      });
    }

    async function cleanUpDrawing(blob, report = (text) => { message.value = text; }) {
      report("Cleaning up your drawing...");
      try {
        const { extractLineDrawing } = await import("/js/pipeline/pipeline.js");
        const { outputBlob } = await extractLineDrawing(blob);
        return { blob: outputBlob, cleaned: true };
      } catch (err) {
        // A pipeline bug shouldn't block Phase 1 — fall back to the raw capture.
        console.warn("Line-extraction pipeline failed, using raw drawing:", err);
        return { blob, cleaned: false };
      } finally {
        report("");
      }
    }

    async function loadFile(event) {
      const file = event.target.files?.[0];
      if (!file || glbBusy.value) return;
      pushUndoSnapshot();
      busy.value = true;
      error.value = null;
      try {
        const { blob, cleaned } = await cleanUpDrawing(file);
        await drawBlobToCanvas(blob);
        // Skip cleaning it again on submit — the canvas already holds the
        // pipeline's output (or, if the pipeline failed, the raw upload).
        uploadCleaned.value = cleaned;
      } catch (err) {
        error.value = err.message;
      } finally {
        busy.value = false;
        event.target.value = "";
      }
    }

    async function loadGlb(event) {
      const file = event.target.files?.[0];
      if (!file || busy.value || renderBusy.value || glbBusy.value) return;
      glbBusy.value = true;
      error.value = null;
      try {
        const avatar = await api.sideloadAvatar(file);
        setAvatar(avatar);
        goTo("pose");
      } catch (err) {
        error.value = err.message;
      } finally {
        glbBusy.value = false;
        event.target.value = "";
      }
    }

    function clearRenderedImage() {
      renderedImage.value = null;
      renderError.value = null;
    }

    async function drawingBlob(report) {
      const raw = await new Promise((resolve) =>
        canvas.value.toBlob(resolve, "image/png"));
      return uploadCleaned.value ? raw : (await cleanUpDrawing(raw, report)).blob;
    }

    async function createAvatarFromDrawing(drawingBlob, onProgress) {
      const avatar = await api.createAvatar(drawingBlob, onProgress);
      setAvatar(avatar);
      // Fire-and-forget: tracked on the shared store so it survives
      // navigating on to "Teach" rather than blocking this step's caller.
      generatePosedAvatar();
      return avatar;
    }

    async function bringToLife() {
      if (busy.value || renderBusy.value || glbBusy.value) return;
      busy.value = true;
      error.value = null;
      progress.value = 0;
      try {
        const imageBlob = await drawingBlob(
          (text) => { message.value = text; });
        await createAvatarFromDrawing(imageBlob, (fraction, msg) => {
          progress.value = fraction;
          message.value = msg;
        });
        goTo("pose");
      } catch (err) {
        error.value = err.message;
      } finally {
        busy.value = false;
      }
    }

    async function renderImage() {
      if (renderBusy.value || busy.value || glbBusy.value
          || !renderPrompt.value.trim() || !hasDrawing.value) return;
      renderBusy.value = true;
      renderError.value = null;
      renderProgress.value = 0;
      renderedImage.value = null;
      try {
        const imageBlob = await drawingBlob(
          (text) => { renderMessage.value = text; });
        const result = await api.renderSketch(imageBlob, renderPrompt.value,
          (fraction, msg) => { renderProgress.value = fraction; renderMessage.value = msg; });
        renderedImage.value = `data:image/${result.output_format};base64,${result.image_base64}`;
      } catch (err) {
        renderError.value = err.message;
      } finally {
        renderBusy.value = false;
      }
    }

    return {
      canvas, busy, progress, message, error, hasDrawing,
      glbBusy,
      renderPrompt, renderBusy, renderError, renderedImage,
      tool, undoStack,
      start, move, end, clearCanvas, loadFile, loadGlb, bringToLife, renderImage, undo,
      percent: computed(() => Math.round(progress.value * 100)),
      renderPercent: computed(() => Math.round(renderProgress.value * 100)),
      renderMessage,
    };
  },
  template: `
    <section class="step">
      <header class="step-head">
        <h2>Draw your character</h2>
        <p>A stick person works perfectly. We'll build a skeleton from it that
           you can teach to move.</p>
      </header>

      <div class="split">
        <div class="panel">
          <canvas ref="canvas" class="sketchpad"
                  @pointerdown="start" @pointermove="move"
                  @pointerup="end" @pointerleave="end"></canvas>
          <div class="tool-toggle">
            <button :class="{ active: tool === 'pen' }" title="Pen"
                    aria-label="Pen" @click="tool = 'pen'">
              <svg viewBox="0 0 24 24" width="16" height="16" fill="none"
                   stroke="currentColor" stroke-width="2" stroke-linecap="round"
                   stroke-linejoin="round" aria-hidden="true">
                <path d="M21.174 6.812a1 1 0 0 0-3.986-3.987L3.842 16.174a2 2 0 0 0-.5.83l-1.321 4.352a.5.5 0 0 0 .622.622l4.353-1.32a2 2 0 0 0 .83-.497z"/>
                <path d="m15 5 4 4"/>
              </svg>
            </button>
            <button :class="{ active: tool === 'eraser' }" title="Eraser"
                    aria-label="Eraser" @click="tool = 'eraser'">
              <svg viewBox="0 0 24 24" width="16" height="16" fill="none"
                   stroke="currentColor" stroke-width="2" stroke-linecap="round"
                   stroke-linejoin="round" aria-hidden="true">
                <path d="m7 21-4.3-4.3c-1-1-1-2.5 0-3.4l9.6-9.6c1-1 2.5-1 3.4 0l5.6 5.6c1 1 1 2.5 0 3.4L13 21"/>
                <path d="M22 21H7"/>
                <path d="m5 11 9 9"/>
              </svg>
            </button>
          </div>
          <div class="row">
            <button class="ghost"
                    :disabled="!undoStack.length || busy || renderBusy || glbBusy"
                    @click="undo">Undo</button>
            <button class="ghost" :disabled="busy || renderBusy || glbBusy"
                    @click="clearCanvas">Clear</button>
            <label class="ghost file">
              Upload a photo
              <input type="file" accept="image/*"
                     :disabled="busy || renderBusy || glbBusy"
                     @change="loadFile" hidden>
            </label>
          </div>
        </div>

        <div class="panel">
          <h4>Rendering prompt</h4>
          <div class="prompt-row">
            <input v-model="renderPrompt" type="text"
                   placeholder="e.g. a friendly robot with a cape, bold colors"
                   @keyup.enter="renderImage"
                   :disabled="renderBusy || busy || glbBusy">
            <button class="primary"
                    :disabled="renderBusy || busy || glbBusy
                               || !renderPrompt.trim() || !hasDrawing"
                    @click="renderImage">
              {{ renderBusy ? 'Rendering…' : 'Render' }}
            </button>
          </div>
          <p class="muted" v-if="!hasDrawing">Draw something
             on the left first.</p>
          <p class="muted" v-else>Sends your line drawing and this prompt to
             a Bedrock image model and shows what it renders.</p>

          <div v-if="renderBusy" class="progress">
            <div class="progress-bar" :style="{ width: renderPercent + '%' }"></div>
            <span>{{ renderMessage }}</span>
          </div>
          <p v-if="renderError" class="error">{{ renderError }}</p>

          <img v-if="renderedImage" :src="renderedImage" alt="Rendered avatar"
               class="rendered-image">

          <button class="primary"
                  :disabled="!hasDrawing || busy || renderBusy || glbBusy"
                  @click="bringToLife">
            {{ busy ? 'Working…' : 'Bring it to life' }}
          </button>
          <p class="muted">Builds a rigged skeleton directly from your drawing,
             then opens Teach.</p>

          <div v-if="busy" class="progress">
            <div class="progress-bar" :style="{ width: percent + '%' }"></div>
            <span>{{ message }}</span>
          </div>

          <div class="sideload-row">
            <span class="muted">Already have a rigged model?</span>
            <label class="ghost file"
                   :class="{ disabled: busy || renderBusy || glbBusy }">
              {{ glbBusy ? 'Loading…' : 'Load GLB' }}
              <input type="file" accept=".glb,model/gltf-binary"
                     :disabled="busy || renderBusy || glbBusy"
                     @change="loadGlb" hidden>
            </label>
          </div>
          <p v-if="error" class="error">{{ error }}</p>
        </div>
      </div>
    </section>
  `,
};
