// Phase 1 — draw a character, get a rigged avatar back.

import { computed, onMounted, ref } from "vue";
import { api } from "../api.js";
import { setAvatar, state, goTo } from "../store.js";
import { AvatarCanvas } from "./AvatarCanvas.js";

export const StepSketch = {
  components: { AvatarCanvas },
  setup() {
    const canvas = ref(null);
    const busy = ref(false);
    const progress = ref(0);
    const message = ref("");
    const error = ref(null);
    const hasDrawing = ref(false);
    const uploadCleaned = ref(false);
    const renderPrompt = ref("");
    const renderBusy = ref(false);
    const renderProgress = ref(0);
    const renderMessage = ref("");
    const renderError = ref(null);
    const renderedImage = ref(null);
    let ctx = null;
    let drawing = false;

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
      ctx.strokeStyle = "#1e293b";
      ctx.lineWidth = 7;
    });

    function positionOf(event) {
      const rect = canvas.value.getBoundingClientRect();
      return {
        x: (event.clientX - rect.left) * (canvas.value.width / rect.width),
        y: (event.clientY - rect.top) * (canvas.value.height / rect.height),
      };
    }

    function start(event) {
      drawing = true;
      hasDrawing.value = true;
      uploadCleaned.value = false;
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

    async function cleanUpDrawing(blob) {
      message.value = "Cleaning up your drawing...";
      try {
        const { extractLineDrawing } = await import("/js/pipeline/pipeline.js");
        const { outputBlob } = await extractLineDrawing(blob);
        return { blob: outputBlob, cleaned: true };
      } catch (err) {
        // A pipeline bug shouldn't block Phase 1 — fall back to the raw capture.
        console.warn("Line-extraction pipeline failed, using raw drawing:", err);
        return { blob, cleaned: false };
      } finally {
        message.value = "";
      }
    }

    async function loadFile(event) {
      const file = event.target.files?.[0];
      if (!file) return;
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

    async function bringToLife() {
      busy.value = true;
      error.value = null;
      progress.value = 0;
      try {
        const raw = await new Promise((resolve) =>
          canvas.value.toBlob(resolve, "image/png"));
        const blob = uploadCleaned.value ? raw : (await cleanUpDrawing(raw)).blob;
        const avatar = await api.createAvatar(blob, (fraction, msg) => {
          progress.value = fraction;
          message.value = msg;
        });
        setAvatar(avatar);
        // A previous render belonged to the old drawing.
        renderedImage.value = null;
        renderError.value = null;
        goTo("pose");
      } catch (err) {
        error.value = err.message;
      } finally {
        busy.value = false;
      }
    }

    async function renderImage() {
      if (!state.avatar || renderBusy.value || !renderPrompt.value.trim()) return;
      renderBusy.value = true;
      renderError.value = null;
      renderProgress.value = 0;
      try {
        const result = await api.renderAvatar(state.avatar.id, renderPrompt.value,
          (fraction, msg) => { renderProgress.value = fraction; renderMessage.value = msg; });
        renderedImage.value = `data:image/${result.output_format};base64,${result.image_base64}`;
      } catch (err) {
        renderError.value = err.message;
      } finally {
        renderBusy.value = false;
      }
    }

    return {
      canvas, busy, progress, message, error, hasDrawing, state,
      renderPrompt, renderBusy, renderError, renderedImage,
      start, move, end, clear, loadFile, bringToLife, renderImage,
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
          <div class="row">
            <button class="ghost" @click="clear">Clear</button>
            <label class="ghost file">
              Upload a photo
              <input type="file" accept="image/*" :disabled="busy"
                     @change="loadFile" hidden>
            </label>
            <button class="primary" :disabled="!hasDrawing || busy"
                    @click="bringToLife">
              {{ busy ? 'Working…' : 'Bring it to life' }}
            </button>
          </div>

          <div v-if="busy" class="progress">
            <div class="progress-bar" :style="{ width: percent + '%' }"></div>
            <span>{{ message }}</span>
          </div>
          <p v-if="error" class="error">{{ error }}</p>
        </div>

        <div class="panel">
          <h4>Rendering prompt</h4>
          <div class="prompt-row">
            <input v-model="renderPrompt" type="text"
                   placeholder="e.g. a friendly robot with a cape, bold colors"
                   @keyup.enter="renderImage" :disabled="renderBusy">
            <button class="primary" :disabled="!state.avatar || renderBusy || !renderPrompt.trim()"
                    @click="renderImage">
              {{ renderBusy ? 'Rendering…' : 'Render' }}
            </button>
          </div>
          <p class="muted" v-if="!state.avatar">Bring your drawing to life
             first, then render it here.</p>
          <p class="muted" v-else>Sends your line drawing and this prompt to
             a Bedrock image model and shows what it renders.</p>

          <div v-if="renderBusy" class="progress">
            <div class="progress-bar" :style="{ width: renderPercent + '%' }"></div>
            <span>{{ renderMessage }}</span>
          </div>
          <p v-if="renderError" class="error">{{ renderError }}</p>

          <img v-if="renderedImage" :src="renderedImage" alt="Rendered avatar"
               class="rendered-image">

          <AvatarCanvas v-if="state.avatar" :rig="state.avatar.rig"
                        label="Your avatar" />
        </div>
      </div>
    </section>
  `,
};
