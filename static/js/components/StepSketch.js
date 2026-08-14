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
    }

    function loadFile(event) {
      const file = event.target.files?.[0];
      if (!file) return;
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
      };
      image.src = URL.createObjectURL(file);
    }

    async function cleanUpDrawing(blob) {
      message.value = "Cleaning up your drawing...";
      try {
        const { extractLineDrawing } = await import("/js/pipeline/pipeline.js");
        const { outputBlob } = await extractLineDrawing(blob);
        return outputBlob;
      } catch (err) {
        // A pipeline bug shouldn't block Phase 1 — fall back to the raw capture.
        console.warn("Line-extraction pipeline failed, using raw drawing:", err);
        return blob;
      }
    }

    async function bringToLife() {
      busy.value = true;
      error.value = null;
      progress.value = 0;
      try {
        const raw = await new Promise((resolve) =>
          canvas.value.toBlob(resolve, "image/png"));
        const blob = await cleanUpDrawing(raw);
        const avatar = await api.createAvatar(blob, (fraction, msg) => {
          progress.value = fraction;
          message.value = msg;
        });
        setAvatar(avatar);
        goTo("pose");
      } catch (err) {
        error.value = err.message;
      } finally {
        busy.value = false;
      }
    }

    return {
      canvas, busy, progress, message, error, hasDrawing, state,
      start, move, end, clear, loadFile, bringToLife,
      percent: computed(() => Math.round(progress.value * 100)),
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
              <input type="file" accept="image/*" @change="loadFile" hidden>
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
          <AvatarCanvas v-if="state.avatar" :rig="state.avatar.rig"
                        label="Your avatar" />
          <div v-else class="placeholder">
            <p>Your avatar will appear here once you've drawn something.</p>
            <p class="muted">Behind the scenes: your drawing goes to a model
               that works out where the head, arms and legs are, then fits a
               skeleton to them. That skeleton is what you'll train.</p>
          </div>
        </div>
      </div>
    </section>
  `,
};
