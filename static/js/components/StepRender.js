import {
  computed,
  nextTick,
  onBeforeUnmount,
  onMounted,
  ref,
} from "vue";

import { recordCanvas } from "../recorder.js";
import {
  refreshBehaviours,
  setActiveBehaviour,
  state,
} from "../store.js";
import {
  VIDEO_HEIGHT,
  VIDEO_WIDTH,
  Viewport,
} from "../viewport.js";

const PRESETS = [
  { id: "studio", label: "Clean studio" },
  { id: "spotlight", label: "Spotlight stage" },
  { id: "color-pop", label: "Color pop" },
];

const CAMERA_MOTIONS = [
  { id: "still", label: "Stay still" },
  { id: "orbit", label: "Sway around" },
  { id: "push-in", label: "Move closer" },
];

function loopingClip(clip) {
  return clip ? { ...clip, loop: true } : null;
}

function downloadFilename(name) {
  const safe = name.toLowerCase().replace(/[^a-z0-9]+/g, "-")
    .replace(/^-|-$/g, "");
  return `${safe || "my-avatar"}-video.webm`;
}

export const StepRender = {
  setup() {
    const canvas = ref(null);
    const resultDialog = ref(null);
    const ready = ref(false);
    const busy = ref(false);
    const paused = ref(false);
    const progress = ref(0);
    const error = ref(null);
    const preset = ref("studio");
    const cameraMotion = ref("orbit");
    const resultUrl = ref(null);
    const resultName = ref("my-avatar-video.webm");
    let viewport = null;

    const selected = computed(() =>
      state.activeBehaviour || state.behaviours[0] || null);
    const percent = computed(() => Math.round(progress.value * 100));

    onMounted(async () => {
      try {
        if (!state.behaviours.length) await refreshBehaviours();
        if (!state.activeBehaviour && state.behaviours[0]) {
          setActiveBehaviour(state.behaviours[0]);
        }

        viewport = new Viewport(canvas.value, { render: true });
        await viewport.setRig(state.avatar.rig);
        viewport.playClip(loopingClip(selected.value?.clip));
        viewport.setScenePreset(preset.value);
        viewport.setCameraMotion(cameraMotion.value);
        ready.value = true;
      } catch (err) {
        error.value = err.message;
      }
    });

    onBeforeUnmount(() => {
      viewport?.dispose();
      if (resultUrl.value) URL.revokeObjectURL(resultUrl.value);
    });

    function selectBehaviour(behaviour) {
      setActiveBehaviour(behaviour);
      paused.value = false;
      viewport?.setPaused(false);
      viewport?.playClip(loopingClip(behaviour.clip));
    }

    function selectPreset(id) {
      preset.value = id;
      viewport?.setScenePreset(id);
    }

    function selectCamera(id) {
      cameraMotion.value = id;
      viewport?.setCameraMotion(id);
    }

    function togglePaused() {
      paused.value = !paused.value;
      viewport?.setPaused(paused.value);
    }

    async function makeVideo() {
      if (!viewport || !selected.value || busy.value) return;
      busy.value = true;
      paused.value = false;
      progress.value = 0;
      error.value = null;
      viewport.setPaused(false);

      try {
        const blob = await recordCanvas(
          canvas.value,
          viewport,
          (value) => { progress.value = value; },
        );
        if (resultUrl.value) URL.revokeObjectURL(resultUrl.value);
        resultUrl.value = URL.createObjectURL(blob);
        resultName.value = downloadFilename(selected.value.name);
        await nextTick();
        resultDialog.value.showModal();
      } catch (err) {
        error.value = err.message || "The video could not be created.";
      } finally {
        busy.value = false;
      }
    }

    function closeResult() {
      resultDialog.value?.close();
    }

    return {
      CAMERA_MOTIONS,
      PRESETS,
      VIDEO_HEIGHT,
      VIDEO_WIDTH,
      busy,
      cameraMotion,
      canvas,
      closeResult,
      error,
      makeVideo,
      paused,
      percent,
      preset,
      progress,
      ready,
      resultDialog,
      resultName,
      resultUrl,
      selectBehaviour,
      selectCamera,
      selected,
      state,
      togglePaused,
      selectPreset,
    };
  },
  template: `
    <section class="step render-step">
      <header class="step-head">
        <h2>Render your avatar</h2>
        <p>Choose a saved move, set the stage, and make a five-second video.</p>
      </header>

      <div class="render-layout">
        <aside class="panel render-controls">
          <fieldset>
            <legend>Move</legend>
            <div class="render-option-list">
              <button v-for="behaviour in state.behaviours"
                      :key="behaviour.id"
                      :class="{ active: selected?.id === behaviour.id }"
                      @click="selectBehaviour(behaviour)">
                <strong>{{ behaviour.name }}</strong>
                <small>{{ behaviour.trained ? 'learned' : 'imagined' }}</small>
              </button>
            </div>
          </fieldset>

          <fieldset>
            <legend>Stage</legend>
            <div class="render-option-list">
              <button v-for="item in PRESETS"
                      :key="item.id"
                      :class="{ active: preset === item.id }"
                      @click="selectPreset(item.id)">
                <span class="render-swatch" :class="'swatch-' + item.id"
                      aria-hidden="true"></span>
                <strong>{{ item.label }}</strong>
              </button>
            </div>
          </fieldset>

          <fieldset>
            <legend>Camera</legend>
            <div class="render-segmented">
              <button v-for="item in CAMERA_MOTIONS"
                      :key="item.id"
                      :class="{ active: cameraMotion === item.id }"
                      @click="selectCamera(item.id)">
                {{ item.label }}
              </button>
            </div>
          </fieldset>
        </aside>

        <div class="render-workspace">
          <div class="render-stage">
            <canvas ref="canvas"
                    :width="VIDEO_WIDTH"
                    :height="VIDEO_HEIGHT"
                    aria-label="Final animated avatar preview"></canvas>
            <div v-if="!ready && !error" class="render-loading">Preparing preview…</div>
            <button class="render-preview-toggle"
                    :disabled="!ready || busy"
                    :title="paused ? 'Play preview' : 'Pause preview'"
                    :aria-label="paused ? 'Play preview' : 'Pause preview'"
                    @click="togglePaused">
              <span aria-hidden="true">{{ paused ? '▶' : 'Ⅱ' }}</span>
            </button>
            <div v-if="busy" class="render-recording">
              <strong>Recording {{ percent }}%</strong>
              <div class="render-progress">
                <span :style="{ width: percent + '%' }"></span>
              </div>
            </div>
          </div>

          <div class="render-bar">
            <div>
              <strong>5 second video</strong>
              <span>{{ VIDEO_WIDTH }} × {{ VIDEO_HEIGHT }} WebM</span>
            </div>
            <button class="primary"
                    :disabled="!ready || !selected || busy"
                    @click="makeVideo">
              {{ busy ? 'Recording…' : 'Make my video' }}
            </button>
          </div>
          <p v-if="error" class="error">{{ error }}</p>
        </div>
      </div>

      <dialog ref="resultDialog" class="render-result">
        <button class="dialog-close" aria-label="Close video"
                title="Close" @click="closeResult">×</button>
        <h3>Your video is ready</h3>
        <video v-if="resultUrl" :src="resultUrl"
               controls autoplay loop muted playsinline></video>
        <div class="row">
          <button class="ghost" @click="closeResult">Make another</button>
          <a class="button primary" :href="resultUrl" :download="resultName">
            Download video
          </a>
        </div>
      </dialog>
    </section>
  `,
};
