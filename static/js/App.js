import { computed } from "vue";
import { canEnter, goTo, state, STEPS } from "./store.js";
import { StepSketch } from "./components/StepSketch.js";
import { StepPose } from "./components/StepPose.js";
import { StepTrain } from "./components/StepTrain.js";
import { StepPlayground } from "./components/StepPlayground.js";
import { StepRender } from "./components/StepRender.js";

const COMPONENTS = {
  sketch: StepSketch,
  pose: StepPose,
  train: StepTrain,
  play: StepPlayground,
  render: StepRender,
};

export const App = {
  components: { StepSketch, StepPose, StepTrain, StepPlayground, StepRender },
  setup() {
    return {
      state, STEPS, goTo, canEnter,
      current: computed(() => COMPONENTS[state.step]),
      // Which parts are real and which are stand-ins. Visible at all times so
      // nobody demos a mock thinking it's the real thing.
      providerList: computed(() => Object.entries(state.providers)),
    };
  },
  template: `
    <div class="app">
      <header class="topbar">
        <h1>Train&nbsp;Your&nbsp;Avatar</h1>
        <nav class="steps">
          <button v-for="(s, i) in STEPS" :key="s.id"
                  class="step-tab"
                  :class="{ active: state.step === s.id, locked: !canEnter(s.id) }"
                  :disabled="!canEnter(s.id)"
                  @click="goTo(s.id)">
            <span class="num">{{ i + 1 }}</span>
            <span class="label">{{ s.label }}</span>
            <span class="hint">{{ s.hint }}</span>
          </button>
        </nav>

        <div class="tpose-status" v-if="state.avatar && state.tpose.status !== 'idle'">
          <span v-if="state.tpose.status === 'running'" class="muted">Posing avatar…</span>
          <a v-else-if="state.tpose.status === 'done'" class="button primary"
             :href="state.tpose.dataUrl" download="posed-avatar.png">
            Download posed avatar
          </a>
          <span v-else-if="state.tpose.status === 'error'" class="error"
                :title="state.tpose.error">
            Posed avatar failed
          </span>
        </div>
      </header>

      <main>
        <p v-if="state.error" class="error banner">{{ state.error }}</p>
        <component v-else-if="state.ready" :is="current" />
        <p v-else class="muted">Starting up…</p>
      </main>
    </div>
  `,
};
