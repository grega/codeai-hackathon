// Phase 2 — describe a move in words, get a clip of poses back.

import { ref } from "vue";
import { api } from "../api.js";
import { addClip, goTo, refreshBehaviours, state } from "../store.js";
import { AvatarCanvas } from "./AvatarCanvas.js";

const SUGGESTIONS = [
  "waves arms in the air",
  "waves hello",
  "jumps up high",
  "does a star jump",
  "walks along",
  "does a silly dance",
  "takes a bow",
];

export const StepPose = {
  components: { AvatarCanvas },
  setup() {
    const prompt = ref("waves arms in the air");
    const busy = ref(false);
    const progress = ref(0);
    const message = ref("");
    const error = ref(null);
    const saved = ref(null);

    async function generate() {
      if (!prompt.value.trim() || busy.value) return;
      busy.value = true;
      error.value = null;
      try {
        const clip = await api.createPose(state.avatar.id, prompt.value,
          (fraction, msg) => { progress.value = fraction; message.value = msg; });
        addClip(clip);
      } catch (err) {
        error.value = err.message;
      } finally {
        busy.value = false;
      }
    }

    async function saveAsBehaviour(clip) {
      // Untrained: this is the move as the model imagined it, not something the
      // avatar has learned yet. The playground marks the difference.
      await api.saveBehaviour({
        avatar_id: state.avatar.id, clip_id: clip.id, name: clip.name,
      });
      await refreshBehaviours();
      saved.value = clip.id;
      setTimeout(() => { saved.value = null; }, 1800);
    }

    return {
      prompt, busy, progress, message, error, state, saved,
      SUGGESTIONS, generate, saveAsBehaviour, goTo,
      select: (clip) => { state.activeClip = clip; },
      use: (text) => { prompt.value = text; generate(); },
    };
  },
  template: `
    <section class="step">
      <header class="step-head">
        <h2>Tell your avatar what to do</h2>
        <p>Describe a move in your own words. A language model turns your words
           into positions for each joint.</p>
      </header>

      <div class="split">
        <div class="panel">
          <div class="prompt-row">
            <input v-model="prompt" placeholder="e.g. waves arms in the air"
                   @keyup.enter="generate" :disabled="busy">
            <button class="primary" :disabled="busy || !prompt.trim()"
                    @click="generate">
              {{ busy ? 'Thinking…' : 'Make the move' }}
            </button>
          </div>

          <div class="chips">
            <button v-for="s in SUGGESTIONS" :key="s" class="chip"
                    :disabled="busy" @click="use(s)">{{ s }}</button>
          </div>

          <div v-if="busy" class="progress">
            <div class="progress-bar" :style="{ width: (progress*100) + '%' }"></div>
            <span>{{ message }}</span>
          </div>
          <p v-if="error" class="error">{{ error }}</p>

          <h3 v-if="state.clips.length">Moves you've made</h3>
          <ul class="clip-list">
            <li v-for="clip in state.clips" :key="clip.id"
                :class="{ active: state.activeClip?.id === clip.id }"
                @click="select(clip)">
              <div>
                <strong>{{ clip.name }}</strong>
                <small>{{ clip.keyframes.length }} poses ·
                       {{ clip.duration.toFixed(1) }}s</small>
              </div>
              <div class="clip-actions">
                <button class="ghost small" @click.stop="saveAsBehaviour(clip)">
                  {{ saved === clip.id ? 'Saved!' : 'Save' }}
                </button>
                <button class="primary small" @click.stop="select(clip); goTo('train')">
                  Train it
                </button>
              </div>
            </li>
          </ul>
        </div>

        <div class="panel">
          <AvatarCanvas :rig="state.avatar.rig" :clip="state.activeClip"
                        :label="state.activeClip ? state.activeClip.name : 'Waiting'" />
          <p class="muted" v-if="!state.activeClip">
            Make a move and it'll play here on a loop.
          </p>
          <p class="muted" v-else>
            This is what the model <em>imagined</em>. Your avatar hasn't learned
            it yet — that's the next step.
          </p>
        </div>
      </div>
    </section>
  `,
};
