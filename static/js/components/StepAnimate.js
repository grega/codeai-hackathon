// Phase 5 — Animate.
//
// Training produces two still poses: the T-pose the avatar started from and
// the one it learned. This step asks a language model to invent the movement
// between them, and plays the result.
//
// The two GLBs never come near the browser. The server already holds the rig
// and can bake the learned pose itself, so this posts a prompt and gets back a
// URL — rather than downloading ~7MB and immediately uploading it again.

import { computed, onMounted, ref } from "vue";
import { api } from "../api.js";
import { goTo, refreshBehaviours, state } from "../store.js";
import { AvatarCanvas } from "./AvatarCanvas.js";

const SUGGESTIONS = [
  "smoothly, like a dancer",
  "quickly and suddenly",
  "bouncy and playful",
  "slowly, like underwater",
];

export const StepAnimate = {
  components: { AvatarCanvas },
  setup() {
    const selected = ref(null);
    const style = ref("");
    const busy = ref(false);
    const progress = ref(0);
    const message = ref("");
    const error = ref(null);
    const result = ref(null);

    onMounted(refreshBehaviours);

    // Only a trained behaviour has a run behind it, and only a run can be
    // animated — an imagined move has no "before" to move from.
    const animatable = computed(() =>
      state.behaviours.filter((b) => b.trained && b.run_id));
    const imagined = computed(() =>
      state.behaviours.filter((b) => !b.trained || !b.run_id));

    function choose(behaviour) {
      selected.value = behaviour;
      result.value = null;
      error.value = null;
    }

    async function animate() {
      if (!selected.value || busy.value) return;
      busy.value = true;
      error.value = null;
      result.value = null;
      try {
        result.value = await api.animateRun(
          selected.value.run_id, style.value.trim(),
          (fraction, msg) => { progress.value = fraction; message.value = msg; });
      } catch (err) {
        error.value = err.message;
      } finally {
        busy.value = false;
      }
    }

    return {
      state, selected, style, busy, progress, message, error, result,
      animatable, imagined, SUGGESTIONS, choose, animate, goTo,
      use: (text) => { style.value = text; },
      // The viewport plays the animation baked into the returned GLB, so it
      // needs that file as its rig rather than the avatar's original.
      animatedRig: computed(() => result.value
        ? { ...state.avatar.rig, glb_url: result.value.glb_url }
        : null),
    };
  },
  template: `
    <section class="step">
      <header class="step-head">
        <h2>Make it move</h2>
        <p>Your avatar learned a pose, but not how to get there. A language
           model works out the movement in between.</p>
      </header>

      <div class="split">
        <div class="panel">
          <h3>What should it do?</h3>
          <p v-if="!animatable.length" class="muted">
            Nothing to animate yet — train a move and save what it learned,
            then come back.
          </p>
          <div class="behaviour-grid">
            <button v-for="b in animatable" :key="b.id" class="behaviour"
                    :class="{ active: selected?.id === b.id }"
                    :disabled="busy" @click="choose(b)">
              <strong>{{ b.name }}</strong>
              <small>learned · best {{ b.best_reward.toFixed(2) }}</small>
            </button>
          </div>

          <p v-if="imagined.length" class="muted" style="margin-top:.7rem">
            {{ imagined.length }} imagined move{{ imagined.length > 1 ? 's' : '' }}
            can't be animated — only moves the avatar actually trained have a
            starting pose to move from.
          </p>

          <template v-if="selected">
            <h3>How should it move?</h3>
            <div class="prompt-row">
              <input v-model="style" :disabled="busy"
                     placeholder="e.g. smoothly, like a dancer"
                     @keyup.enter="animate">
              <button class="primary" :disabled="busy" @click="animate">
                {{ busy ? 'Thinking…' : 'Animate' }}
              </button>
            </div>
            <div class="chips">
              <button v-for="s in SUGGESTIONS" :key="s" class="chip"
                      :disabled="busy" @click="use(s)">{{ s }}</button>
            </div>
            <p class="muted">Leave it blank and it'll just find a natural path.</p>
          </template>

          <div v-if="busy" class="progress">
            <div class="progress-bar" :style="{ width: (progress*100) + '%' }"></div>
            <span>{{ message }}</span>
          </div>
          <p v-if="error" class="error">{{ error }}</p>

          <div v-if="result" class="row">
            <button class="primary" :disabled="busy" @click="animate">
              Try again
            </button>
            <button class="ghost" @click="goTo('render')">Make a video →</button>
            <!-- Secondary on purpose: the point of this step is watching it
                 move, not collecting a file. -->
            <a class="link-quiet" :href="result.glb_url"
               :download="(result.animation_name || 'animation') + '.glb'">
              save .glb
            </a>
          </div>
        </div>

        <div class="panel">
          <AvatarCanvas v-if="animatedRig" :key="result.glb_url"
                        :rig="animatedRig" :bakedAnimation="result.animation_name"
                        :label="result.animation_name || 'Animated'" />
          <AvatarCanvas v-else-if="state.avatar"
                        :rig="state.avatar.rig"
                        :pose="selected ? null : state.schema.rest_pose"
                        :clip="selected?.clip"
                        :label="selected ? selected.name : 'Standing by'" />
          <p class="muted" v-if="result">
            Playing on a loop — this is the movement the model invented from
            your avatar's two poses.
          </p>
          <p class="muted" v-else-if="selected">
            The pose your avatar learned. Press Animate to work out how it
            gets there.
          </p>
        </div>
      </div>
    </section>
  `,
};
