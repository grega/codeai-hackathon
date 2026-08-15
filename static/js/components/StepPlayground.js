// Playground — preview the saved behaviours before choosing one to render.
//
// "Present the avatar with novel environments to navigate" is the next phase of
// this project and is not built. What exists here is the landing place for it:
// the behaviour library, and a trigger for each behaviour. Whoever builds the
// environments plugs them in here — a behaviour is already the right unit
// (a named clip the avatar can perform on cue).

import { onMounted } from "vue";
import {
  goTo,
  refreshBehaviours,
  setActiveBehaviour,
  state,
} from "../store.js";
import { AvatarCanvas } from "./AvatarCanvas.js";

export const StepPlayground = {
  components: { AvatarCanvas },
  setup() {
    onMounted(refreshBehaviours);

    return {
      state,
      play: setActiveBehaviour,
      render: () => goTo("render"),
    };
  },
  template: `
    <section class="step">
      <header class="step-head">
        <h2>Play</h2>
        <p>Everything your avatar knows. Tap a behaviour to make it perform.</p>
      </header>

      <div class="split">
        <div class="panel">
          <h3>Its behaviours</h3>
          <p v-if="!state.behaviours.length" class="muted">
            Nothing saved yet. Make a move on the Teach step, or train one and
            save what it learned.
          </p>
          <div class="behaviour-grid">
            <button v-for="b in state.behaviours" :key="b.id"
                    class="behaviour"
                    :class="{ active: state.activeBehaviour?.id === b.id }"
                    @click="play(b)">
              <strong>{{ b.name }}</strong>
              <small v-if="b.trained">learned · best {{ b.best_reward.toFixed(2) }}</small>
              <small v-else>imagined</small>
            </button>
          </div>
          <div class="row" v-if="state.activeBehaviour">
            <button class="primary" @click="render">Make a video</button>
          </div>

          <div class="stub-note">
            <h4>Coming next: environments</h4>
            <p>This is where the avatar gets dropped into a world it hasn't seen
               before — obstacles to get past, things to reach — and has to pick
               which of its behaviours to use. Not built yet; the behaviours
               above are the pieces it will choose from.</p>
          </div>
        </div>

        <div class="panel">
          <AvatarCanvas :rig="state.avatar.rig"
                        :clip="state.activeBehaviour?.clip"
                        :label="state.activeBehaviour?.name || 'Standing by'" />
        </div>
      </div>
    </section>
  `,
};
