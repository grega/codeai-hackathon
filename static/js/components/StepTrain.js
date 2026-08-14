// Phases 3 & 4 — the training screen. This is the lesson.
//
// Left: the target pose the model imagined. Right: the learner, updating live.
// Below: what earns points, and the learning curve that results.

import { computed, onBeforeUnmount, ref } from "vue";
import { api } from "../api.js";
import { characteristicPose } from "../pose.js";
import { refreshBehaviours, state } from "../store.js";
import { AvatarCanvas } from "./AvatarCanvas.js";
import { RewardChart } from "./RewardChart.js";

export const StepTrain = {
  components: { AvatarCanvas, RewardChart },
  setup() {
    const weights = ref({ ...state.schema.reward_weights });
    const episodesTotal = ref(300);
    const speed = ref(1);
    const history = ref([]);
    const latest = ref(null);
    const run = ref(null);
    const status = ref("idle"); // idle | running | paused | finished | error
    const error = ref(null);
    const saved = ref(false);
    let closeStream = null;

    const targetPose = computed(() =>
      state.activeClip
        ? characteristicPose(state.activeClip, state.schema.rest_pose)
        : state.schema.rest_pose);

    const learnerPose = computed(() => latest.value?.pose || state.schema.rest_pose);
    const jointHeat = computed(() => latest.value?.per_joint_error || null);
    const matchPercent = computed(() =>
      Math.round((latest.value?.match ?? 0) * 100));

    // With every slider at zero there is nothing to earn, so the learner gets
    // no signal at all. Worth saying out loud — otherwise it just looks broken.
    const noReward = computed(() =>
      Object.values(weights.value).every((w) => w <= 0));

    async function start() {
      stopStream();
      history.value = [];
      latest.value = null;
      error.value = null;
      saved.value = false;
      status.value = "running";

      try {
        run.value = await api.startRun(
          state.avatar.id, state.activeClip.id,
          { episodes: episodesTotal.value, reward_weights: weights.value },
          speed.value);

        closeStream = api.streamRun(run.value.id, {
          onEpisode(episode) {
            latest.value = episode;
            // Thin the chart data on long runs — one point per pixel is plenty
            // and keeps the redraw cheap.
            const stride = Math.ceil(episodesTotal.value / 400);
            if (episode.episode % stride === 0 || episode.done) {
              history.value.push(episode);
            }
          },
          onEnd() {
            status.value = "finished";
          },
          onError(err) {
            error.value = err.message;
            status.value = "error";
          },
        });
      } catch (err) {
        error.value = err.message;
        status.value = "error";
      }
    }

    async function stop() {
      if (!run.value) return;
      await api.runControl(run.value.id, "stop");
      status.value = "finished";
    }

    async function togglePause() {
      if (!run.value) return;
      const paused = status.value === "paused";
      await api.runControl(run.value.id, paused ? "resume" : "pause");
      status.value = paused ? "running" : "paused";
    }

    async function changeSpeed(value) {
      speed.value = Number(value);
      if (run.value && status.value !== "idle") {
        await api.runControl(run.value.id, "speed", { speed: speed.value });
      }
    }

    function reset() {
      stopStream();
      history.value = [];
      latest.value = null;
      run.value = null;
      status.value = "idle";
    }

    function stopStream() {
      closeStream?.();
      closeStream = null;
    }

    async function saveLearned() {
      // Save what the avatar actually LEARNED — a single pose it reached — as
      // distinct from the clip the model imagined. That difference is the point
      // of the whole exercise.
      await api.saveBehaviour({
        avatar_id: state.avatar.id,
        name: `${state.activeClip.name} (learned)`,
        trained: true,
        best_reward: latest.value.best_reward,
        clip: {
          name: `${state.activeClip.name} (learned)`,
          fps: 24,
          keyframes: [{ t: 0, pose: latest.value.pose }],
        },
      });
      await refreshBehaviours();
      saved.value = true;
    }

    onBeforeUnmount(stopStream);

    return {
      state, weights, episodesTotal, speed, history, latest, status, error, run,
      targetPose, learnerPose, jointHeat, matchPercent, noReward, saved,
      start, stop, togglePause, reset, changeSpeed, saveLearned,
      labels: state.schema.reward_labels,
      running: computed(() => status.value === "running" || status.value === "paused"),
    };
  },
  template: `
    <section class="step">
      <header class="step-head">
        <h2>Train your avatar</h2>
        <p>Your avatar starts out knowing nothing. It tries small changes, and
           keeps the ones that earn points. You decide what earns points.</p>
      </header>

      <div v-if="!state.activeClip" class="placeholder">
        <p>Pick a move on the previous step first.</p>
      </div>

      <template v-else>
        <div class="arena">
          <AvatarCanvas :rig="state.avatar.rig" :pose="targetPose"
                        label="Target" />
          <div class="arena-meter">
            <div class="match-ring" :style="{ '--pct': matchPercent }">
              <span>{{ matchPercent }}%</span>
              <small>match</small>
            </div>
            <dl>
              <div><dt>Try</dt><dd>{{ latest?.episode ?? 0 }} / {{ episodesTotal }}</dd></div>
              <div><dt>Score</dt><dd>{{ (latest?.reward ?? 0).toFixed(2) }}</dd></div>
              <div><dt>Best</dt><dd>{{ (latest?.best_reward ?? 0).toFixed(2) }}</dd></div>
              <div><dt>Trying new things</dt>
                   <dd>{{ Math.round((latest?.exploration ?? 0) * 100) }}%</dd></div>
            </dl>
            <p v-if="latest?.note" class="note">{{ latest.note }}</p>
          </div>
          <AvatarCanvas :rig="state.avatar.rig" :pose="learnerPose"
                        :jointHeat="jointHeat" label="Learner" />
        </div>
        <p class="muted legend">
          <i class="dot green"></i> joint is on target
          <i class="dot red"></i> joint is still wrong
        </p>

        <div class="split">
          <div class="panel">
            <h3>What earns points?</h3>
            <p class="muted">Change these and train again — the avatar will
               learn something different.</p>
            <div v-for="(label, key) in labels" :key="key" class="slider-row">
              <label>{{ label }}</label>
              <input type="range" min="0" max="1" step="0.05"
                     v-model.number="weights[key]" :disabled="running">
              <output>{{ weights[key].toFixed(2) }}</output>
            </div>
            <p v-if="noReward" class="warn">
              Nothing earns points right now, so there's nothing to learn.
              Turn at least one slider up.
            </p>

            <div class="slider-row">
              <label>How many tries</label>
              <input type="range" min="50" max="1000" step="50"
                     v-model.number="episodesTotal" :disabled="running">
              <output>{{ episodesTotal }}</output>
            </div>
            <div class="slider-row">
              <label>Speed</label>
              <input type="range" min="0.25" max="5" step="0.25"
                     :value="speed" @input="changeSpeed($event.target.value)">
              <output>{{ speed }}×</output>
            </div>

            <div class="row">
              <button class="primary" v-if="!running" @click="start">
                {{ status === 'finished' ? 'Train again' : 'Start training' }}
              </button>
              <button class="ghost" v-if="running" @click="togglePause">
                {{ status === 'paused' ? 'Resume' : 'Pause' }}
              </button>
              <button class="ghost" v-if="running" @click="stop">Stop</button>
              <button class="ghost" v-if="status === 'finished'" @click="reset">
                Clear
              </button>
              <button class="primary" v-if="status === 'finished' && latest"
                      @click="saveLearned" :disabled="saved">
                {{ saved ? 'Saved!' : 'Save what it learned' }}
              </button>
            </div>
            <p v-if="error" class="error">{{ error }}</p>
          </div>

          <div class="panel">
            <h3>The learning curve</h3>
            <RewardChart :episodes="history" :total="episodesTotal" />
            <p class="muted">
              The blue line jumps around because the avatar is guessing. The
              green line only ever goes up — that's its best score so far. If
              green flattens out, it has stopped improving.
            </p>
          </div>
        </div>
      </template>
    </section>
  `,
};
