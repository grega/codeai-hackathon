// Shared reactive state. Small enough that a Vue `reactive` object beats
// pulling in a state library.

import { reactive } from "vue";
import { api } from "./api.js";

export const STEPS = [
  { id: "sketch", label: "Draw", hint: "Draw a character" },
  { id: "pose", label: "Teach", hint: "Describe a move" },
  { id: "train", label: "Train", hint: "Reward what you want" },
  { id: "play", label: "Play", hint: "Try it out" },
  { id: "render", label: "Render", hint: "Make a video" },
];

export const state = reactive({
  ready: false,
  schema: null,       // bone names, rest pose, reward labels — from GET /api/schema
  providers: {},      // which of the three are mock vs real
  step: "sketch",
  avatar: null,
  clips: [],          // every move generated this session
  activeClip: null,   // the one being previewed / trained
  behaviours: [],
  activeBehaviour: null,
  error: null,
  // status: idle | running | done | error. Kept here rather than in
  // StepSketch so the download button survives navigating on to "Teach" —
  // the job is fired the moment the avatar is created, before the user has
  // had a chance to look at it.
  tpose: { status: "idle", progress: 0, message: "", dataUrl: null, error: null },
});

export async function boot() {
  try {
    const schema = await api.schema();
    state.schema = schema;
    state.providers = schema.providers;
    state.ready = true;
  } catch (err) {
    state.error = err.message;
  }
}

export function goTo(step) {
  if (canEnter(step)) state.step = step;
}

/** Steps unlock in order — you can't train a move you haven't made yet. */
export function canEnter(step) {
  if (step === "sketch") return true;
  if (step === "pose") return Boolean(state.avatar);
  if (step === "train") return Boolean(state.avatar && state.clips.length);
  if (step === "play") return Boolean(state.avatar);
  if (step === "render") {
    return Boolean(state.avatar && state.behaviours.length);
  }
  return false;
}

// Bumped by every generatePosedAvatar() call (and on a new avatar) so an
// older in-flight job — from the raw sketch, superseded by one from a
// render, or from a since-replaced avatar — can tell it's been superseded
// and drop its result instead of clobbering a newer one that finished first.
let tposeGeneration = 0;

export function setAvatar(avatar) {
  state.avatar = avatar;
  state.clips = [];
  state.activeClip = null;
  state.behaviours = [];
  state.activeBehaviour = null;
  tposeGeneration++;
  state.tpose = { status: "idle", progress: 0, message: "", dataUrl: null, error: null };
}

/** Kick off the T-pose transform for the current avatar. Fire-and-forget —
 * callers don't await this, so it runs in the background while the wizard
 * moves on; progress is tracked on state.tpose instead of a return value.
 * Always starts a fresh job rather than skipping while one is running: a
 * render finishing while the sketch-based job is still in flight needs its
 * own job, not to be dropped waiting for the stale one. */
export async function generatePosedAvatar() {
  if (!state.avatar) return;
  const generation = ++tposeGeneration;
  state.tpose = { status: "running", progress: 0, message: "Posing your character...", dataUrl: null, error: null };
  try {
    const result = await api.tposeAvatar(state.avatar.id, (fraction, message) => {
      if (generation !== tposeGeneration) return;
      state.tpose.progress = fraction;
      state.tpose.message = message;
    });
    if (generation !== tposeGeneration) return;
    state.tpose.status = "done";
    state.tpose.dataUrl = `data:image/${result.output_format};base64,${result.image_base64}`;
  } catch (err) {
    if (generation !== tposeGeneration) return;
    state.tpose.status = "error";
    state.tpose.error = err.message;
  }
}

export function addClip(clip) {
  state.clips.unshift(clip);
  state.activeClip = clip;
}

export async function refreshBehaviours() {
  if (!state.avatar) return;
  const { behaviours } = await api.listBehaviours(state.avatar.id);
  state.behaviours = behaviours;
  const selectedId = state.activeBehaviour?.id;
  state.activeBehaviour = behaviours.find((item) => item.id === selectedId)
    || behaviours[0]
    || null;
}

export function setActiveBehaviour(behaviour) {
  state.activeBehaviour = behaviour;
}
