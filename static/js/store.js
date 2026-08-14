// Shared reactive state. Small enough that a Vue `reactive` object beats
// pulling in a state library.

import { reactive } from "vue";
import { api } from "./api.js";

export const STEPS = [
  { id: "sketch", label: "Draw", hint: "Draw a character" },
  { id: "pose", label: "Teach", hint: "Describe a move" },
  { id: "train", label: "Train", hint: "Reward what you want" },
  { id: "play", label: "Play", hint: "Try it out" },
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
  error: null,
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
  return false;
}

export function setAvatar(avatar) {
  state.avatar = avatar;
  state.clips = [];
  state.activeClip = null;
  state.behaviours = [];
}

export function addClip(clip) {
  state.clips.unshift(clip);
  state.activeClip = clip;
}

export async function refreshBehaviours() {
  if (!state.avatar) return;
  const { behaviours } = await api.listBehaviours(state.avatar.id);
  state.behaviours = behaviours;
}
