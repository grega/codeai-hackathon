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

export function setAvatar(avatar) {
  state.avatar = avatar;
  state.clips = [];
  state.activeClip = null;
  state.behaviours = [];
  state.activeBehaviour = null;
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
