// Wraps a Viewport in a Vue component and keeps it in sync with props.
// Every screen that shows an avatar uses this rather than touching three.js.

import { onBeforeUnmount, onMounted, ref, watch } from "vue";
import { Viewport } from "../viewport.js";

export const AvatarCanvas = {
  props: {
    rig: { type: Object, required: true },
    pose: { type: Object, default: null },     // static pose to hold
    clip: { type: Object, default: null },     // clip to play on a loop
    ghostPose: { type: Object, default: null },// translucent comparison figure
    jointHeat: { type: Object, default: null },// bone -> 0..1 error, tints limbs
    label: { type: String, default: "" },
    //: Play an animation baked into the GLB instead of a contract pose/clip.
    //: Name or index; null hands the bones back to applyPose.
    bakedAnimation: { type: [String, Number], default: null },
  },
  setup(props) {
    const canvas = ref(null);
    let viewport = null;

    onMounted(async () => {
      viewport = new Viewport(canvas.value, { ghost: true });
      await viewport.setRig(props.rig);
      if (props.clip) viewport.playClip(props.clip);
      else if (props.pose) viewport.applyPose(props.pose);
      if (props.ghostPose) viewport.setGhostPose(props.ghostPose);
      if (props.jointHeat) viewport.setJointHeat(props.jointHeat);
      if (props.bakedAnimation !== null) {
        viewport.playGlbAnimation(props.bakedAnimation);
      }
    });

    onBeforeUnmount(() => viewport?.dispose());

    watch(() => props.rig, async (rig) => {
      if (viewport && rig) await viewport.setRig(rig);
    });
    watch(() => props.clip, (clip) => viewport?.playClip(clip));
    watch(() => props.pose, (pose) => {
      // A clip is animating; a static pose would fight it for control.
      if (pose && !props.clip) viewport?.applyPose(pose);
    });
    watch(() => props.ghostPose, (pose) => viewport?.setGhostPose(pose));
    watch(() => props.bakedAnimation, (which) => {
      if (which === null) viewport?.stopGlbAnimation();
      else viewport?.playGlbAnimation(which);
    });
    watch(() => props.jointHeat, (heat) => {
      if (heat) viewport.setJointHeat(heat);
      else viewport?.clearJointHeat();
    });

    return { canvas };
  },
  template: `
    <div class="stage">
      <span v-if="label" class="stage-label">{{ label }}</span>
      <canvas ref="canvas"></canvas>
    </div>
  `,
};
