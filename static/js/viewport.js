// three.js viewport: renders an avatar and poses it.
//
// One Viewport owns one <canvas>. The training screen makes two of them
// (target and learner) side by side.
//
// It renders BOTH rig formats through the same public API:
//   rig.format === "procedural" -> geometry built here from the bone tree
//   rig.format === "glb"        -> GLTFLoader, bones matched by name
// so switching a real auto-rigger on never touches this file's callers.

import * as THREE from "three";
import { OrbitControls } from "three/addons/controls/OrbitControls.js";
import { GLTFLoader } from "three/addons/loaders/GLTFLoader.js";
import { clone as cloneSkinned } from "three/addons/utils/SkeletonUtils.js";
import { samplePose } from "./pose.js";

const BODY_COLOUR = 0x6c5ce7;
const GHOST_COLOUR = 0xb6c2e1;
const HEAT_COLD = new THREE.Color(0x23c48a); // on target
const HEAT_HOT = new THREE.Color(0xff6b6b);  // way off

const IDENTITY = [0, 0, 0, 1];
//: Rigs arrive at wildly different scales; normalise to roughly this height.
const TARGET_HEIGHT = 1.55;
//: Uniform arrays are fixed-size in GLSL, and reading past the end is
//: undefined behaviour. A full Mixamo rig with fingers and helper bones runs
//: to ~90, so leave real headroom AND clamp the index in the shader.
const MAX_HEAT_BONES = 128;

// Scratch quaternions — #applyPoseGlb runs 16 times a frame, twice over for
// the two training viewports, so it allocates nothing.
const _local = new THREE.Quaternion();
const _desired = new THREE.Quaternion();
const _parentWorld = new THREE.Quaternion();

/**
 * Reduce a bone name to something comparable across exporters.
 *
 * Real rigs reach us via several tools and each mangles names differently. A
 * Mixamo bone authored as "mixamorig:LeftArm" has been seen arriving as:
 *
 *   mixamorig:LeftArm       authored
 *   mixamorigLeftArm        three's PropertyBinding.sanitizeNodeName strips ":"
 *   mixamorig_LeftArm       FBX conversion swaps ":" for "_"
 *   mixamorig_LeftArm_011   ...plus a uniquifying index from the exporter
 *
 * So: drop one trailing _<digits> index, then drop every separator, then
 * lowercase. The index is removed in a SINGLE pass on purpose — stripping all
 * trailing digits would fold "Spine_02" and "Spine1_03" onto the same key and
 * the wrong bone could win.
 */
function normaliseBoneName(name) {
  return (name || "")
    .replace(/_\d+$/, "")
    .replace(/[\s.:/[\]_-]/g, "")
    .toLowerCase();
}

export const VIDEO_WIDTH = 1280;
export const VIDEO_HEIGHT = 720;
export const VIDEO_DURATION_MS = 5000;
export const VIDEO_FPS = 30;

export const SCENE_PRESETS = {
  studio: {
    background: 0xe9eeef,
    floor: 0xd5dddf,
    key: 0xffffff,
    fill: 0x8ac9ff,
    rim: 0xffd39b,
  },
  spotlight: {
    background: 0x151719,
    floor: 0x25282b,
    key: 0xffd19b,
    fill: 0x67aef5,
    rim: 0xff765f,
  },
  "color-pop": {
    background: 0x49bdb4,
    floor: 0xffce68,
    key: 0xffffff,
    fill: 0xff7165,
    rim: 0x6a68d9,
  },
};

export class Viewport {
  constructor(canvas, { ghost = false, render = false } = {}) {
    this.canvas = canvas;
    this.renderMode = render;
    this.bones = {};        // bone name -> Object3D
    this.limbs = {};        // bone name -> [meshes that move with it]
    this.ghostBones = {};
    this.clock = new THREE.Clock();
    this.clip = null;
    this.playing = false;
    this.paused = false;
    this.disposed = false;
    this.captureStartedAt = null;
    this.cameraMotion = "orbit";
    this.cameraTarget = new THREE.Vector3(0, 1, 0);
    this.baseDistance = 4;

    this.renderer = new THREE.WebGLRenderer({
      canvas,
      antialias: true,
      alpha: !render,
      powerPreference: "high-performance",
      preserveDrawingBuffer: render,
    });
    this.renderer.setPixelRatio(render ? 1 : Math.min(devicePixelRatio, 2));
    this.renderer.outputColorSpace = THREE.SRGBColorSpace;
    if (render) {
      this.renderer.setSize(VIDEO_WIDTH, VIDEO_HEIGHT, false);
      this.renderer.toneMapping = THREE.ACESFilmicToneMapping;
      this.renderer.toneMappingExposure = 1.05;
      this.renderer.shadowMap.enabled = true;
      this.renderer.shadowMap.type = THREE.PCFShadowMap;
    }

    this.scene = new THREE.Scene();
    this.camera = new THREE.PerspectiveCamera(
      render ? 35 : 38,
      render ? VIDEO_WIDTH / VIDEO_HEIGHT : 1,
      render ? 0.01 : 0.1,
      100,
    );

    if (render) {
      this.#addStudio();
    } else {
      this.camera.position.set(0, 1.05, 3.3);
      this.controls = new OrbitControls(this.camera, canvas);
      this.controls.target.set(0, 0.85, 0);
      this.controls.enableDamping = true;
      this.controls.enablePan = false;
      this.controls.minDistance = 1.5;
      this.controls.maxDistance = 6;
      this.controls.update();

      this.scene.add(new THREE.HemisphereLight(0xffffff, 0x556677, 2.2));
      const key = new THREE.DirectionalLight(0xffffff, 1.6);
      key.position.set(2, 4, 3);
      this.scene.add(key);
    }

    // The rig has its origin at the hips and the feet reach 0.75 below that,
    // so lift the whole figure to stand it on the ground plane.
    this.root = new THREE.Group();
    this.root.position.y = render ? 0 : 0.75;
    this.scene.add(this.root);
    this.ghostEnabled = ghost;

    if (!render) this.#addGround();
    this.#observeResize();
    this.#animate();
  }

  #addStudio() {
    this.hemisphere = new THREE.HemisphereLight(0xffffff, 0x69767a, 1.4);
    this.key = new THREE.DirectionalLight(0xffffff, 3.2);
    this.fill = new THREE.DirectionalLight(0x8ac9ff, 1.5);
    this.rim = new THREE.DirectionalLight(0xffd39b, 1.8);

    this.key.position.set(4, 6, 5);
    this.key.castShadow = true;
    this.key.shadow.mapSize.set(2048, 2048);
    this.key.shadow.camera.near = 0.1;
    this.key.shadow.camera.far = 20;
    this.key.shadow.camera.left = -4;
    this.key.shadow.camera.right = 4;
    this.key.shadow.camera.top = 4;
    this.key.shadow.camera.bottom = -4;
    this.fill.position.set(-5, 3, 2);
    this.rim.position.set(1, 4, -5);
    this.scene.add(this.hemisphere, this.key, this.fill, this.rim);

    this.floor = new THREE.Mesh(
      new THREE.PlaneGeometry(20, 20),
      new THREE.MeshStandardMaterial({ color: 0xd5dddf, roughness: 0.82 }),
    );
    this.floor.rotation.x = -Math.PI / 2;
    this.floor.position.y = -0.012;
    this.floor.receiveShadow = true;
    this.scene.add(this.floor);
    this.setScenePreset("studio");
  }

  #addGround() {
    const grid = new THREE.GridHelper(6, 12, 0xc3cdea, 0xd9e0f5);
    grid.position.y = -0.001;
    grid.material.transparent = true;
    grid.material.opacity = 0.55;
    this.scene.add(grid);
  }

  #observeResize() {
    if (this.renderMode) return;

    const resize = () => {
      const { clientWidth: w, clientHeight: h } = this.canvas;
      if (!w || !h) return;
      this.renderer.setSize(w, h, false);
      this.camera.aspect = w / h;
      this.camera.updateProjectionMatrix();
    };
    this.resizeObserver = new ResizeObserver(resize);
    this.resizeObserver.observe(this.canvas);
    resize();
  }

  // -- building --------------------------------------------------------

  /** Replace whatever is on screen with this rig. */
  async setRig(rig) {
    this.root.clear();
    this.root.position.set(0, this.renderMode ? 0 : 0.75, 0);
    this.root.scale.setScalar(1);
    this.bones = {};
    this.limbs = {};
    this.ghostBones = {};
    this.boneTree = rig.bone_tree;
    // Stale bind data would silently retarget the next rig against the last
    // one's skeleton, so clear every GLB-specific field up front.
    this.bindWorld = null;
    this.ghostBindWorld = null;
    this.glbGroup = null;
    this.skinned = [];
    this._reverseBones = null;

    if (rig.format === "glb" && rig.glb_url) {
      await this.#loadGlb(rig);
    } else {
      const built = this.#buildProcedural(rig.bone_tree, BODY_COLOUR, 1);
      this.bones = built.bones;
      this.limbs = built.limbs;
      this.root.add(built.group);
    }

    if (this.ghostEnabled) {
      if (this.glbGroup) this.#buildGlbGhost();
      else {
        const built = this.#buildProcedural(rig.bone_tree, GHOST_COLOUR, 0.22);
        this.ghostBones = built.bones;
        this.ghostGroup = built.group;
      }
      this.ghostGroup.visible = false;
      this.root.add(this.ghostGroup);
    }

    if (this.renderMode) this.#frameRig();
  }

  /**
   * Translucent copy of a GLB rig.
   *
   * SkeletonUtils.clone(), not group.clone() — a plain clone of a SkinnedMesh
   * keeps a reference to the ORIGINAL skeleton, so the ghost and the figure
   * would move as one and the comparison would show nothing.
   */
  #buildGlbGhost() {
    const ghost = cloneSkinned(this.glbGroup);
    ghost.traverse((node) => {
      if (node.isMesh) {
        node.material = node.material.clone();
        node.material.onBeforeCompile = undefined;   // no heat tint on the ghost
        node.material.transparent = true;
        node.material.opacity = 0.22;
        node.material.depthWrite = false;
        node.material.color = new THREE.Color(GHOST_COLOUR);
      }
    });

    this.ghostBones = {};
    const byName = new Map();
    ghost.traverse((n) => byName.set(n.name, n));
    for (const [bone, node] of Object.entries(this.bones)) {
      const twin = byName.get(node.name);
      if (twin) this.ghostBones[bone] = twin;
    }

    ghost.updateMatrixWorld(true);
    this.ghostBindWorld = {};
    for (const [bone, node] of Object.entries(this.ghostBones)) {
      this.ghostBindWorld[bone] = node.getWorldQuaternion(new THREE.Quaternion());
    }
    this.ghostGroup = ghost;
  }

  /**
   * Load a rigged GLB (Meshy output is Mixamo-convention).
   *
   * Two problems to solve, and neither is the posing itself — setting
   * `bone.quaternion` drives a skinned mesh exactly as it drives the
   * procedural figure.
   *
   * 1. NAMES. A Mixamo rig calls our `L_shoulder` "mixamorig:LeftArm", so
   *    bones are resolved by contract name first, then via the alias map the
   *    server sends. Unmapped bones (fingers, toes, Spine1/2) stay at bind.
   *
   * 2. BONE FRAMES. Each GLB bone's local axes are whatever the exporter
   *    produced — typically +Y down the limb, not axis-aligned like ours. So
   *    a contract quaternion cannot be written to a GLB bone directly. We
   *    capture each bone's BIND world rotation at load, then retarget through
   *    world space every frame (see #applyPoseGlb).
   *
   * This works without per-bone correction constants only because both
   * skeletons bind in a T-pose. An A-pose rig would need an extra term.
   */
  async #loadGlb(rig) {
    const gltf = await new GLTFLoader().loadAsync(rig.glb_url);
    const group = gltf.scene;

    // A baked idle animation would fight applyPose for control of the bones.
    // We never create a mixer, so clips simply go unplayed — but say so, since
    // silently dropping an animation the rigger exported is confusing.
    if (gltf.animations?.length) {
      console.info(`[viewport] ignoring ${gltf.animations.length} baked ` +
                   `animation(s); poses are driven by the trainer`);
    }

    const alias = rig.bone_aliases?.mixamo || {};
    const byName = new Map();
    group.traverse((node) => {
      // Shadows only exist in the studio used for video capture; the preview
      // viewport has no shadow map, so setting these there costs nothing but
      // buys nothing either.
      if (this.renderMode && node.isMesh) {
        node.castShadow = true;
        node.receiveShadow = true;
      }
      byName.set(node.name, node);
      // GLTFLoader runs every node name through PropertyBinding.sanitizeNodeName,
      // which STRIPS ". : / [ ]" — so a rig authored with "mixamorig:LeftArm"
      // arrives as "mixamorigLeftArm" and an exact lookup finds nothing. Index
      // a normalised form too, and match case-insensitively while we're here,
      // since exporters vary on capitalisation.
      const key = normaliseBoneName(node.name);
      if (key && !byName.has(key)) byName.set(key, node);
    });

    const find = (name) =>
      name ? (byName.get(name) || byName.get(normaliseBoneName(name))) : undefined;

    for (const bone of rig.skeleton) {
      const node = find(bone) || find(alias[bone]);
      if (node) this.bones[bone] = node;
    }

    const missing = rig.skeleton.filter((b) => !this.bones[b]);
    if (missing.length) {
      const present = [];
      group.traverse((n) => { if (n.isBone) present.push(n.name); });
      console.warn("[viewport] GLB has no bone for:", missing,
                   "— those joints will not move. Bones in the file:", present);
    }

    // Bind rotations must be read before anything poses the rig.
    group.updateMatrixWorld(true);
    this.bindWorld = {};
    for (const [bone, node] of Object.entries(this.bones)) {
      this.bindWorld[bone] = node.getWorldQuaternion(new THREE.Quaternion());
    }

    this.#normaliseScale(group);
    this.#prepareSkinnedHeat(group);
    this.root.add(group);
    this.glbGroup = group;
  }

  /**
   * Scale the rig to roughly the height the camera is framed for.
   *
   * Rigs in the Mixamo lineage are often exported in centimetres, so a 1.7m
   * character arrives 170 units tall and fills the screen with a kneecap.
   */
  #normaliseScale(group) {
    const box = new THREE.Box3().setFromObject(group);
    const height = box.max.y - box.min.y;
    if (!Number.isFinite(height) || height <= 0) return;

    const scale = TARGET_HEIGHT / height;
    if (Math.abs(scale - 1) > 0.02) {
      group.scale.setScalar(scale);
      console.info(`[viewport] rig was ${height.toFixed(2)} units tall, ` +
                   `scaled by ${scale.toFixed(3)}`);
    }
    // Stand it on the ground plane. root is lifted for the procedural figure,
    // so undo that here rather than assuming both rigs share an origin.
    group.updateMatrixWorld(true);
    const grounded = new THREE.Box3().setFromObject(group);
    group.position.y -= grounded.min.y + this.root.position.y;
  }

  #frameRig() {
    this.root.updateMatrixWorld(true);
    const bounds = new THREE.Box3().setFromObject(this.root, true);
    const size = bounds.getSize(new THREE.Vector3());
    const center = bounds.getCenter(new THREE.Vector3());
    const largest = Math.max(size.x, size.y, size.z);
    if (!Number.isFinite(largest) || largest <= 0) {
      throw new Error("This avatar has no visible shape to render.");
    }

    const scale = 2.4 / largest;
    this.root.scale.setScalar(scale);
    this.root.position.set(
      -center.x * scale,
      -bounds.min.y * scale,
      -center.z * scale,
    );
    this.root.updateMatrixWorld(true);

    const framedBounds = new THREE.Box3().setFromObject(this.root, true);
    const framedSize = framedBounds.getSize(new THREE.Vector3());
    this.cameraTarget.copy(framedBounds.getCenter(new THREE.Vector3()));

    const verticalFov = THREE.MathUtils.degToRad(this.camera.fov);
    const horizontalFov = 2 * Math.atan(
      Math.tan(verticalFov / 2) * this.camera.aspect,
    );
    const verticalDistance = framedSize.y / (2 * Math.tan(verticalFov / 2));
    const horizontalDistance = framedSize.x / (2 * Math.tan(horizontalFov / 2));
    this.baseDistance = Math.max(
      (Math.max(verticalDistance, horizontalDistance) + framedSize.z / 2) * 1.3,
      1.5,
    );
    this.camera.near = Math.max(0.01, this.baseDistance / 100);
    this.camera.far = Math.max(50, this.baseDistance * 20);
    this.camera.updateProjectionMatrix();
    this.#updateRenderCamera(0, true);
  }

  /**
   * Build a stick figure directly from the bone tree.
   *
   * Each bone becomes an Object3D at its rest offset; each parent gets a capsule
   * drawn towards every child, so the limbs follow the joints automatically.
   * That means the figure is always exactly the skeleton the contract describes
   * — there is no separate model that could drift out of sync with it.
   */
  #buildProcedural(boneTree, colour, opacity) {
    const group = new THREE.Group();
    const bones = {};
    const limbs = {};
    const transparent = opacity < 1;

    // Two passes: create every node, then parent them. One pass would depend on
    // the bone tree arriving parents-first, and JSON gives no such guarantee —
    // a serialiser that sorts keys alphabetically puts L_elbow before
    // L_shoulder and the whole figure fails to build.
    for (const [name, { offset }] of Object.entries(boneTree)) {
      const node = new THREE.Object3D();
      node.position.set(offset[0], offset[1], offset[2]);
      bones[name] = node;
      limbs[name] = [];
    }
    for (const [name, { parent }] of Object.entries(boneTree)) {
      (parent ? bones[parent] : group).add(bones[name]);
    }

    const material = () => new THREE.MeshStandardMaterial({
      color: colour, roughness: 0.45, metalness: 0.05,
      transparent, opacity, depthWrite: !transparent,
    });

    for (const [name, { parent, offset }] of Object.entries(boneTree)) {
      if (!parent) continue;
      const length = Math.hypot(...offset);
      if (length < 1e-4) continue;

      // Capsules are built along +Y, so rotate to point down the offset.
      const limb = new THREE.Mesh(
        new THREE.CapsuleGeometry(name === "head" ? 0.12 : 0.045,
                                  Math.max(length - 0.02, 0.01), 4, 10),
        material());
      limb.position.set(offset[0] / 2, offset[1] / 2, offset[2] / 2);
      limb.quaternion.setFromUnitVectors(
        new THREE.Vector3(0, 1, 0),
        new THREE.Vector3(...offset).normalize());
      bones[parent].add(limb);
      // The limb is drawn between parent and child, but it is the CHILD bone's
      // rotation that a pose changes, so heat-colour it as the child's.
      limbs[name].push(limb);
    }

    // A head that reads as a head, and joint pips so the rig is legible.
    const head = new THREE.Mesh(new THREE.SphereGeometry(0.15, 20, 16), material());
    bones.head.add(head);
    limbs.head.push(head);

    for (const [name, node] of Object.entries(bones)) {
      if (name === "head" || name === "hips") continue;
      const pip = new THREE.Mesh(new THREE.SphereGeometry(0.055, 12, 10), material());
      node.add(pip);
      limbs[name].push(pip);
    }

    return { group, bones, limbs };
  }

  // -- final render controls --------------------------------------------

  setScenePreset(id) {
    if (!this.renderMode || !SCENE_PRESETS[id]) return;
    const preset = SCENE_PRESETS[id];
    this.scene.background = new THREE.Color(preset.background);
    this.floor.material.color.setHex(preset.floor);
    this.key.color.setHex(preset.key);
    this.fill.color.setHex(preset.fill);
    this.rim.color.setHex(preset.rim);
    this.hemisphere.intensity = id === "spotlight" ? 0.65 : 1.4;
    this.key.intensity = id === "spotlight" ? 4 : 3.2;
    this.fill.intensity = id === "color-pop" ? 2 : 1.5;
    this.rim.intensity = id === "spotlight" ? 2.5 : 1.8;
  }

  setCameraMotion(motion) {
    if (!["still", "orbit", "push-in"].includes(motion)) return;
    this.cameraMotion = motion;
    if (this.renderMode) this.#updateRenderCamera(0, true);
  }

  setPaused(paused) {
    this.paused = paused;
    if (!paused) this.clock.getDelta();
  }

  isPaused() {
    return this.paused;
  }

  startCapture(startedAt) {
    this.captureStartedAt = startedAt;
    this.paused = false;
    this.playing = Boolean(this.clip);
    this.elapsed = 0;
    if (this.clip) this.applyPose(samplePose(this.clip, 0));
    if (this.renderMode) this.#updateRenderCamera(0, true);
  }

  stopCapture() {
    this.captureStartedAt = null;
    this.clock.getDelta();
  }

  #updateRenderCamera(elapsedSeconds, recording) {
    if (!this.renderMode) return;

    const baseAzimuth = THREE.MathUtils.degToRad(32);
    let azimuth = baseAzimuth;
    let distance = this.baseDistance;

    if (this.cameraMotion === "orbit") {
      const phase = recording
        ? Math.min(elapsedSeconds / (VIDEO_DURATION_MS / 1000), 1)
        : (elapsedSeconds % 5) / 5;
      azimuth += Math.sin(phase * Math.PI * 2) * THREE.MathUtils.degToRad(16);
    } else if (this.cameraMotion === "push-in") {
      const phase = recording
        ? Math.min(elapsedSeconds / (VIDEO_DURATION_MS / 1000), 1)
        : (Math.sin(elapsedSeconds * 1.1) + 1) / 2;
      const eased = phase * phase * (3 - 2 * phase);
      distance *= 1.16 - 0.16 * eased;
    }

    const elevation = THREE.MathUtils.degToRad(13);
    const horizontalDistance = Math.cos(elevation) * distance;
    this.camera.position.set(
      this.cameraTarget.x + Math.sin(azimuth) * horizontalDistance,
      this.cameraTarget.y + Math.sin(elevation) * distance,
      this.cameraTarget.z + Math.cos(azimuth) * horizontalDistance,
    );
    this.camera.lookAt(this.cameraTarget);
  }

  // -- posing ----------------------------------------------------------

  /** Apply a pose (bone name -> quaternion) to the main figure. */
  applyPose(pose, target = this.bones) {
    if (!pose) return;
    if (this.bindWorld && target === this.bones) {
      this.#applyPoseGlb(pose, this.bones, this.bindWorld);
      return;
    }
    for (const [bone, q] of Object.entries(pose)) {
      target[bone]?.quaternion.set(q[0], q[1], q[2], q[3]);
    }
  }

  /**
   * Retarget a contract pose onto a GLB skeleton, through world space.
   *
   * A contract pose is a local rotation per bone in OUR frame, where every
   * bone binds at identity. The same rotation written straight to a GLB bone
   * would be interpreted in that bone's own frame — usually +Y down the limb —
   * and produce a knot. So:
   *
   *   1. forward-kinematics our pose to a world rotation per bone. Because our
   *      bind is identity, that world rotation IS the delta from bind.
   *   2. apply that delta to the GLB bone's captured bind world rotation.
   *   3. convert back to the GLB bone's local frame, against its parent.
   *
   * Parents are handled before children (BONE_TREE is ordered that way) so a
   * parent's world matrix is current by the time a child reads it.
   */
  #applyPoseGlb(pose, bones, bindWorld) {
    const worldDelta = {};

    for (const [name, { parent }] of Object.entries(this.boneTree)) {
      const q = pose[name] || IDENTITY;
      _local.set(q[0], q[1], q[2], q[3]);
      worldDelta[name] = parent
        ? worldDelta[parent].clone().multiply(_local)
        : _local.clone();

      const node = bones[name];
      if (!node) continue;

      // World-space rotations pre-multiply: delta THEN the bind orientation.
      _desired.copy(worldDelta[name]).multiply(bindWorld[name]);

      if (node.parent) {
        node.parent.updateWorldMatrix(true, false);
        node.parent.getWorldQuaternion(_parentWorld);
        node.quaternion.copy(_parentWorld.invert()).multiply(_desired);
      } else {
        node.quaternion.copy(_desired);
      }
    }
  }

  /**
   * Give a skinned mesh a per-bone tint uniform.
   *
   * The procedural figure colours joints by swapping each limb's material,
   * which a skinned mesh cannot do — it is one mesh with one material. Instead
   * inject a per-bone colour array and have the fragment shader blend it using
   * the vertex's own skin weights, so the tint follows the deformation.
   */
  #prepareSkinnedHeat(group) {
    this.skinned = [];

    group.traverse((node) => {
      if (!node.isSkinnedMesh) return;
      // Each mesh gets its OWN tint array. A character export is often several
      // skinned meshes (body, hair, clothes), and they do not have to share a
      // skeleton — one shared array would have them overwriting each other.
      const uniform = { value: Array.from({ length: MAX_HEAT_BONES },
                                          () => new THREE.Color(1, 1, 1)) };
      node.userData.heatUniform = uniform;
      this.skinned.push(node);
      node.material = node.material.clone();
      node.material.onBeforeCompile = (shader) => {
        shader.uniforms.boneTint = uniform;
        shader.vertexShader = shader.vertexShader
          .replace("#include <common>",
            `#include <common>
             uniform vec3 boneTint[${MAX_HEAT_BONES}];
             varying vec3 vBoneTint;`)
          .replace("#include <skinning_vertex>",
            `#include <skinning_vertex>
             int bi0 = int(clamp(skinIndex.x, 0.0, float(${MAX_HEAT_BONES - 1})));
             int bi1 = int(clamp(skinIndex.y, 0.0, float(${MAX_HEAT_BONES - 1})));
             int bi2 = int(clamp(skinIndex.z, 0.0, float(${MAX_HEAT_BONES - 1})));
             int bi3 = int(clamp(skinIndex.w, 0.0, float(${MAX_HEAT_BONES - 1})));
             vBoneTint =
               boneTint[bi0] * skinWeight.x +
               boneTint[bi1] * skinWeight.y +
               boneTint[bi2] * skinWeight.z +
               boneTint[bi3] * skinWeight.w;`);
        shader.fragmentShader = shader.fragmentShader
          .replace("#include <common>",
            "#include <common>\nvarying vec3 vBoneTint;")
          .replace("#include <color_fragment>",
            "#include <color_fragment>\ndiffuseColor.rgb *= vBoneTint;");
      };
      node.material.needsUpdate = true;
    });
  }

  /** Show a translucent copy of a pose behind the figure, for comparison. */
  setGhostPose(pose) {
    if (!this.ghostGroup) return;
    this.ghostGroup.visible = Boolean(pose);
    if (!pose) return;
    if (this.ghostBindWorld) {
      this.#applyPoseGlb(pose, this.ghostBones, this.ghostBindWorld);
    } else {
      this.applyPose(pose, this.ghostBones);
    }
  }

  showGhost(visible) {
    if (this.ghostGroup) this.ghostGroup.visible = visible;
  }

  /** Play a clip on a loop. Pass null to stop. */
  playClip(clip) {
    this.clip = clip;
    this.playing = Boolean(clip);
    this.clock.start();
    this.elapsed = 0;
    if (clip) this.applyPose(samplePose(clip, 0));
  }

  /** Jump to a specific time in the current clip (for the scrubber). */
  seek(seconds) {
    if (!this.clip) return;
    this.playing = false;
    this.elapsed = seconds;
    this.applyPose(samplePose(this.clip, seconds));
  }

  /**
   * Tint each limb by how wrong that joint is: green on target, red way off.
   * This is the bit that makes "which part hasn't it learned yet" visible.
   */
  setJointHeat(perJointError) {
    if (!perJointError) return;
    if (this.skinned?.length) { this.#setSkinnedHeat(perJointError); return; }
    for (const bone of Object.keys(this.limbs)) {
      // Only the articulated bones are scored, but leaving hands and feet at
      // the default blue reads as a third state the legend never explains. Let
      // them inherit the nearest scored ancestor so the whole figure is either
      // green or red.
      const error = this.#inheritedError(bone, perJointError);
      if (error === undefined) continue;
      const colour = HEAT_COLD.clone().lerp(HEAT_HOT,
        Math.min(1, Math.max(0, error * 2.5)));
      for (const mesh of this.limbs[bone]) mesh.material.color.copy(colour);
    }
  }

  #inheritedError(bone, perJointError) {
    let name = bone;
    while (name) {
      if (perJointError[name] !== undefined) return perJointError[name];
      name = this.boneTree?.[name]?.parent;
    }
    return undefined;
  }

  /**
   * Fill the per-bone tint array the shader reads.
   *
   * Indices are the skeleton's own bone order, not our contract order, so each
   * GLB bone has to be looked up by identity. Bones we do not score inherit
   * their nearest scored ancestor, as in the procedural path.
   */
  #setSkinnedHeat(perJointError) {
    for (const mesh of this.skinned) {
      const tints = mesh.userData.heatUniform.value;
      const list = mesh.skeleton.bones;
      for (let i = 0; i < list.length && i < MAX_HEAT_BONES; i++) {
        const contractName = this.#contractNameOf(list[i]);
        const error = contractName
          ? this.#inheritedError(contractName, perJointError) : undefined;
        if (error === undefined) { tints[i].setRGB(1, 1, 1); continue; }
        tints[i].copy(HEAT_COLD).lerp(HEAT_HOT,
          Math.min(1, Math.max(0, error * 2.5)));
      }
    }
  }

  /** Which contract bone, if any, a GLB node corresponds to. */
  #contractNameOf(node) {
    if (!this._reverseBones) {
      this._reverseBones = new Map(
        Object.entries(this.bones).map(([name, n]) => [n, name]));
    }
    // Walk up: a Mixamo Spine2 is unmapped, but sits under our `spine`.
    for (let n = node; n; n = n.parent) {
      const hit = this._reverseBones.get(n);
      if (hit) return hit;
    }
    return null;
  }

  clearJointHeat() {
    if (this.skinned?.length) {
      for (const mesh of this.skinned) {
        for (const tint of mesh.userData.heatUniform.value) tint.setRGB(1, 1, 1);
      }
      return;
    }
    for (const meshes of Object.values(this.limbs)) {
      for (const mesh of meshes) mesh.material.color.setHex(BODY_COLOUR);
    }
  }

  // -- loop ------------------------------------------------------------

  #animate = () => {
    if (this.disposed) return;
    this.animationFrame = requestAnimationFrame(this.#animate);

    const now = performance.now();
    const delta = this.clock.getDelta();
    if (this.captureStartedAt !== null) {
      this.elapsed = Math.max(0, (now - this.captureStartedAt) / 1000);
      if (this.clip) this.applyPose(samplePose(this.clip, this.elapsed));
      this.#updateRenderCamera(this.elapsed, true);
    } else if (!this.paused && this.playing && this.clip) {
      this.elapsed += delta;
      this.applyPose(samplePose(this.clip, this.elapsed));
      if (!this.clip.loop && this.elapsed >= this.clip.duration) this.playing = false;
    }

    if (this.renderMode && this.captureStartedAt === null && !this.paused) {
      this.#updateRenderCamera(now / 1000, false);
    }
    this.controls?.update();
    this.renderer.render(this.scene, this.camera);
  };

  dispose() {
    this.disposed = true;
    cancelAnimationFrame(this.animationFrame);
    this.resizeObserver?.disconnect();
    this.controls?.dispose();
    this.floor?.geometry.dispose();
    this.floor?.material.dispose();
    this.renderer.dispose();
  }
}
