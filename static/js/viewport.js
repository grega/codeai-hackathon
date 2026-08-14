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
import { samplePose } from "./pose.js";

const BODY_COLOUR = 0x6c5ce7;
const GHOST_COLOUR = 0xb6c2e1;
const HEAT_COLD = new THREE.Color(0x23c48a); // on target
const HEAT_HOT = new THREE.Color(0xff6b6b);  // way off

export class Viewport {
  constructor(canvas, { ghost = false } = {}) {
    this.canvas = canvas;
    this.bones = {};        // bone name -> Object3D
    this.limbs = {};        // bone name -> [meshes that move with it]
    this.ghostBones = {};
    this.clock = new THREE.Clock();
    this.clip = null;
    this.playing = false;
    this.disposed = false;

    this.renderer = new THREE.WebGLRenderer({
      canvas, antialias: true, alpha: true,
    });
    this.renderer.setPixelRatio(Math.min(devicePixelRatio, 2));

    this.scene = new THREE.Scene();
    this.camera = new THREE.PerspectiveCamera(38, 1, 0.1, 100);
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

    // The rig has its origin at the hips and the feet reach 0.75 below that,
    // so lift the whole figure to stand it on the ground plane.
    this.root = new THREE.Group();
    this.root.position.y = 0.75;
    this.scene.add(this.root);
    this.ghostEnabled = ghost;

    this.#addGround();
    this.#observeResize();
    this.#animate();
  }

  #addGround() {
    const grid = new THREE.GridHelper(6, 12, 0xc3cdea, 0xd9e0f5);
    grid.position.y = -0.001;
    grid.material.transparent = true;
    grid.material.opacity = 0.55;
    this.scene.add(grid);
  }

  #observeResize() {
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
    this.bones = {};
    this.limbs = {};
    this.ghostBones = {};
    this.boneTree = rig.bone_tree;

    if (rig.format === "glb" && rig.glb_url) {
      await this.#loadGlb(rig);
    } else {
      const built = this.#buildProcedural(rig.bone_tree, BODY_COLOUR, 1);
      this.bones = built.bones;
      this.limbs = built.limbs;
      this.root.add(built.group);
    }

    if (this.ghostEnabled) {
      const built = this.#buildProcedural(rig.bone_tree, GHOST_COLOUR, 0.22);
      this.ghostBones = built.bones;
      this.ghostGroup = built.group;
      this.ghostGroup.visible = false;
      this.root.add(this.ghostGroup);
    }
  }

  async #loadGlb(rig) {
    // Untested against a real asset — no rigger produces GLBs yet. The bone
    // lookup below is the whole integration: a GLB whose bone names match
    // schemas.BONES will pose correctly through the same applyPose() calls.
    const gltf = await new GLTFLoader().loadAsync(rig.glb_url);
    const group = gltf.scene;
    group.traverse((node) => {
      if (node.isBone || node.isObject3D) {
        if (rig.skeleton.includes(node.name)) this.bones[node.name] = node;
      }
    });
    this.root.add(group);

    const missing = rig.skeleton.filter((b) => !this.bones[b]);
    if (missing.length) {
      console.warn("[viewport] GLB is missing contract bones:", missing);
    }
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

  // -- posing ----------------------------------------------------------

  /** Apply a pose (bone name -> quaternion) to the main figure. */
  applyPose(pose, target = this.bones) {
    if (!pose) return;
    for (const [bone, q] of Object.entries(pose)) {
      target[bone]?.quaternion.set(q[0], q[1], q[2], q[3]);
    }
  }

  /** Show a translucent copy of a pose behind the figure, for comparison. */
  setGhostPose(pose) {
    if (!this.ghostGroup) return;
    this.ghostGroup.visible = Boolean(pose);
    if (pose) this.applyPose(pose, this.ghostBones);
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

  clearJointHeat() {
    for (const meshes of Object.values(this.limbs)) {
      for (const mesh of meshes) mesh.material.color.setHex(BODY_COLOUR);
    }
  }

  // -- loop ------------------------------------------------------------

  #animate = () => {
    if (this.disposed) return;
    requestAnimationFrame(this.#animate);

    const delta = this.clock.getDelta();
    if (this.playing && this.clip) {
      this.elapsed += delta;
      this.applyPose(samplePose(this.clip, this.elapsed));
      if (!this.clip.loop && this.elapsed >= this.clip.duration) this.playing = false;
    }

    this.controls.update();
    this.renderer.render(this.scene, this.camera);
  };

  dispose() {
    this.disposed = true;
    this.resizeObserver?.disconnect();
    this.controls.dispose();
    this.renderer.dispose();
  }
}
