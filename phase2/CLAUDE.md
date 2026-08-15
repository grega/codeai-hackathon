# phase2 — LLM-generated skeletal animation for glTF/GLB

Exploration of a text-prompt → skeletal animation pipeline: take an auto-rigged,
un-animated `.glb` (a T-posed humanoid with no baked animation), ask an LLM to
write the *code* for a procedural animation against a small kinematics API, run
that code, and bake the result into a new `.glb` with a glTF animation clip.

The core bet (validated through this build): general-purpose LLMs are unreliable
at emitting dense per-frame keyframe numbers directly, but are good at writing a
few lines of trig against a well-designed helper API. Almost all of the effort
here went into making that API hard to misuse, not into the animations themselves.

## Running it

```
uv run main.py "wave arms in air"                       # local MLX server (see below)
uv run main.py "jump in the air" --openrouter            # anthropic/claude-sonnet-5 via OpenRouter
uv run main.py "spin around in a circle" --no-llm        # hardcoded fallback only, no LLM call
```

- `--openrouter` needs `OPENROUTER_API_KEY` in `../.env` (repo root, gitignored).
- Local models are served via `python -m mlx_vlm.server --model <repo-or-path> --port <n>`
  (an OpenAI-compatible endpoint), then pass `--model` / `--base-url` to point at it.
  `mlx_vlm`, not `mlx_lm`, because the models tried so far (Gemma 3, Qwen3.6-A3B)
  are natively multimodal — text-only use works fine, the vision half is just unused.
- Generated `.glb` files are always written to `output/` next to this script
  (gitignored — reproducible from `rigged_human.glb` + a prompt) regardless of cwd.
- A window opens immediately with a "Generating..." spinner (served from
  `status.html`, not a `data:` URL — see gotchas below) and swaps to the animated
  model once generation finishes.

## File map

| File | Role |
|---|---|
| `rigged_human.glb` | Source asset: Mixamo-rigged T-pose human, CC-BY-4.0, no animation |
| `gltf_utils.py` | Read joint hierarchy from a GLB; write new animation channels back in; `local_rotation_for_world_delta` — the core world-axis rotation helper |
| `quat.py` | Quaternion math: multiply, axis-angle (radians and degrees), `combine()` |
| `animator.py` | Dispatches a prompt to the LLM path, falling back to two hand-written generators (`generate_wave`, `generate_spin`) if it fails |
| `llm_animator.py` | The LLM code-gen path: prompt construction, sandboxed `exec()`, validation, retry-with-error-feedback |
| `llm_client.py` | Thin OpenAI-compatible client — same code path for a local MLX server or OpenRouter |
| `main.py` | CLI entry point, pywebview window lifecycle |
| `server.py` | Local static file server (model-viewer can't `fetch()` a binary GLB over `file://`) |
| `viewer.html` / `status.html` | `<model-viewer>` preview page / loading-and-error page |

## Non-obvious things worth knowing before touching this

**The rig's own local bone axes are unusable directly.** Mixamo-style joints each
have their own arbitrary rest-rotation quaternion baked in; rotating "around local
X" does not mean anything predictable per-bone. Every rotation in this codebase is
authored as a delta in **world space** (+Y up, +Z forward, ±X left/right for this
character, confirmed empirically via the toe-bone direction) and converted to a
joint's local space via `gltf_utils.local_rotation_for_world_delta`. Do not
reintroduce direct local-axis rotation — it's what sent an early wave animation's
arm behind the character's back.

**This rig rests in a T-pose**, not arms-down. A rotation *raises* an arm from
already-horizontal; ~80-100° on the shoulder/upper-arm alone is enough to bring a
hand overhead. Stacking another large rotation on the child joint (forearm) on top
of that compounds and overshoots — this produced a real bug (arms crossing behind
the head) that's now called out explicitly in `llm_animator.SYSTEM_PROMPT`.

**Composing multiple rotations on one joint** needs `quat.combine()` (exposed to
generated code as `combine`). Without it, an LLM asked for "lift AND wave" will
either silently drop one motion (observed: Gemma computed a sway delta and then
never used it) or spread it across unrelated joints. Also watch for a joint
animated together with its *parent*: their rotations compose, and giving the child
the inverse of the parent's delta "to correct it" freezes everything below that
joint in world space — a real bug seen once (shoulder visibly rotated, everything
below it stayed frozen).

**The LLM sandbox contract is degrees, not radians**, deliberately: models kept
writing `angle = 20 * sin(...)` and passing that straight to a radians-based
`axis_angle`, silently producing ~1146° swings instead of 20°. The sandbox only
exposes a degrees-native `axis_angle(axis, angle_degrees)`; the radians-based
`quat.from_axis_angle` is never given to generated code, closing off that failure
mode entirely rather than relying on a prompt instruction.

**Joint name lookup is fuzzy on purpose.** This rig's own naming is irregular
(`LeftLeg_063` but `RightLeg_00`, not the `_064` the sibling pattern would
suggest), which is exactly the kind of thing an LLM guessing a joint name gets
plausibly wrong. `gltf_utils.find_joint` falls back to matching with trailing
`_NN` suffixes stripped rather than crashing on a `None`.

**`exec()`-ing LLM-written code needs a real sandbox, not just restraint.**
`llm_animator._run_generated_code` uses a minimal `__builtins__`, a regex
blocklist (`import`, `open(`, `os.`, dunders, etc.), and a thread-based timeout —
plus a retry loop that feeds the exact error back to the model. Two bugs already
came from getting this wrong: (1) errors escaping `_run_generated_code` uncaught
skipped the retry-with-feedback path entirely; (2) the original timeout used
`signal.alarm`, which only works on the main thread — it broke silently (always
timed out, ate both retries) once generation moved into pywebview's background
worker thread. Current timeout uses `ThreadPoolExecutor` + `future.result(timeout=...)`.

**pywebview's macOS backend does not handle `data:` URLs as a window's initial
URL** — it routes them through its own local server and 404s. The loading/error
page (`status.html`) is a real served file for this reason, not an inline string.

## Current state / known rough edges

- Wave-type animations are still not fully convincing ("hands facing outwards,
  arms rotating, not quite waving") even from Claude Sonnet 5 — the sandbox API
  and prompt guardrails above fix outright broken results (wrong side, crossed
  behind the head, frozen limbs) but don't yet produce great *motion quality*.
  Next lever to pull is probably richer worked examples in the system prompt, or
  a wider validation pass (e.g. simulate forward kinematics on the result and
  reject/retry if a limb ends up implausibly positioned) rather than trusting the
  LLM's spatial reasoning alone.
- Only tested against one rig (`rigged_human.glb`, Mixamo joint naming). Nothing
  here is Mixamo-specific by design (`find_joint`/`local_rotation_for_world_delta`
  are generic), but it's untested against other skeletons/conventions.
- `animator.GENERATORS` (`wave`, `spin`) is a last-resort fallback for when the
  LLM path fails outright, not a serious animation system — don't extend it,
  extend the LLM prompt/tools instead.
