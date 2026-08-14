// The learning curve. Plain canvas 2D — a chart library would be more code
// than the ~60 lines of drawing this needs.
//
// Two lines on purpose:
//   reward  (what the algorithm is actually maximising — noisy, it's meant to be)
//   best    (the high-water mark — the "it's still improving" line)
// plus a faint match line, so a child can see reward and similarity-to-target
// come apart when the sliders reward something other than copying.

import { onMounted, ref, watch } from "vue";

const COLOURS = {
  reward: "#6c5ce7",
  best: "#14a06a",
  match: "#f0a020",
  grid: "#d9e0f5",
  axis: "#6a7196",
};

export const RewardChart = {
  props: {
    episodes: { type: Array, required: true },  // [{episode, reward, best_reward, match}]
    total: { type: Number, default: 300 },
  },
  setup(props) {
    const canvas = ref(null);

    function draw() {
      const el = canvas.value;
      if (!el) return;
      const dpr = Math.min(devicePixelRatio, 2);
      const w = el.clientWidth;
      const h = el.clientHeight;
      if (!w || !h) return;
      el.width = w * dpr;
      el.height = h * dpr;

      const ctx = el.getContext("2d");
      ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
      ctx.clearRect(0, 0, w, h);

      const pad = { top: 10, right: 8, bottom: 20, left: 30 };
      const plotW = w - pad.left - pad.right;
      const plotH = h - pad.top - pad.bottom;

      // Reward is always 0..1, so the y-axis is fixed. An auto-scaling axis
      // would make a flat curve look like progress.
      const x = (episode) => pad.left + (episode / Math.max(props.total, 1)) * plotW;
      const y = (value) => pad.top + (1 - value) * plotH;

      ctx.strokeStyle = COLOURS.grid;
      ctx.fillStyle = COLOURS.axis;
      ctx.font = "10px system-ui, sans-serif";
      ctx.lineWidth = 1;
      for (const value of [0, 0.25, 0.5, 0.75, 1]) {
        ctx.beginPath();
        ctx.moveTo(pad.left, y(value));
        ctx.lineTo(w - pad.right, y(value));
        ctx.stroke();
        ctx.fillText(value.toFixed(2), 2, y(value) + 3);
      }
      ctx.fillText("episode", w - pad.right - 44, h - 6);

      const line = (key, colour, width, alpha = 1) => {
        if (props.episodes.length < 2) return;
        ctx.globalAlpha = alpha;
        ctx.strokeStyle = colour;
        ctx.lineWidth = width;
        ctx.lineJoin = "round";
        ctx.beginPath();
        props.episodes.forEach((point, index) => {
          const px = x(point.episode);
          const py = y(Math.min(1, Math.max(0, point[key] ?? 0)));
          if (index === 0) ctx.moveTo(px, py);
          else ctx.lineTo(px, py);
        });
        ctx.stroke();
        ctx.globalAlpha = 1;
      };

      line("match", COLOURS.match, 1.5, 0.5);
      line("reward", COLOURS.reward, 1.5);
      line("best_reward", COLOURS.best, 2);
    }

    onMounted(() => {
      draw();
      new ResizeObserver(draw).observe(canvas.value);
    });
    watch(() => props.episodes.length, draw);

    return { canvas };
  },
  template: `
    <div class="chart">
      <canvas ref="canvas"></canvas>
      <div class="chart-key">
        <span><i style="background:#14a06a"></i>best score</span>
        <span><i style="background:#6c5ce7"></i>score each try</span>
        <span><i style="background:#f0a020"></i>match to target</span>
      </div>
    </div>
  `,
};
