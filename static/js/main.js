import { createApp } from "vue";
import { App } from "./App.js";
import { boot } from "./store.js";

// Load the schema first so components can read bone names and reward labels
// synchronously in setup() rather than guarding every access.
await boot();
createApp(App).mount("#app");
