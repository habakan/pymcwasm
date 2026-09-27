// Loads and fits off the main thread, so the page can draw each snapshot as
// advi() reports it instead of freezing until the one call returns.

import { loadDecoder, trainDecoder } from "./app.js";

let loaded;

self.onmessage = async ({ data }) => {
  try {
    if (data.type === "load") {
      loaded = await loadDecoder();
      const { sampler, ...stats } = loaded;
      self.postMessage({ type: "loaded", ...stats });
    } else if (data.type === "train") {
      const r = trainDecoder(loaded, {
        onSnapshot: (iter, mu, elbo) =>
          self.postMessage({ type: "snapshot", iter, mu, elbo }, [mu.buffer, elbo.buffer]),
      });
      self.postMessage({ type: "trained", ...r }, [r.mu.buffer]);
    }
  } catch (e) {
    self.postMessage({ type: "error", message: e?.message ?? String(e) });
  }
};
