// Compiles and fits off the main thread, so the page can draw each snapshot as
// advi() reports it instead of freezing until the one call returns.

import { compileDecoder, trainDecoder } from "./app.js";

let compiled;

self.onmessage = async ({ data }) => {
  try {
    if (data.type === "compile") {
      compiled = await compileDecoder(data);
      const { sampler, ...stats } = compiled;
      self.postMessage({ type: "compiled", ...stats });
    } else if (data.type === "train") {
      const r = trainDecoder(compiled, {
        onSnapshot: (iter, mu, elbo) =>
          self.postMessage({ type: "snapshot", iter, mu, elbo }, [mu.buffer, elbo.buffer]),
      });
      self.postMessage({ type: "trained", ...r }, [r.mu.buffer]);
    }
  } catch (e) {
    self.postMessage({ type: "error", message: e?.message ?? String(e) });
  }
};
