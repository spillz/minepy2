/**
 * Background chunk generation worker.
 * @module worker
 */
import { globalGenerator } from './mapgen.js';

/**
 * Worker entry: generates procedural terrain using deterministic noise.
 */
self.onmessage = (event) => {
  const { type, cx, cz } = event.data;
  if (type !== 'generate') return;
  const buffer = globalGenerator.generate(cx, cz).buffer;
  self.postMessage({ cx, cz, buffer }, [buffer]);
};
