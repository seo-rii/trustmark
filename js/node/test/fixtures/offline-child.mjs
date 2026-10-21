// Child process for offline.test.mjs, run with the no-network.cjs preload.
// Usage: node --require no-network.cjs offline-child.mjs <mode> <modelDir> <image>
import fs from 'node:fs';
import https from 'node:https';
import { createRequire } from 'node:module';

const require = createRequire(import.meta.url);
const { createTrustmark } = require('../..');
const [mode, modelDir, imagePath] = process.argv.slice(2);
const image = fs.readFileSync(imagePath);
const out = {};

if (mode === 'control') {
  // The guard works: downloads and direct requests are blocked and recorded
  const tm = createTrustmark({ modelDir, allowDownload: true });
  out.encode = await tm.encode(image, '1').then(() => 'ok', (e) => e.code);
  await tm.close();
  out.https = await new Promise((resolve) => {
    try {
      https.get('https://example.com/', () => resolve('ok')).on('error', (e) => resolve(e.code));
    } catch (e) {
      resolve(e.code);
    }
  });
} else {
  const tm = createTrustmark({ modelDir });
  const payload = '1011001110001111000011111000001111110000001111111000000011111';
  try {
    const encoded = await tm.encode(image, payload);
    const decoded = await tm.decode(encoded.bytes);
    out.encode = 'ok';
    out.exact = decoded.payload === payload;
  } catch (error) {
    out.encode = error.code;
  }
  const decoded = await tm.decode(image);
  out.decode = decoded.status === 'error' ? decoded.error.code : decoded.status;
  await tm.close();
}
console.log(JSON.stringify(out));
