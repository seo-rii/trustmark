// Command-line demo of release.mjs: releases a "generated" image file under a job id.
//
//   TRUSTMARK_MODEL_DIR=/path/to/models node examples/verified-release/demo.mjs \
//     generated.png /srv/private /srv/public job-0001 [png|jpeg]
//
// Run it twice with the same job id: the second run reports the existing release.

import fs from 'node:fs/promises';
import { createRequire } from 'node:module';

import { createReleaser } from './release.mjs';

const require = createRequire(import.meta.url);
const { createTrustmark } = require('../..');

const [input, privateDir, publicDir, jobId, format = 'png'] = process.argv.slice(2);
if (!input || !privateDir || !publicDir || !jobId) {
  console.error('usage: node examples/verified-release/demo.mjs <generated.png|jpg> <privateDir> <publicDir> <jobId> [png|jpeg]');
  process.exit(2);
}

let trustmark = null;
try {
  trustmark = createTrustmark({ modelDir: process.env.TRUSTMARK_MODEL_DIR });
  const releaser = createReleaser({ trustmark, privateDir, publicDir, format, retryStrengths: [1.25, 1.5] });
  const released = await releaser.release(jobId, await fs.readFile(input), { signal: AbortSignal.timeout(120000) });
  console.log(`${released.reused ? 'already released' : 'released'} ${released.path} (sha256 ${released.sha256}, payload ${released.payload})`);
} catch (error) {
  // Nothing was published: the public directory has no file for this job
  console.error(`not released: ${error.code || error.name}${error.reason ? ` (${error.reason})` : ''}: ${error.message}`);
  process.exitCode = 1;
} finally {
  if (trustmark) await trustmark.close();
}
