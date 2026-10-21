// Compile-only checks of index.d.ts (run by types.test.mjs with tsc --noEmit). The lines after
// each expect-error directive at the end must fail to type-check.
import { capabilities, createTrustmark, encodeVerified, TrustmarkError, WatermarkVerificationError } from '../..';
import type { DecodeResult, EncodeResult, Schema, Trustmark, TrustmarkErrorCode, VerifiedEncodeResult } from '../..';

async function main(): Promise<void> {
  const tm: Trustmark = createTrustmark({
    modelDir: '/models',
    variant: 'Q',
    schema: 'BCH_5',
    allowDownload: false,
    expectedDigests: { 'encoder_Q.onnx': '0'.repeat(64) },
    executionProviders: ['cuda', { name: 'cuda', deviceId: 1 }, 'cpu'],
    sessionOptions: { intraOpNumThreads: 2 },
    maxPixels: 10_000_000,
    maxConcurrency: 2,
    maxQueue: Infinity,
    orientation: 'apply',
  });
  const capacity: number = tm.capacity;
  const encoded: EncodeResult = await tm.encode(Buffer.from([]), '0101', { format: 'jpeg', quality: 90, strength: 1.2, background: [255, 255, 255], signal: AbortSignal.timeout(1000) });
  const bytes: Buffer = encoded.bytes;
  const schema: Schema = encoded.schema;
  await tm.encode({ data: new Uint8ClampedArray(4), width: 1, height: 1 }, '1');
  await tm.encode({ data: new Uint8Array(3), width: 1, height: 1, channels: 3 }, '1');

  const decoded: DecodeResult = await tm.decode(bytes, { schemaPolicy: 'BCH_5' });
  if (decoded.status === 'detected') {
    const payload: string = decoded.payload;
    const recovered: Schema = decoded.schema;
    void payload; void recovered;
  } else if (decoded.status === 'error') {
    const code: TrustmarkErrorCode = decoded.error.code;
    void code;
  } else {
    const nothing: null = decoded.payload;
    void nothing;
  }
  try {
    const verified: VerifiedEncodeResult = await encodeVerified(tm, bytes, '0101', { format: 'jpeg', quality: 85, retryStrengths: [1.5, 2], verifier: tm, signal: AbortSignal.timeout(1000) });
    const digest: string = verified.sha256;
    const attempts: number = verified.verification.attempts.length;
    void digest; void attempts;
  } catch (error) {
    if (error instanceof WatermarkVerificationError) {
      const reason: 'not_detected' | 'payload_mismatch' | 'schema_mismatch' | 'decode_error' | 'bytes_changed' = error.reason;
      const code: 'VERIFICATION_FAILED' = error.code;
      void reason; void code;
    }
  }
  const status = await tm.ready();
  const backend: string | null = status.models.encoder.backend;
  const canEmbedP: boolean | undefined = capabilities().variants.P?.encode;
  await tm.close();

  try {
    await tm.decode(bytes);
  } catch (error) {
    if (error instanceof TrustmarkError && error.code === 'CLOSED') return;
  }
  void capacity; void schema; void backend; void canEmbedP;

  // @ts-expect-error payloads are bit strings, never numbers
  await tm.encode(bytes, 12345);
  // @ts-expect-error modelDir is required
  createTrustmark({ variant: 'Q' });
  // @ts-expect-error unknown schema
  createTrustmark({ modelDir: '/m', schema: 'BCH_6' });
  // @ts-expect-error unsupported output format
  await tm.encode(bytes, '1', { format: 'webp' });
  // @ts-expect-error file paths are not accepted
  await tm.decode('/tmp/image.png');
  // @ts-expect-error retry strengths are numbers
  await encodeVerified(tm, bytes, '1', { retryStrengths: ['2'] });
  // @ts-expect-error a not_detected result has no payload string
  if (decoded.status === 'not_detected') { const p: string = decoded.payload; void p; }
}
void main;
