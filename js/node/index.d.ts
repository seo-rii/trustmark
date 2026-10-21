/// <reference types="node" />

/** TrustMark model variant. Which ones can embed or decode depends on the JavaScript core; see capabilities(). */
export type Variant = 'Q' | 'P' | 'B' | 'C';

/** BCH error-correction schema; the payload capacity is 40, 61, 68 or 75 bits. */
export type Schema = 'BCH_SUPER' | 'BCH_5' | 'BCH_4' | 'BCH_3';

/**
 * A payload as a string of '0' and '1' characters (for BCH_5 up to 61 of them). Payloads are
 * never JavaScript numbers, which cannot hold 61 bits exactly.
 */
export type Bitstring = string;

export type OutputFormat = 'png' | 'jpeg';

/** Which decoded schemas count as a detection; see DecodeOptions.schemaPolicy. */
export type SchemaPolicy = 'signalled' | 'any' | Schema;

/** An ONNX Runtime execution provider, e.g. 'cpu', 'cuda' or { name: 'cuda', deviceId: 1 }. */
export type ExecutionProvider =
  | 'cpu'
  | 'cuda'
  | 'tensorrt'
  | 'dml'
  | 'coreml'
  | 'webgpu'
  | (string & {})
  | { readonly name: string; readonly [option: string]: unknown };

/** Raw 8-bit pixels, row by row: RGBA by default, or RGB with channels: 3. */
export interface RawImage {
  readonly data: Uint8Array | Uint8ClampedArray;
  readonly width: number;
  readonly height: number;
  readonly channels?: 3 | 4;
}

/** PNG or JPEG file bytes (a Buffer is a Uint8Array), or raw pixels. */
export type ImageInput = Uint8Array | ArrayBuffer | RawImage;

export interface TrustmarkOptions {
  /** Model variant (default 'Q'). */
  variant?: Variant;
  /** Schema used when embedding (default 'BCH_5'). Decoding recovers the schema from the image. */
  schema?: Schema;
  /** Directory that holds resizer.onnx and encoder_<V>.onnx / decoder_<V>.onnx, or receives them with allowDownload. */
  modelDir: string;
  /** Download missing models from modelBaseUrl (default false: the network is never used). */
  allowDownload?: boolean;
  /** Where models are downloaded from (https; http only for localhost). */
  modelBaseUrl?: string;
  /** Expected SHA-256 hex digests by file name, e.g. { 'encoder_Q.onnx': '…' }, checked on every load. */
  expectedDigests?: Readonly<Record<string, string>>;
  /** Refuse to load a model that has no entry in expectedDigests (default false). */
  requireDigests?: boolean;
  /** Largest model file accepted, in bytes (default 256 MiB). */
  maxModelBytes?: number;
  /** Providers tried in order for the encoder and decoder (default ['cpu']). The resizer (decoding only) always runs on the CPU. */
  executionProviders?: readonly ExecutionProvider[];
  /** Extra ONNX Runtime session options, e.g. { intraOpNumThreads: 4 }. executionProviders is ignored here. */
  sessionOptions?: Readonly<Record<string, unknown>>;
  /** Largest image accepted, in pixels (default 25,000,000). */
  maxPixels?: number;
  /** Jobs (encode or decode calls) that run at the same time (default 1). */
  maxConcurrency?: number;
  /** Jobs that may wait for a slot before calls fail with QUEUE_FULL (default 100; Infinity allowed). */
  maxQueue?: number;
  /** EXIF orientation of PNG/JPEG input: 'apply' rotates the pixels as viewers do (default), 'ignore' uses the stored pixels. */
  orientation?: 'apply' | 'ignore';
  /** fetch implementation for downloads (default globalThis.fetch). */
  fetch?: (input: string, init?: { signal?: AbortSignal; redirect?: 'follow' }) => Promise<Response>;
  /** ONNX Runtime module to use instead of onnxruntime-node. */
  ort?: unknown;
}

export interface EncodeOptions {
  /** Watermark strength (default 1.0, like WM_STRENGTH in Python). */
  strength?: number;
  /** Output format (default 'png', lossless). */
  format?: OutputFormat;
  /** JPEG quality from 1 to 100 (default 90). Only for format 'jpeg'. */
  quality?: number;
  /** Colour to flatten transparent pixels onto before embedding, required for JPEG output of images with transparency. */
  background?: readonly [number, number, number];
  signal?: AbortSignal;
}

export interface EncodeResult {
  /** The encoded output file. */
  bytes: Buffer;
  format: OutputFormat;
  mimeType: 'image/png' | 'image/jpeg';
  /** JPEG quality, or null for PNG. */
  quality: number | null;
  /** The payload as embedded: padded with zeros to the schema capacity. */
  payload: Bitstring;
  schema: Schema;
  variant: Variant;
  strength: number;
  /** Execution provider of the encoder session, e.g. 'cpu' or 'cuda'. */
  backend: string;
  width: number;
  height: number;
}

export interface DecodeOptions {
  /**
   * Which schemas count as a detection. 'signalled' (default) accepts only the schema in the
   * packet's version bits, like the Python implementation; 'any' also accepts the other schemas,
   * which tolerates corrupt version bits but has several times more false positives; a schema
   * name accepts only that schema, whatever the version bits say.
   */
  schemaPolicy?: SchemaPolicy;
  signal?: AbortSignal;
}

interface DecodeResultBase {
  variant: Variant;
  /** Execution provider of the decoder session, or null if it never loaded. */
  backend: string | null;
}

export interface DecodeDetected extends DecodeResultBase {
  status: 'detected';
  /** The recovered payload, at the full capacity of the schema. */
  payload: Bitstring;
  schema: Schema;
  /** Bits corrected by the BCH code. */
  bitFlips: number | null;
  /** C2PA soft binding assertion for the watermark. */
  softBinding: Record<string, unknown> | null;
  error: null;
}

export interface DecodeNotDetected extends DecodeResultBase {
  status: 'not_detected';
  payload: null;
  schema: null;
  bitFlips: null;
  softBinding: null;
  error: null;
}

/** The image could not be checked (unreadable image, missing model, inference failure): it may still carry a watermark. */
export interface DecodeFailed extends DecodeResultBase {
  status: 'error';
  payload: null;
  schema: null;
  bitFlips: null;
  softBinding: null;
  error: TrustmarkError;
}

export type DecodeResult = DecodeDetected | DecodeNotDetected | DecodeFailed;

export type ModelRole = 'resizer' | 'encoder' | 'decoder';

export interface ModelStatus {
  file: string;
  status: 'idle' | 'loading' | 'ready' | 'error' | 'closed';
  requestedProviders: string[];
  /** Provider the session was created on, or null if not loaded. */
  backend: string | null;
  /** Providers that failed before `backend` was used. */
  fallbacks: Array<{ provider: string; error: string }>;
  path: string;
  sha256: string | null;
  size: number | null;
  /** True when the file matched an expected digest. */
  verified: boolean;
  downloaded: boolean;
  loadMs: number | null;
  error: { code: TrustmarkErrorCode | null; message: string } | null;
}

export interface TrustmarkStatus {
  variant: Variant;
  schema: Schema;
  closed: boolean;
  models: Record<ModelRole, ModelStatus>;
}

export interface Trustmark {
  readonly variant: Variant;
  readonly schema: Schema;
  /** Payload capacity of the schema in bits. */
  readonly capacity: number;
  /** Embeds a payload and returns the encoded image. Rejects instead of returning an image without a watermark. */
  encode(input: ImageInput, payload: Bitstring, options?: EncodeOptions): Promise<EncodeResult>;
  /** Reads a watermark. Resolves with status 'error' when the image could not be checked; rejects on invalid arguments, abort, QUEUE_FULL and CLOSED. */
  decode(input: ImageInput, options?: DecodeOptions): Promise<DecodeResult>;
  /** Loads the models now instead of on first use. */
  ready(options?: { signal?: AbortSignal }): Promise<TrustmarkStatus>;
  status(): TrustmarkStatus;
  /** Rejects waiting jobs, waits for running ones and releases the sessions. Idempotent. */
  close(): Promise<void>;
}

export function createTrustmark(options: TrustmarkOptions): Trustmark;

export interface Capabilities {
  variants: Partial<Record<Variant, { encode: boolean; decode: boolean }>>;
  schemas: Array<{ name: Schema; capacity: number }>;
}

/** Variants and schemas supported by the JavaScript core. */
export function capabilities(): Capabilities;

export type TrustmarkErrorCode =
  | 'INVALID_ARGUMENT'
  | 'UNSUPPORTED'
  | 'INVALID_IMAGE'
  | 'IMAGE_TOO_LARGE'
  | 'MODEL_NOT_FOUND'
  | 'MODEL_DOWNLOAD_FAILED'
  | 'MODEL_INTEGRITY'
  | 'MODEL_LOAD_FAILED'
  | 'INFERENCE_FAILED'
  | 'QUEUE_FULL'
  | 'CLOSED';

export class TrustmarkError extends Error {
  constructor(code: TrustmarkErrorCode, message: string, options?: { cause?: unknown; details?: unknown });
  readonly code: TrustmarkErrorCode;
  readonly details?: unknown;
}
