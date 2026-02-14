/**
 * TPC Certificate parser and local verifier.
 *
 * Binary format:
 *   Header (64 bytes):     magic(4) + version(4) + UUID(16) + timestamp(8) + solver_hash(32)
 *   Layers (variable):     [json_len(4) + json + blob_count(4) + blobs]*
 *   Signature (128 bytes): pubkey(32) + signature(64) + content_hash(32)
 */

import { sha256 } from '@noble/hashes/sha256';
import { verify as ed25519Verify } from '@noble/ed25519';
import { InvalidCertificate, InvalidHash, InvalidSignature } from './errors';

// ═══════════════════════════════════════════════════════════════════════════
// Constants
// ═══════════════════════════════════════════════════════════════════════════

const TPC_MAGIC = new Uint8Array([0x54, 0x50, 0x43, 0x01]); // "TPC\x01"
const HEADER_SIZE = 64;
const SIGNATURE_SECTION_SIZE = 128;
const PUBKEY_SIZE = 32;
const SIGNATURE_SIZE = 64;

// ═══════════════════════════════════════════════════════════════════════════
// Types
// ═══════════════════════════════════════════════════════════════════════════

export interface CertificateHeader {
  magic: Uint8Array;
  version: number;
  certificateId: string;
  timestampNs: bigint;
  solverHash: Uint8Array;
  timestamp: Date;
}

export interface Blob {
  name: string;
  data: Uint8Array;
}

export interface Layer {
  metadata: Record<string, unknown>;
  blobs: Blob[];
}

export interface SignatureSection {
  pubkey: Uint8Array;
  signature: Uint8Array;
  contentHash: Uint8Array;
  pubkeyHex: string;
  signatureHex: string;
  contentHashHex: string;
}

export interface VerificationResult {
  valid: boolean;
  hashValid: boolean;
  signatureValid: boolean;
  certificateId: string;
  signerPubkey: string;
  error?: string;
}

// ═══════════════════════════════════════════════════════════════════════════
// Helpers
// ═══════════════════════════════════════════════════════════════════════════

function toHex(bytes: Uint8Array): string {
  return Array.from(bytes)
    .map((b) => b.toString(16).padStart(2, '0'))
    .join('');
}

function fromHex(hex: string): Uint8Array {
  const clean = hex.startsWith('0x') ? hex.slice(2) : hex;
  const bytes = new Uint8Array(clean.length / 2);
  for (let i = 0; i < bytes.length; i++) {
    bytes[i] = parseInt(clean.substring(i * 2, i * 2 + 2), 16);
  }
  return bytes;
}

function readU32LE(data: Uint8Array, offset: number): number {
  return (
    data[offset] |
    (data[offset + 1] << 8) |
    (data[offset + 2] << 16) |
    (data[offset + 3] << 24)
  ) >>> 0;
}

function readU16LE(data: Uint8Array, offset: number): number {
  return data[offset] | (data[offset + 1] << 8);
}

function readI64LE(data: Uint8Array, offset: number): bigint {
  const view = new DataView(data.buffer, data.byteOffset + offset, 8);
  return view.getBigInt64(0, true);
}

function uuidFromBytes(bytes: Uint8Array): string {
  const hex = toHex(bytes);
  return [
    hex.slice(0, 8),
    hex.slice(8, 12),
    hex.slice(12, 16),
    hex.slice(16, 20),
    hex.slice(20, 32),
  ].join('-');
}

function arraysEqual(a: Uint8Array, b: Uint8Array): boolean {
  if (a.length !== b.length) return false;
  for (let i = 0; i < a.length; i++) {
    if (a[i] !== b[i]) return false;
  }
  return true;
}

// ═══════════════════════════════════════════════════════════════════════════
// Certificate Class
// ═══════════════════════════════════════════════════════════════════════════

export class Certificate {
  private readonly _raw: Uint8Array;
  private _header: CertificateHeader;
  private _layers: Layer[];
  private _signatureSection: SignatureSection;

  constructor(raw: Uint8Array) {
    this._raw = raw;
    const { header, layers, signatureSection } = Certificate.parse(raw);
    this._header = header;
    this._layers = layers;
    this._signatureSection = signatureSection;
  }

  /** Create from hex string. */
  static fromHex(hex: string): Certificate {
    return new Certificate(fromHex(hex));
  }

  /** Create from Buffer (Node.js). */
  static fromBuffer(buf: Buffer): Certificate {
    return new Certificate(new Uint8Array(buf));
  }

  // ── Properties ──────────────────────────────────────────────────────

  get raw(): Uint8Array {
    return this._raw;
  }

  get header(): CertificateHeader {
    return this._header;
  }

  get layers(): Layer[] {
    return this._layers;
  }

  get signatureSection(): SignatureSection {
    return this._signatureSection;
  }

  get certificateId(): string {
    return this._header.certificateId;
  }

  get domain(): string | undefined {
    for (const layer of this._layers) {
      if (typeof layer.metadata['domain'] === 'string') {
        return layer.metadata['domain'] as string;
      }
    }
    return undefined;
  }

  get contentHash(): string {
    const content = this._raw.slice(0, this._raw.length - SIGNATURE_SECTION_SIZE);
    return toHex(sha256(content));
  }

  get sizeBytes(): number {
    return this._raw.length;
  }

  // ── Parsing ─────────────────────────────────────────────────────────

  private static parse(data: Uint8Array): {
    header: CertificateHeader;
    layers: Layer[];
    signatureSection: SignatureSection;
  } {
    const minSize = HEADER_SIZE + SIGNATURE_SECTION_SIZE;
    if (data.length < minSize) {
      throw new InvalidCertificate(
        `Certificate too short: ${data.length} bytes (minimum ${minSize})`
      );
    }

    // Validate magic
    if (!arraysEqual(data.slice(0, 4), TPC_MAGIC)) {
      throw new InvalidCertificate(
        `Invalid TPC magic: ${toHex(data.slice(0, 4))} (expected ${toHex(TPC_MAGIC)})`
      );
    }

    // Header (64 bytes)
    const version = readU32LE(data, 4);
    const certIdBytes = data.slice(8, 24);
    const timestampNs = readI64LE(data, 24);
    const solverHash = data.slice(32, 64);

    const header: CertificateHeader = {
      magic: data.slice(0, 4),
      version,
      certificateId: uuidFromBytes(certIdBytes),
      timestampNs,
      solverHash,
      timestamp: new Date(Number(timestampNs / 1_000_000n)),
    };

    // Signature section (last 128 bytes)
    const sigStart = data.length - SIGNATURE_SECTION_SIZE;
    const sigData = data.slice(sigStart);

    const pubkey = sigData.slice(0, PUBKEY_SIZE);
    const signature = sigData.slice(PUBKEY_SIZE, PUBKEY_SIZE + SIGNATURE_SIZE);
    const contentHash = sigData.slice(PUBKEY_SIZE + SIGNATURE_SIZE);

    const signatureSection: SignatureSection = {
      pubkey,
      signature,
      contentHash,
      pubkeyHex: toHex(pubkey),
      signatureHex: toHex(signature),
      contentHashHex: toHex(contentHash),
    };

    // Layers
    const layers: Layer[] = [];
    let offset = HEADER_SIZE;
    const layerEnd = sigStart;
    const decoder = new TextDecoder();

    while (offset < layerEnd) {
      if (offset + 4 > layerEnd) break;

      const jsonLen = readU32LE(data, offset);
      offset += 4;

      if (offset + jsonLen > layerEnd) break;

      const jsonBytes = data.slice(offset, offset + jsonLen);
      offset += jsonLen;

      let metadata: Record<string, unknown>;
      try {
        metadata = JSON.parse(decoder.decode(jsonBytes));
      } catch {
        metadata = { raw: toHex(jsonBytes) };
      }

      if (offset + 4 > layerEnd) {
        layers.push({ metadata, blobs: [] });
        break;
      }

      const blobCount = readU32LE(data, offset);
      offset += 4;

      const blobs: Blob[] = [];
      for (let i = 0; i < blobCount; i++) {
        if (offset + 2 > layerEnd) break;

        const nameLen = readU16LE(data, offset);
        offset += 2;

        if (offset + nameLen > layerEnd) break;

        const name = decoder.decode(data.slice(offset, offset + nameLen));
        offset += nameLen;

        if (offset + 4 > layerEnd) break;

        const blobLen = readU32LE(data, offset);
        offset += 4;

        if (offset + blobLen > layerEnd) break;

        const blobData = data.slice(offset, offset + blobLen);
        offset += blobLen;

        blobs.push({ name, data: blobData });
      }

      layers.push({ metadata, blobs });
    }

    return { header, layers, signatureSection };
  }

  // ── Verification ────────────────────────────────────────────────────

  /** Verify certificate integrity (hash + Ed25519 signature). */
  async verify(): Promise<VerificationResult> {
    const sig = this._signatureSection;
    const certId = this._header.certificateId;

    // Step 1: Verify content hash
    const content = this._raw.slice(0, this._raw.length - SIGNATURE_SECTION_SIZE);
    const computedHash = sha256(content);
    const hashValid = arraysEqual(computedHash, sig.contentHash);

    if (!hashValid) {
      return {
        valid: false,
        hashValid: false,
        signatureValid: false,
        certificateId: certId,
        signerPubkey: sig.pubkeyHex,
        error: 'Content hash mismatch',
      };
    }

    // Step 2: Verify Ed25519 signature
    let signatureValid: boolean;
    try {
      signatureValid = await ed25519Verify(sig.signature, computedHash, sig.pubkey);
    } catch (e) {
      return {
        valid: false,
        hashValid: true,
        signatureValid: false,
        certificateId: certId,
        signerPubkey: sig.pubkeyHex,
        error: `Signature verification error: ${e}`,
      };
    }

    return {
      valid: hashValid && signatureValid,
      hashValid,
      signatureValid,
      certificateId: certId,
      signerPubkey: sig.pubkeyHex,
      error: signatureValid ? undefined : 'Ed25519 signature invalid',
    };
  }

  /** Verify, throwing on failure. */
  async verifyStrict(): Promise<void> {
    const result = await this.verify();
    if (!result.hashValid) {
      throw new InvalidHash(`Content hash mismatch for ${result.certificateId}`);
    }
    if (!result.signatureValid) {
      throw new InvalidSignature(
        `Signature invalid for ${result.certificateId}: ${result.error}`
      );
    }
  }

  /** Human-readable summary. */
  summary(): Record<string, unknown> {
    return {
      certificateId: this._header.certificateId,
      version: this._header.version,
      timestamp: this._header.timestamp.toISOString(),
      domain: this.domain,
      solverHash: toHex(this._header.solverHash),
      signerPubkey: this._signatureSection.pubkeyHex,
      contentHash: this.contentHash,
      sizeBytes: this.sizeBytes,
      layers: this._layers.length,
      totalBlobs: this._layers.reduce((sum, l) => sum + l.blobs.length, 0),
    };
  }
}
