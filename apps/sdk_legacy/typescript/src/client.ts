/**
 * HTTP client for the TPC Certificate Authority.
 */

import { Certificate } from './certificate';
import { CertificateNotFound, TPCError } from './errors';

// ═══════════════════════════════════════════════════════════════════════════
// Types
// ═══════════════════════════════════════════════════════════════════════════

export interface ClientConfig {
  /** Base URL of the certificate authority. */
  caUrl: string;
  /** API key for authentication. */
  apiKey?: string;
  /** Request timeout in milliseconds (default: 30000). */
  timeout?: number;
}

export interface IssueRequest {
  /** Physics domain. */
  domain: string;
  /** Proof bytes as hex string. */
  proof: string;
  /** Public input values (hex strings). */
  publicInputs?: string[];
  /** SHA-256 hash of the solver binary (hex). */
  solverHash?: string;
  /** Additional metadata. */
  metadata?: Record<string, string>;
}

export interface IssueResult {
  certificateId: string;
  contentHash: string;
  signerPubkey: string;
  domain: string;
  sizeBytes: number;
  issuedAt: string;
  onChainStatus: string;
}

interface VerifyResponse {
  valid: boolean;
  hash_valid: boolean;
  signature_valid: boolean;
  signer_pubkey: string;
  certificate_id: string;
  domain?: string;
  error?: string;
}

interface HealthResponse {
  status: string;
  service: string;
  version: string;
  uptime_seconds: number;
  signer_pubkey: string;
}

interface StatsResponse {
  total_issued: number;
  total_verified: number;
  total_failed: number;
  uptime_seconds: number;
  signer_pubkey: string;
  storage_dir: string;
  certificates_by_domain: Record<string, number>;
}

// ═══════════════════════════════════════════════════════════════════════════
// Client
// ═══════════════════════════════════════════════════════════════════════════

export class TPCClient {
  private readonly baseUrl: string;
  private readonly headers: Record<string, string>;
  private readonly timeout: number;

  constructor(config: ClientConfig) {
    this.baseUrl = config.caUrl.replace(/\/$/, '');
    this.timeout = config.timeout ?? 30_000;
    this.headers = {
      'Content-Type': 'application/json',
    };
    if (config.apiKey) {
      this.headers['Authorization'] = `Bearer ${config.apiKey}`;
    }
  }

  /** Issue a new TPC certificate. */
  async issue(req: IssueRequest): Promise<IssueResult> {
    const body = {
      domain: req.domain,
      proof: req.proof,
      public_inputs: req.publicInputs ?? [],
      solver_hash: req.solverHash,
      metadata: req.metadata,
    };

    const resp = await this.fetch(`${this.baseUrl}/v1/certificates/issue`, {
      method: 'POST',
      body: JSON.stringify(body),
    });

    if (resp.status !== 201) {
      const text = await resp.text();
      throw new TPCError(`Certificate issuance failed (HTTP ${resp.status}): ${text}`);
    }

    const data = await resp.json();
    return {
      certificateId: data.certificate_id,
      contentHash: data.content_hash,
      signerPubkey: data.signer_pubkey,
      domain: data.domain,
      sizeBytes: data.size_bytes,
      issuedAt: data.issued_at,
      onChainStatus: data.on_chain_status,
    };
  }

  /** Retrieve a certificate by ID. */
  async getCertificate(id: string): Promise<Certificate> {
    const resp = await this.fetch(`${this.baseUrl}/v1/certificates/${id}`, {
      method: 'GET',
    });

    if (resp.status === 404) {
      throw new CertificateNotFound(`Certificate ${id} not found`);
    }

    if (resp.status !== 200) {
      const text = await resp.text();
      throw new TPCError(`Failed to retrieve certificate (HTTP ${resp.status}): ${text}`);
    }

    const buffer = await resp.arrayBuffer();
    return new Certificate(new Uint8Array(buffer));
  }

  /** Verify a certificate via the CA. */
  async verify(opts: {
    certificate?: Certificate;
    certificateId?: string;
  }): Promise<VerifyResponse> {
    const body: Record<string, string> = {};

    if (opts.certificate) {
      body.certificate = Array.from(opts.certificate.raw)
        .map((b) => b.toString(16).padStart(2, '0'))
        .join('');
    } else if (opts.certificateId) {
      body.certificate_id = opts.certificateId;
    } else {
      throw new TPCError('Provide either certificate or certificateId');
    }

    const resp = await this.fetch(`${this.baseUrl}/v1/certificates/verify`, {
      method: 'POST',
      body: JSON.stringify(body),
    });

    if (resp.status !== 200 && resp.status !== 422) {
      const text = await resp.text();
      throw new TPCError(`Verification failed (HTTP ${resp.status}): ${text}`);
    }

    return resp.json();
  }

  /** Get CA statistics. */
  async stats(): Promise<StatsResponse> {
    const resp = await this.fetch(`${this.baseUrl}/v1/certificates/stats`, {
      method: 'GET',
    });
    return resp.json();
  }

  /** Check CA health. */
  async health(): Promise<HealthResponse> {
    const resp = await this.fetch(`${this.baseUrl}/health`, {
      method: 'GET',
    });
    return resp.json();
  }

  /** Internal fetch with headers and timeout. */
  private async fetch(url: string, init: RequestInit): Promise<Response> {
    const controller = new AbortController();
    const timeoutId = setTimeout(() => controller.abort(), this.timeout);

    try {
      return await globalThis.fetch(url, {
        ...init,
        headers: { ...this.headers, ...(init.headers as Record<string, string> | undefined) },
        signal: controller.signal,
      });
    } finally {
      clearTimeout(timeoutId);
    }
  }
}
