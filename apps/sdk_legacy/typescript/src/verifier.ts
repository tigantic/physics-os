/**
 * On-chain TPC certificate verification via viem.
 */

import { Certificate } from './certificate';
import { OnChainVerificationFailed } from './errors';
import { CertificateStatus, REGISTRY_ABI } from './types';

// ═══════════════════════════════════════════════════════════════════════════
// Types
// ═══════════════════════════════════════════════════════════════════════════

export interface VerifierConfig {
  /** Ethereum JSON-RPC URL */
  rpcUrl?: string;
  /** TPCCertificateRegistry contract address */
  registryAddress?: string;
  /** Network name (sepolia, base_sepolia, mainnet, base) */
  network?: string;
}

export interface OnChainResult {
  registered: boolean;
  status?: CertificateStatus;
  index?: bigint;
  contentHash?: string;
  signerPubkey?: string;
  domain?: number;
  registeredAt?: bigint;
  proofTxHash?: string;
  hasPqcCommitment?: boolean;
  error?: string;
}

interface FullVerificationResult {
  valid: boolean;
  hashValid: boolean;
  signatureValid: boolean;
  certificateId: string;
  signerPubkey: string;
  onChain?: OnChainResult;
  error?: string;
}

/** Known registry deployments */
const KNOWN_REGISTRIES: Record<string, { rpc: string; registry: string }> = {
  sepolia: {
    rpc: 'https://rpc.sepolia.org',
    registry: '', // Filled after deployment
  },
  base_sepolia: {
    rpc: 'https://sepolia.base.org',
    registry: '',
  },
  mainnet: {
    rpc: 'https://eth.llamarpc.com',
    registry: '',
  },
  base: {
    rpc: 'https://mainnet.base.org',
    registry: '',
  },
};

// ═══════════════════════════════════════════════════════════════════════════
// Verifier
// ═══════════════════════════════════════════════════════════════════════════

export class TPCVerifier {
  private rpcUrl?: string;
  private registryAddress?: string;
  private client: unknown = null;

  constructor(config: VerifierConfig = {}) {
    if (config.network && config.network in KNOWN_REGISTRIES) {
      const net = KNOWN_REGISTRIES[config.network];
      this.rpcUrl = config.rpcUrl ?? net.rpc;
      this.registryAddress = config.registryAddress ?? net.registry;
    } else {
      this.rpcUrl = config.rpcUrl;
      this.registryAddress = config.registryAddress;
    }
  }

  /** Lazy-initialize viem public client. */
  private async getClient(): Promise<{
    readContract: (args: {
      address: `0x${string}`;
      abi: typeof REGISTRY_ABI;
      functionName: string;
      args: unknown[];
    }) => Promise<unknown>;
  }> {
    if (this.client) return this.client as ReturnType<typeof this.getClient> extends Promise<infer T> ? T : never;

    if (!this.rpcUrl || !this.registryAddress) {
      throw new OnChainVerificationFailed(
        'RPC URL and registry address required for on-chain verification'
      );
    }

    let viem: typeof import('viem');
    try {
      viem = await import('viem');
    } catch {
      throw new OnChainVerificationFailed(
        'viem package required for on-chain verification. ' +
          'Install with: npm install viem'
      );
    }

    const { createPublicClient, http } = viem;
    const client = createPublicClient({
      transport: http(this.rpcUrl),
    });

    this.client = client;
    return client as ReturnType<typeof this.getClient> extends Promise<infer T> ? T : never;
  }

  /** Full verification: local + on-chain. */
  async verify(cert: Certificate): Promise<FullVerificationResult> {
    const local = await cert.verify();

    let onChain: OnChainResult | undefined;
    if (this.rpcUrl && this.registryAddress) {
      try {
        onChain = await this.verifyOnChain(cert);
      } catch (e) {
        onChain = {
          registered: false,
          error: e instanceof Error ? e.message : String(e),
        };
      }
    }

    return {
      valid: local.valid && (onChain ? onChain.registered : local.valid),
      hashValid: local.hashValid,
      signatureValid: local.signatureValid,
      certificateId: local.certificateId,
      signerPubkey: local.signerPubkey,
      onChain,
    };
  }

  /** On-chain verification only. */
  async verifyOnChain(cert: Certificate): Promise<OnChainResult> {
    const client = await this.getClient();
    const registryAddr = this.registryAddress as `0x${string}`;
    const contentHashHex = cert.contentHash;
    const contentHash = ('0x' + contentHashHex) as `0x${string}`;

    // Call verifyCertificate(bytes32)
    let exists: boolean;
    let status: number;
    let index: bigint;

    try {
      const result = (await client.readContract({
        address: registryAddr,
        abi: REGISTRY_ABI,
        functionName: 'verifyCertificate',
        args: [contentHash],
      })) as [boolean, number, bigint];

      [exists, status, index] = result;
    } catch (e) {
      throw new OnChainVerificationFailed(
        `verifyCertificate call failed: ${e instanceof Error ? e.message : String(e)}`
      );
    }

    if (!exists) {
      return { registered: false, contentHash: contentHashHex };
    }

    // Check PQC commitment
    let hasPqcCommitment = false;
    try {
      hasPqcCommitment = (await client.readContract({
        address: registryAddr,
        abi: REGISTRY_ABI,
        functionName: 'hasPQCCommitment',
        args: [index],
      })) as boolean;
    } catch {
      // Non-critical
    }

    return {
      registered: true,
      status: status as CertificateStatus,
      index,
      contentHash: contentHashHex,
      hasPqcCommitment,
    };
  }

  /** Local-only verification. */
  async verifyLocal(cert: Certificate): Promise<FullVerificationResult> {
    const local = await cert.verify();
    return {
      valid: local.valid,
      hashValid: local.hashValid,
      signatureValid: local.signatureValid,
      certificateId: local.certificateId,
      signerPubkey: local.signerPubkey,
    };
  }
}
