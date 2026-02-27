/**
 * @fluidelite/verify — TPC Certificate Verification SDK
 *
 * Verify Trustless Physics Certificates locally and against on-chain state.
 *
 * @example
 * ```typescript
 * import { Certificate, TPCVerifier, TPCClient } from '@fluidelite/verify';
 *
 * // Local verification
 * const cert = Certificate.fromHex(hexData);
 * const result = cert.verify();
 * console.log(result.valid);
 *
 * // On-chain verification
 * const verifier = new TPCVerifier({ rpcUrl: 'https://sepolia...', registryAddress: '0x...' });
 * const onChain = await verifier.verifyOnChain(cert);
 *
 * // CA client
 * const client = new TPCClient({ caUrl: 'https://ca.fluidelite.io', apiKey: 'KEY' });
 * const issued = await client.issue({ domain: 'thermal', proof: proofHex });
 * ```
 */

export { Certificate } from './certificate';
export type {
  CertificateHeader,
  Layer,
  Blob,
  SignatureSection,
  VerificationResult,
} from './certificate';

export { TPCVerifier } from './verifier';
export type { OnChainResult, VerifierConfig } from './verifier';

export { TPCClient } from './client';
export type { ClientConfig, IssueRequest, IssueResult } from './client';

export {
  TPCError,
  InvalidCertificate,
  InvalidSignature,
  InvalidHash,
  CertificateNotFound,
  OnChainVerificationFailed,
} from './errors';

export { Domain, DOMAIN_MAP, CertificateStatus } from './types';
