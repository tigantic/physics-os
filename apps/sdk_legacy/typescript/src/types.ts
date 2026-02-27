/** Physics simulation domain. */
export enum Domain {
  Thermal = 'thermal',
  Euler3d = 'euler3d',
  NsImex = 'ns_imex',
  Fluidelite = 'fluidelite',
}

/** On-chain domain ID mapping. */
export const DOMAIN_MAP: Record<Domain, number> = {
  [Domain.Thermal]: 0,
  [Domain.Euler3d]: 1,
  [Domain.NsImex]: 2,
  [Domain.Fluidelite]: 3,
};

/** Certificate status (matches TPCCertificateRegistry.sol). */
export enum CertificateStatus {
  Valid = 0,
  Revoked = 1,
  Superseded = 2,
}

/** TPCCertificateRegistry ABI (minimal — view functions only). */
export const REGISTRY_ABI = [
  {
    inputs: [{ name: 'contentHash', type: 'bytes32' }],
    name: 'verifyCertificate',
    outputs: [
      { name: 'exists', type: 'bool' },
      { name: 'status', type: 'uint8' },
      { name: 'index', type: 'uint256' },
    ],
    stateMutability: 'view',
    type: 'function',
  },
  {
    inputs: [{ name: '', type: 'bytes32' }],
    name: 'certificates',
    outputs: [
      { name: 'contentHash', type: 'bytes32' },
      { name: 'signerPubkey', type: 'bytes32' },
      { name: 'domain', type: 'uint8' },
      { name: 'registeredAt', type: 'uint256' },
      { name: 'registeredBy', type: 'address' },
      { name: 'proofTxHash', type: 'bytes32' },
      { name: 'status', type: 'uint8' },
      { name: 'certificateId', type: 'bytes16' },
    ],
    stateMutability: 'view',
    type: 'function',
  },
  {
    inputs: [{ name: 'index', type: 'uint256' }],
    name: 'hasPQCCommitment',
    outputs: [{ name: '', type: 'bool' }],
    stateMutability: 'view',
    type: 'function',
  },
] as const;
