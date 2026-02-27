/** Custom error classes for TPC certificate verification. */

export class TPCError extends Error {
  constructor(message: string) {
    super(message);
    this.name = 'TPCError';
  }
}

export class InvalidCertificate extends TPCError {
  constructor(message: string) {
    super(message);
    this.name = 'InvalidCertificate';
  }
}

export class InvalidSignature extends TPCError {
  constructor(message: string) {
    super(message);
    this.name = 'InvalidSignature';
  }
}

export class InvalidHash extends TPCError {
  constructor(message: string) {
    super(message);
    this.name = 'InvalidHash';
  }
}

export class CertificateNotFound extends TPCError {
  constructor(message: string) {
    super(message);
    this.name = 'CertificateNotFound';
  }
}

export class OnChainVerificationFailed extends TPCError {
  constructor(message: string) {
    super(message);
    this.name = 'OnChainVerificationFailed';
  }
}
