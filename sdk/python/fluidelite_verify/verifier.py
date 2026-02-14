"""
On-chain TPC certificate verification via ethers/web3.

Checks whether a certificate's content hash is registered in the
TPCCertificateRegistry contract on Ethereum (mainnet, Sepolia, Base).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from fluidelite_verify.certificate import Certificate
from fluidelite_verify.errors import OnChainVerificationFailed

# TPCCertificateRegistry ABI (minimal — only the view functions we need)
REGISTRY_ABI = [
    {
        "inputs": [{"name": "contentHash", "type": "bytes32"}],
        "name": "verifyCertificate",
        "outputs": [
            {"name": "exists", "type": "bool"},
            {"name": "status", "type": "uint8"},
            {"name": "index", "type": "uint256"},
        ],
        "stateMutability": "view",
        "type": "function",
    },
    {
        "inputs": [{"name": "", "type": "bytes32"}],
        "name": "certificates",
        "outputs": [
            {"name": "contentHash", "type": "bytes32"},
            {"name": "signerPubkey", "type": "bytes32"},
            {"name": "domain", "type": "uint8"},
            {"name": "registeredAt", "type": "uint256"},
            {"name": "registeredBy", "type": "address"},
            {"name": "proofTxHash", "type": "bytes32"},
            {"name": "status", "type": "uint8"},
            {"name": "certificateId", "type": "bytes16"},
        ],
        "stateMutability": "view",
        "type": "function",
    },
    {
        "inputs": [{"name": "", "type": "uint256"}],
        "name": "pqcCommitments",
        "outputs": [{"name": "", "type": "bytes32"}],
        "stateMutability": "view",
        "type": "function",
    },
    {
        "inputs": [{"name": "index", "type": "uint256"}],
        "name": "hasPQCCommitment",
        "outputs": [{"name": "", "type": "bool"}],
        "stateMutability": "view",
        "type": "function",
    },
]

# Known registry deployments
KNOWN_REGISTRIES: dict[str, dict[str, str]] = {
    "sepolia": {
        "rpc": "https://rpc.sepolia.org",
        "registry": "",  # Filled after deployment
    },
    "base_sepolia": {
        "rpc": "https://sepolia.base.org",
        "registry": "",
    },
    "mainnet": {
        "rpc": "https://eth.llamarpc.com",
        "registry": "",
    },
    "base": {
        "rpc": "https://mainnet.base.org",
        "registry": "",
    },
}


@dataclass
class OnChainResult:
    """Result of on-chain certificate verification."""
    registered: bool
    status: Optional[int] = None  # 0=Valid, 1=Revoked, 2=Superseded
    index: Optional[int] = None
    content_hash: Optional[str] = None
    signer_pubkey: Optional[str] = None
    domain: Optional[int] = None
    registered_at: Optional[int] = None
    proof_tx_hash: Optional[str] = None
    has_pqc_commitment: Optional[bool] = None
    error: Optional[str] = None

    @property
    def status_name(self) -> str:
        """Human-readable status."""
        if self.status is None:
            return "unknown"
        return {0: "valid", 1: "revoked", 2: "superseded"}.get(self.status, "unknown")


@dataclass
class VerificationResult:
    """Combined local + on-chain verification result."""
    valid: bool
    hash_valid: bool
    signature_valid: bool
    certificate_id: str
    signer_pubkey: str
    on_chain: Optional[OnChainResult] = None
    error: Optional[str] = None


class TPCVerifier:
    """
    TPC Certificate Verifier with local and on-chain verification.

    Usage:
        verifier = TPCVerifier(rpc_url="https://sepolia.infura.io/v3/KEY",
                               registry_address="0x...")
        result = verifier.verify(cert)
        on_chain = verifier.verify_on_chain(cert)
    """

    def __init__(
        self,
        rpc_url: Optional[str] = None,
        registry_address: Optional[str] = None,
        network: Optional[str] = None,
    ) -> None:
        """
        Initialize the verifier.

        Args:
            rpc_url: Ethereum JSON-RPC URL.
            registry_address: TPCCertificateRegistry contract address.
            network: Network name (sepolia, base_sepolia, mainnet, base).
                     Used to look up default RPC + registry if not provided.
        """
        if network and network in KNOWN_REGISTRIES:
            net = KNOWN_REGISTRIES[network]
            self._rpc_url = rpc_url or net["rpc"]
            self._registry_address = registry_address or net["registry"]
        else:
            self._rpc_url = rpc_url
            self._registry_address = registry_address

        self._w3 = None
        self._contract = None

    def _get_contract(self):
        """Lazy-initialize Web3 and contract instance."""
        if self._contract is not None:
            return self._contract

        if not self._rpc_url or not self._registry_address:
            raise OnChainVerificationFailed(
                "RPC URL and registry address required for on-chain verification"
            )

        try:
            from web3 import Web3
        except ImportError:
            raise OnChainVerificationFailed(
                "web3 package required for on-chain verification. "
                "Install with: pip install 'fluidelite-verify[web3]'"
            )

        self._w3 = Web3(Web3.HTTPProvider(self._rpc_url))
        if not self._w3.is_connected():
            raise OnChainVerificationFailed(
                f"Cannot connect to RPC: {self._rpc_url}"
            )

        self._contract = self._w3.eth.contract(
            address=Web3.to_checksum_address(self._registry_address),
            abi=REGISTRY_ABI,
        )
        return self._contract

    def verify(self, cert: Certificate) -> VerificationResult:
        """
        Full verification: local integrity + on-chain registration.

        Args:
            cert: Certificate to verify.

        Returns:
            VerificationResult with both local and on-chain status.
        """
        # Local verification
        local = cert.verify()

        # On-chain verification (if configured)
        on_chain = None
        if self._rpc_url and self._registry_address:
            try:
                on_chain = self.verify_on_chain(cert)
            except OnChainVerificationFailed as e:
                on_chain = OnChainResult(registered=False, error=str(e))

        return VerificationResult(
            valid=local.valid and (on_chain.registered if on_chain else local.valid),
            hash_valid=local.hash_valid,
            signature_valid=local.signature_valid,
            certificate_id=local.certificate_id,
            signer_pubkey=local.signer_pubkey,
            on_chain=on_chain,
        )

    def verify_on_chain(self, cert: Certificate) -> OnChainResult:
        """
        Verify a certificate's on-chain registration.

        Args:
            cert: Certificate to verify on-chain.

        Returns:
            OnChainResult with registration status and metadata.
        """
        contract = self._get_contract()

        # Compute content hash
        content_hash_hex = cert.content_hash
        content_hash_bytes = bytes.fromhex(content_hash_hex)

        # Call verifyCertificate(bytes32)
        try:
            exists, status, index = contract.functions.verifyCertificate(
                content_hash_bytes
            ).call()
        except Exception as e:
            raise OnChainVerificationFailed(
                f"verifyCertificate call failed: {e}"
            )

        if not exists:
            return OnChainResult(
                registered=False,
                content_hash=content_hash_hex,
            )

        # Fetch full certificate record
        try:
            record = contract.functions.certificates(content_hash_bytes).call()
            stored_hash, signer_pubkey, domain, registered_at, registered_by, proof_tx, rec_status, cert_id = record
        except Exception:
            # View function might not return all fields; fall back
            return OnChainResult(
                registered=True,
                status=status,
                index=index,
                content_hash=content_hash_hex,
            )

        # Check PQC commitment
        has_pqc = False
        try:
            has_pqc = contract.functions.hasPQCCommitment(index).call()
        except Exception:
            pass

        return OnChainResult(
            registered=True,
            status=status,
            index=index,
            content_hash=content_hash_hex,
            signer_pubkey=signer_pubkey.hex() if isinstance(signer_pubkey, bytes) else str(signer_pubkey),
            domain=domain,
            registered_at=registered_at,
            proof_tx_hash=proof_tx.hex() if isinstance(proof_tx, bytes) else str(proof_tx),
            has_pqc_commitment=has_pqc,
        )

    def verify_local(self, cert: Certificate) -> VerificationResult:
        """
        Local-only verification (no on-chain check).

        Args:
            cert: Certificate to verify.

        Returns:
            VerificationResult with local integrity status.
        """
        local = cert.verify()
        return VerificationResult(
            valid=local.valid,
            hash_valid=local.hash_valid,
            signature_valid=local.signature_valid,
            certificate_id=local.certificate_id,
            signer_pubkey=local.signer_pubkey,
        )
