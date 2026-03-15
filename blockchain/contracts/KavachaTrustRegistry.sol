// SPDX-License-Identifier: MIT
pragma solidity ^0.8.20;

contract KavachaTrustRegistry {
    struct VerificationEvent {
        bytes32 proofHash;      // Hash of ZK proof π
        bytes32 txPayloadHash;  // Hash of transaction
        uint256 timestamp;      // Block timestamp
        bool    verified;       // Verification status
    }

    mapping(address => VerificationEvent[]) public auditTrail;
    mapping(address => bytes32) public biometricCommitments; // H(B) only

    event IdentityVerified(address indexed user, bytes32 proofHash, uint256 ts);
    event FraudAttemptLogged(address indexed user, uint256 ts);

    function recordVerification(bytes32 proofHash, bytes32 txHash) external {
        auditTrail[msg.sender].push(
            VerificationEvent(proofHash, txHash, block.timestamp, true)
        );
        emit IdentityVerified(msg.sender, proofHash, block.timestamp);
    }
    
    function recordFraudAttempt() external {
        emit FraudAttemptLogged(msg.sender, block.timestamp);
    }
}
