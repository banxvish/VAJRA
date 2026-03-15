const hre = require("hardhat");

async function main() {
  const [deployer] = await hre.ethers.getSigners();
  console.log("Deploying KavachaTrustRegistry with account:", deployer.address);

  // We get the contract to deploy
  const KavachaTrustRegistry = await hre.ethers.getContractFactory("KavachaTrustRegistry");
  const registry = await KavachaTrustRegistry.deploy();

  await registry.waitForDeployment();

  console.log("KavachaTrustRegistry deployed to:", await registry.getAddress());
  
  // NOTE: Copy this deployed address! 
  // You will need to plug it into your frontend BlockchainLedger component.
}

main().catch((error) => {
  console.error(error);
  process.exitCode = 1;
});
