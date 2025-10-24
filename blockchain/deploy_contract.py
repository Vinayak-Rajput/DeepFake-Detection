# Save this as deploy_contract.py
import json
import os
from web3 import Web3
from solcx import compile_standard, install_solc
import time

# --- Configuration ---
GANACHE_URL = "http://127.0.0.1:7545" # Your Ganache RPC URL
SOLIDITY_FILE_PATH = "blockchain/DetectionLogger.sol" # Path to your contract
CONTRACT_NAME = "DetectionLogger" # The name of the contract inside the .sol file
CONFIG_FILE_PATH = "blockchain/contract_config.json" # Where to save the address
ABI_FILE_PATH = "blockchain/DetectionLoggerABI.json" # Where to save the ABI

# --- Load Ganache Account Details ---
try:
    with open(CONFIG_FILE_PATH, 'r') as f:
        config = json.load(f)
    deployer_address = config["wallet_address"]
    pk = config["private_key"]
    private_key = pk if pk.startswith('0x') else '0x' + pk
    print(f"Using deployer account: {deployer_address}")
except FileNotFoundError:
    print(f"ERROR: Config file '{CONFIG_FILE_PATH}' not found.")
    print("Please create it with your Ganache wallet_address and private_key.")
    exit()
except KeyError as e:
    print(f"ERROR: Missing key in '{CONFIG_FILE_PATH}': {e}")
    exit()

# --- 1. Compile Solidity Contract ---
print("Compiling Solidity contract...")
try:
    # Ensure Solidity version is installed (e.g., 0.8.0 used in the contract)
    solc_version = "0.8.0" # Make sure this matches your pragma line
    print(f"Checking/installing solc version {solc_version}...")
    install_solc(solc_version)

    with open(SOLIDITY_FILE_PATH, 'r') as file:
        contract_source_code = file.read()

    compiled_sol = compile_standard(
        {
            "language": "Solidity",
            "sources": {os.path.basename(SOLIDITY_FILE_PATH): {"content": contract_source_code}},
            "settings": {
                "outputSelection": {
                    "*": {
                        "*": ["abi", "metadata", "evm.bytecode", "evm.sourceMap"]
                    }
                }
            },
        },
        solc_version=solc_version,
    )

    # Get ABI and Bytecode
    contract_interface = compiled_sol["contracts"][os.path.basename(SOLIDITY_FILE_PATH)][CONTRACT_NAME]
    abi = contract_interface["abi"]
    bytecode = contract_interface["evm"]["bytecode"]["object"]

    # Save ABI to file
    with open(ABI_FILE_PATH, 'w') as f:
        json.dump(abi, f, indent=2)
    print(f"Contract ABI saved to {ABI_FILE_PATH}")

except FileNotFoundError:
    print(f"ERROR: Solidity file not found at {SOLIDITY_FILE_PATH}")
    exit()
except Exception as e:
    print(f"ERROR: Compilation failed: {e}")
    exit()

# --- 2. Connect to Ganache ---
print(f"Connecting to Ganache at {GANACHE_URL}...")
w3 = Web3(Web3.HTTPProvider(GANACHE_URL))
if not w3.is_connected():
    print("ERROR: Failed to connect to Ganache.")
    exit()
print("Connected successfully.")

# Ensure the deployer account has funds
checksum_deployer = Web3.to_checksum_address(deployer_address)
balance = w3.eth.get_balance(checksum_deployer)
print(f"Deployer balance: {w3.from_wei(balance, 'ether')} ETH")
if balance == 0:
    print("ERROR: Deployer account has zero balance in Ganache.")
    exit()

# --- 3. Deploy Contract ---
print("Deploying contract...")
try:
    Contract = w3.eth.contract(abi=abi, bytecode=bytecode)
    nonce = w3.eth.get_transaction_count(checksum_deployer)

    # Build constructor transaction
    transaction = Contract.constructor().build_transaction(
        {
            "chainId": 1337, # Ganache default chain ID
            "from": checksum_deployer,
            "nonce": nonce,
            "gasPrice": w3.to_wei("20", "gwei"),
            # 'gas': 3000000 # Let web3 estimate gas, or set manually
        }
    )

    # Estimate gas
    estimated_gas = w3.eth.estimate_gas(transaction)
    transaction['gas'] = estimated_gas + 50000 # Add a buffer
    print(f"Estimated Gas: {estimated_gas}")


    # Sign transaction
    signed_txn = w3.eth.account.sign_transaction(transaction, private_key=private_key)

    # Send transaction
    tx_hash = w3.eth.send_raw_transaction(signed_txn.raw_transaction)
    print(f"Deployment transaction sent. Tx Hash: {w3.to_hex(tx_hash)}")

    # --- 4. Get Contract Address ---
    print("Waiting for transaction receipt...")
    # Wait up to 120 seconds for the transaction to be mined
    tx_receipt = w3.eth.wait_for_transaction_receipt(tx_hash, timeout=120)

    if tx_receipt.status == 1:
        contract_address = tx_receipt.contractAddress
        print(f"✅ Contract deployed successfully at address: {contract_address}")

        # Update the config file with the new address
        config["contract_address"] = contract_address
        with open(CONFIG_FILE_PATH, 'w') as f:
            json.dump(config, f, indent=2)
        print(f"Contract address saved to {CONFIG_FILE_PATH}")
    else:
        print("❌ Contract deployment failed. Status:", tx_receipt.status)
        print("Receipt:", tx_receipt)

except Exception as e:
    print(f"ERROR during deployment: {e}")