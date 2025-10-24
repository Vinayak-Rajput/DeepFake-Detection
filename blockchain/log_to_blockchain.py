# blockchain/log_to_blockchain.py
import json
from web3 import Web3
import os
import time # Import time

# Load configuration once when the module is imported
CONFIG_PATH = os.path.join(os.path.dirname(__file__), 'contract_config.json')
ABI_PATH = os.path.join(os.path.dirname(__file__), 'DetectionLoggerABI.json')

config = None
web3 = None
contract = None
sender_address = None
private_key = None
is_connected = False

try:
    with open(CONFIG_PATH) as f:
        config = json.load(f)

    web3 = Web3(Web3.HTTPProvider(config["network_rpc"]))
    is_connected = web3.is_connected()

    if is_connected:
        sender_address = config["wallet_address"]
        # Ensure private key starts with 0x if it doesn't
        pk = config["private_key"]
        private_key = pk if pk.startswith('0x') else '0x' + pk
        contract_address = config["contract_address"]

        with open(ABI_PATH) as f:
            abi = json.load(f)

        contract = web3.eth.contract(address=contract_address, abi=abi)
        print("[INFO] Blockchain connection successful.")
    else:
        print("[ERROR] Failed to connect to Blockchain RPC:", config.get("network_rpc"))

except FileNotFoundError as e:
    print(f"[ERROR] Blockchain config/ABI file not found: {e}. Blockchain disabled.")
    config = {} # Prevent errors later
except Exception as e:
    print(f"[ERROR] Failed to initialize blockchain connection: {e}. Blockchain disabled.")
    config = {} # Prevent errors later


def log_detection_to_chain(media_hash, label, confidence):
    """Logs detection results to the configured blockchain."""
    if not is_connected or contract is None or not config:
        print("[WARN] Blockchain not connected or configured. Skipping log.")
        return "[SKIPPED: Blockchain not ready]"

    try:
        # Convert confidence (e.g., 0.95) to an integer (e.g., 95)
        confidence_int = int(float(confidence) * 100)

        # Ensure sender address is checksummed
        checked_sender = Web3.to_checksum_address(sender_address)

        nonce = web3.eth.get_transaction_count(checked_sender)

        # Build transaction using the contract function
        txn_dict = contract.functions.logDetection(
            media_hash,
            label,
            confidence_int
        ).build_transaction({
            'from': checked_sender,
            'nonce': nonce,
            'gas': 500000,  # Estimated gas needed; adjust if needed
            'gasPrice': web3.to_wei('20', 'gwei'), # Standard gas price
            'chainId': 1337 # Ganache default Chain ID
        })

        # Sign the transaction
        signed_txn = web3.eth.account.sign_transaction(txn_dict, private_key=private_key)

        # Send the transaction
        tx_hash = web3.eth.send_raw_transaction(signed_txn.raw_transaction)

        hex_hash = web3.to_hex(tx_hash)
        print(f"[INFO] Logged to blockchain. Tx: {hex_hash}")
        return hex_hash

    except Exception as e:
        print(f"[ERROR] Blockchain logging failed: {e}")
        # Consider more specific error handling (e.g., insufficient funds)
        return f"[ERROR: {e}]"


# --- NEW FUNCTION: Query Blockchain ---
def query_detection_by_hash(media_hash):
    """
    Queries the blockchain for a detection record by media hash.

    Returns:
        A dictionary with detection details if found (and timestamp > 0),
        otherwise None.
    """
    if not is_connected or contract is None or not config:
        print("[WARN] Blockchain not connected or configured. Cannot query.")
        return None

    try:
        print(f"[INFO] Querying blockchain for hash: {media_hash[:10]}...")
        # Call the 'getDetectionByHash' function on the smart contract
        result = contract.functions.getDetectionByHash(media_hash).call()

        # The result is a tuple: (hash, label, confidence_int, timestamp)
        stored_hash, label, confidence_int, timestamp = result

        # Check if a valid record was found (timestamp will be > 0 if logged)
        if timestamp > 0:
            print(f"[INFO] Found existing record on blockchain. Timestamp: {timestamp}")
            # Convert confidence back to float (0.0 to 1.0)
            confidence_float = float(confidence_int) / 100.0
            # Convert timestamp to human-readable format (optional)
            log_time = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(timestamp))
            return {
                "hash": stored_hash,
                "label": label,
                "confidence": confidence_float,
                "timestamp": timestamp, # Keep original timestamp
                "log_time_str": log_time # Add readable time
            }
        else:
            print("[INFO] No existing record found for this hash.")
            return None

    except Exception as e:
        print(f"[ERROR] Blockchain query failed: {e}")
        return None
# --- END NEW FUNCTION ---
