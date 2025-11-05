# blockchain/log_to_blockchain.py
import json
from web3 import Web3
import os
import time
import datetime # Import for timestamp conversion

# --- (Keep existing setup code: CONFIG_PATH, ABI_PATH, config, web3, etc.) ---
CONFIG_PATH = os.path.join(os.path.dirname(__file__), 'contract_config.json')
ABI_PATH = os.path.join(os.path.dirname(__file__), 'DetectionLoggerABI.json')
config = None; web3 = None; contract = None; sender_address = None; private_key = None; is_connected = False
try:
    with open(CONFIG_PATH) as f: config = json.load(f)
    web3 = Web3(Web3.HTTPProvider(config["network_rpc"]))
    is_connected = web3.is_connected()
    if is_connected:
        sender_address = config["wallet_address"]
        pk = config["private_key"]; private_key = pk if pk.startswith('0x') else '0x' + pk
        contract_address = config["contract_address"]
        with open(ABI_PATH) as f: abi = json.load(f)
        contract = web3.eth.contract(address=contract_address, abi=abi)
        print("[INFO] Blockchain connection successful.")
    else: print("[ERROR] Failed to connect to Blockchain RPC:", config.get("network_rpc"))
except FileNotFoundError as e: print(f"[ERROR] Blockchain config/ABI file not found: {e}. Blockchain disabled.") ; config = {}
except Exception as e: print(f"[ERROR] Failed to initialize blockchain connection: {e}. Blockchain disabled.") ; config = {}


def log_detection_to_chain(media_hash, label, confidence, explanation_file):
    """Logs detection results to the configured blockchain."""
    if not is_connected or contract is None: return "[SKIPPED: Blockchain not ready]"
    if explanation_file is None: explanation_file = ""
    try:
        confidence_int = int(float(confidence) * 100)
        checked_sender = Web3.to_checksum_address(sender_address)
        nonce = web3.eth.get_transaction_count(checked_sender)
        txn_dict = contract.functions.logDetection(
            media_hash, label, confidence_int, explanation_file
        ).build_transaction({
            'from': checked_sender, 'nonce': nonce, 'gas': 500000,
            'gasPrice': web3.to_wei('20', 'gwei'), 'chainId': 1337
        })
        signed_txn = web3.eth.account.sign_transaction(txn_dict, private_key=private_key)
        tx_hash = web3.eth.send_raw_transaction(signed_txn.raw_transaction)
        hex_hash = web3.to_hex(tx_hash)
        print(f"[INFO] Logged to blockchain. Tx: {hex_hash}")
        return hex_hash
    except Exception as e: print(f"[ERROR] Blockchain logging failed: {e}"); return f"[ERROR: {e}]"


def query_detection_by_hash(media_hash):
    """Queries the blockchain for a detection record by media hash."""
    if not is_connected or contract is None: return None
    try:
        print(f"[INFO] Querying blockchain for hash: {media_hash[:10]}...")
        # Contract returns 5 values: (hash, label, confidence, timestamp, explanationFile)
        record = contract.functions.getDetectionByHash(media_hash).call()
        timestamp = record[3]
        
        if timestamp > 0: # Check if a valid record was found
            print(f"[INFO] Found existing record on blockchain. Timestamp: {timestamp}")
            explanation_file = record[4] if record[4] != "" else None
            return {
                "hash": record[0],
                "label": record[1],
                "confidence": float(record[2]) / 100.0,
                "timestamp": timestamp,
                "explanation_file": explanation_file
            }
        else:
            print("[INFO] No existing record found for this hash.")
            return None
    except Exception as e:
        print(f"[ERROR] Blockchain query failed: {e}")
        return None

# --- NEW FUNCTION: Get All Logs ---
def get_all_detections():
    """Queries the blockchain for all detection records."""
    if not is_connected or contract is None:
        raise Exception("Blockchain not connected or configured. Cannot query history.")
    
    try:
        print("[INFO] Querying blockchain for all detection logs...")
        log_count = contract.functions.getDetectionCount().call()
        logs = []
        
        # Iterate backwards to get latest logs first
        # Limit to last 50 to avoid potential performance issues
        start_index = max(0, log_count - 50) 
        for i in range(log_count - 1, start_index - 1, -1):
            try:
                # Get record by its chronological index
                record = contract.functions.getDetectionByIndex(i).call()
                # (mediaHash, label, confidence, timestamp, explanationFile)
                log_time = datetime.datetime.fromtimestamp(record[3])
                logs.append({
                    "hash": record[0],
                    "label": record[1],
                    "confidence": float(record[2]) / 100.0,
                    "timestamp": record[3],
                    "explanation_file": record[4] if record[4] != "" else None,
                    "log_time": log_time.strftime('%Y-%m-%d %H:%M:%S') # Add formatted time
                })
            except Exception as e:
                print(f"[WARN] Could not retrieve log at index {i}: {e}")
        
        print(f"[INFO] Found {len(logs)} logs (up to 50 latest).")
        return logs

    except Exception as e:
        print(f"[ERROR] Failed to get all detections: {e}")
        raise
# --- END NEW FUNCTION ---