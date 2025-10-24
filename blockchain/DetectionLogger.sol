// blockchain/DetectionLogger.sol
// SPDX-License-Identifier: MIT
pragma solidity ^0.8.0; // Specifies the Solidity compiler version

contract DetectionLogger {
    // Defines the structure to hold information about each detection
    struct Detection {
        string mediaHash;   // SHA-256 hash of the media file
        string label;       // The prediction result ("Real" or "Fake")
        uint256 confidence; // Confidence score (e.g., 88 for 88%)
        uint256 timestamp;  // When the detection was logged (Unix timestamp)
    }

    // A mapping to look up the latest detection record using the media hash
    // This allows quick checks but only stores the *last* record for a given hash
    mapping(string => Detection) public detectionsByHash;

    // An array to store all detection records chronologically
    // This keeps a full history
    Detection[] public allDetections;

    // An event that gets emitted every time a new detection is logged
    // External applications can listen for these events
    event DetectionLogged(string mediaHash, string label, uint256 confidence, uint256 timestamp);

    /**
     * @notice Logs a new detection result to the blockchain.
     * @param _mediaHash The SHA-256 hash of the analyzed media file.
     * @param _label The prediction result ("Real" or "Fake").
     * @param _confidence The confidence score (as an integer, e.g., 88 for 88%).
     */
    function logDetection(string memory _mediaHash, string memory _label, uint256 _confidence) public {
        // Create a new Detection record in memory
        Detection memory newDetection = Detection({
            mediaHash: _mediaHash,
            label: _label,
            confidence: _confidence,
            timestamp: block.timestamp // Uses the current block's timestamp
        });

        // Add the new record to the chronological array
        allDetections.push(newDetection);

        // Update the mapping with this latest record for the given hash
        detectionsByHash[_mediaHash] = newDetection;

        // Emit an event to notify listeners
        emit DetectionLogged(_mediaHash, _label, _confidence, block.timestamp);
    }

    /**
     * @notice Returns the total number of detections logged.
     * @return The count of all detection records.
     */
    function getDetectionCount() public view returns (uint256) {
        return allDetections.length;
    }

    /**
     * @notice Retrieves the latest detection record associated with a specific media hash.
     * @param _mediaHash The SHA-256 hash to query.
     * @return The details of the detection record (hash, label, confidence, timestamp).
     * Returns empty/zero values if the hash is not found in the mapping.
     */
    function getDetectionByHash(string memory _mediaHash) public view returns (string memory, string memory, uint256, uint256) {
         Detection memory det = detectionsByHash[_mediaHash];
         return (det.mediaHash, det.label, det.confidence, det.timestamp);
    }

     /**
     * @notice Retrieves a detection record by its index in the chronological array.
     * @param _index The index of the record to retrieve (0-based).
     * @return The details of the detection record (hash, label, confidence, timestamp).
     * Will revert if the index is out of bounds.
     */
    function getDetectionByIndex(uint256 _index) public view returns (string memory, string memory, uint256, uint256) {
        require(_index < allDetections.length, "Index out of bounds");
        Detection memory det = allDetections[_index];
        return (det.mediaHash, det.label, det.confidence, det.timestamp);
    }
}