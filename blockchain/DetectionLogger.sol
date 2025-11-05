// blockchain/DetectionLogger.sol
// SPDX-License-Identifier: MIT
pragma solidity ^0.8.0;

/**
 * @title DetectionLogger
 * @dev Stores deepfake detection results immutably on the blockchain.
 * Includes a field for storing an associated explanation file path.
 */
contract DetectionLogger {
    // Defines the structure to hold information about each detection
    struct Detection {
        string mediaHash;       // SHA-256 hash of the media file
        string label;           // The prediction result ("Real" or "Fake")
        uint256 confidence;     // Confidence score (e.g., 88 for 88%)
        uint256 timestamp;      // When the detection was logged (Unix timestamp)
        string explanationFile; // Filename of the LIME/XAI explanation image
    }

    // A mapping to look up the latest detection record using the media hash
    mapping(string => Detection) public detectionsByHash;

    // An array to store all detection records chronologically
    Detection[] public allDetections;

    // An event that gets emitted every time a new detection is logged
    event DetectionLogged(
        string mediaHash,
        string label,
        uint256 confidence,
        uint256 timestamp,
        string explanationFile
    );

    /**
     * @notice Logs a new detection result to the blockchain.
     * @param _mediaHash The SHA-256 hash of the analyzed media file.
     * @param _label The prediction result ("Real" or "Fake").
     * @param _confidence The confidence score (as an integer, e.g., 88 for 88%).
     * @param _explanationFile The filename of the saved explanation (e.g., "lime_image.jpg").
     */
    function logDetection(
        string memory _mediaHash,
        string memory _label,
        uint256 _confidence,
        string memory _explanationFile
    ) public {
        
        Detection memory newDetection = Detection({
            mediaHash: _mediaHash,
            label: _label,
            confidence: _confidence,
            timestamp: block.timestamp,
            explanationFile: _explanationFile
        });
        
        allDetections.push(newDetection);
        detectionsByHash[_mediaHash] = newDetection;

        emit DetectionLogged(
            _mediaHash,
            _label,
            _confidence,
            block.timestamp,
            _explanationFile
        );
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
     * @return mediaHash The hash of the media file.
     * @return label The predicted label ("Real" or "Fake").
     * @return confidence The confidence score.
     * @return timestamp The time the record was logged.
     * @return explanationFile The filename of the explanation.
     */
    function getDetectionByHash(string memory _mediaHash) 
        public 
        view 
        returns (
            string memory mediaHash,
            string memory label,
            uint256 confidence,
            uint256 timestamp,
            string memory explanationFile
        )
    {
         Detection memory det = detectionsByHash[_mediaHash];
         return (
             det.mediaHash,
             det.label,
             det.confidence,
             det.timestamp,
             det.explanationFile
         );
    }

     /**
     * @notice Retrieves a detection record by its index in the chronological array.
     * @param _index The index of the record to retrieve (0-based).
     * @return mediaHash The hash of the media file.
     * @return label The predicted label ("Real" or "Fake").
     * @return confidence The confidence score.
     * @return timestamp The time the record was logged.
     * @return explanationFile The filename of the explanation.
     */
    function getDetectionByIndex(uint256 _index) 
        public 
        view 
        returns (
            string memory mediaHash,
            string memory label,
            uint256 confidence,
            uint256 timestamp,
            string memory explanationFile
        ) 
    {
        require(_index < allDetections.length, "DetectionLogger: Index out of bounds");
        Detection memory det = allDetections[_index];
        return (
            det.mediaHash,
            det.label,
            det.confidence,
            det.timestamp,
            det.explanationFile
        );
    }
}