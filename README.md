# **Deep Learning for Media Authentication with Blockchain Logging**

## **📌 Project Overview**

This project implements a web application for detecting deepfakes and manipulated media (images and videos) using deep learning models. It features a unique integration with a local blockchain (Ganache) to provide an immutable, tamper-proof log of all analysis results.

The system aims to address the growing challenge of verifying digital content authenticity by providing both AI-powered detection and a trustworthy record of the findings.

## **✨ Key Features**

* **Web Interface:** A simple Flask-based web application allowing users to upload media files.  
* **Dual AI Models:**  
  * Uses a **Convolutional Neural Network (CNN)** based on Xception (fine-tuned on image frames) for analyzing static images.  
  * Employs a **CNN \+ LSTM** architecture (Xception features fed into an LSTM) for analyzing video sequences, capturing temporal inconsistencies.  
* **Blockchain Integration:**  
  * Connects to a local **Ganache** Ethereum blockchain.  
  * Deploys a **Solidity** smart contract (DetectionLogger) to store analysis results.  
  * Logs the **SHA-256 hash** of the media file, the prediction (**Real/Fake**), the model's **confidence score**, and a **timestamp** immutably on the blockchain after each analysis.  
* **Blockchain Query:** Before running AI prediction on an uploaded file, the application calculates its hash and queries the blockchain to check if an analysis record already exists. If found, it displays the stored result, saving computation time.  
* **Media Preview:** Displays a preview of the uploaded image within the results. (Video preview is currently disabled in the UI).

## **🧠 Technology Stack**

* **Backend:** Python, Flask  
* **AI/ML:** TensorFlow, Keras, OpenCV (for video processing), Scikit-learn (for data splitting)  
* **Blockchain:**  
  * **Smart Contract:** Solidity  
  * **Local Blockchain:** Ganache  
  * **Interaction:** Web3.py  
* **Frontend:** HTML, CSS, JavaScript (basic)  
* **Compiler:** py-solc-x (for automated deployment)

## **📊 Dataset**

* The primary dataset used for training is **Celeb-DF**.  
* **Preprocessing:** Videos are processed to extract individual frames (using preprocess.py) or loaded frame-by-frame during training/inference by the data generators/predictors.

## **⚙️ Project Structure**

Major Project Phase 2/  
│  
├── .venv/                      \# Python virtual environment  
├── blockchain/  
│   ├── DetectionLogger.sol     \# Smart contract source  
│   ├── DetectionLoggerABI.json \# Generated contract ABI  
│   ├── contract\_config.json    \# Ganache & contract connection details  
│   └── log\_to\_blockchain.py    \# Script for blockchain interaction  
├── templates/  
│   └── index.html              \# Web app frontend  
├── utils/  
│   └── predictor.py            \# AI model loading & prediction logic  
├── uploads/                    \# Temporary storage for uploaded files  
├── app.py                      \# Main Flask application  
├── deploy\_contract.py          \# Script to deploy the smart contract  
├── cnn\_lstm\_fake\_detector\_fully\_trained.keras \# Trained video model  
├── xception\_regularized\_detector.keras       \# Trained image model  
├── requirements.txt            \# Python dependencies  
├── cnn\_lstm\_trainer.py         \# Training script for video model  
├── cnn\_trainer\_v3\_regularized.py \# Training script for image model  
└── preprocess.py               \# Initial video-to-frame script

## **🚀 Setup and Running the Project**

1. **Prerequisites:**  
   * Python 3.10+ installed.  
   * **Ganache:** Install Ganache GUI from [Truffle Suite](https://trufflesuite.com/ganache/).  
   * **(Optional but Recommended)** Git for cloning.  
2. **Clone the Repository (if applicable):**  
   git clone \<your-repo-url\>  
   cd \<your-repo-name\>

3. **Set Up Python Environment:**  
   \# Create virtual environment  
   python \-m venv .venv  
   \# Activate (Windows PowerShell)  
   .\\.venv\\Scripts\\activate  
   \# Activate (Linux/macOS)  
   \# source .venv/bin/activate

   \# Install dependencies  
   pip install \-r requirements.txt  
   \# OR install manually:  
   \# pip install Flask tensorflow opencv-python web3 py-solc-x numpy matplotlib scikit-learn tqdm Werkzeug

   *Note: py-solc-x might require C++ build tools.*  
4. **Start Ganache:**  
   * Launch the Ganache application. Use "Quickstart" or load a saved workspace.  
   * Copy the **RPC Server URL** (e.g., http://127.0.0.1:7545).  
   * Choose an account, copy its **Address** and **Private Key**.  
5. **Configure Blockchain Connection:**  
   * Open blockchain/contract\_config.json.  
   * Paste the Ganache **RPC URL**, **Wallet Address**, and **Private Key** into the respective fields. Leave contract\_address blank for now.

{  
  "contract\_address": "",  
  "network\_rpc": "\[http://127.0.0.1:7545\](http://127.0.0.1:7545)",  
  "wallet\_address": "YOUR\_GANACHE\_WALLET\_ADDRESS",  
  "private\_key": "YOUR\_GANACHE\_PRIVATE\_KEY"  
}

6. **Deploy the Smart Contract:**  
   * Run the deployment script from your activated terminal:  
     python deploy\_contract.py

   * This will compile DetectionLogger.sol, deploy it to Ganache, save the ABI to DetectionLoggerABI.json, and update contract\_config.json with the deployed contract address.  
7. **Run the Flask Application:**  
   * Make sure Ganache is still running.  
   * Run the main app script:  
     python app.py

   * The application should start and print URLs (e.g., http://127.0.0.1:5000/).  
8. **Access the Web App:**  
   * Open your web browser and navigate to http://127.0.0.1:5000/.  
   * Upload image or video files to test.

## **📈 Current Status & Limitations**

* The end-to-end pipeline (upload \-\> predict \-\> hash \-\> log/query blockchain) is functional.  
* The web interface provides basic interaction and results display.  
* **Model Accuracy:** The current AI models (especially the video model) have relatively low accuracy (\~68% on real videos, \~41% on fakes from Celeb-DF) and require further training, fine-tuning, or architectural improvements.  
* **Explainability:** Grad-CAM integration was attempted but removed due to technical issues; the system currently lacks visual explainability.  
* **Video Preview:** Disabled in the UI due to browser compatibility/codec issues.  
* **Scalability:** Designed for local testing with Ganache; deploying to a public testnet/mainnet would require managing real gas fees and security considerations.

## **🔮 Future Work**

* Improve model accuracy through fine-tuning, hyperparameter optimization, or different architectures.  
* Re-integrate or implement a working explainability feature (e.g., Grad-CAM, SHAP).  
* Enable video previews or thumbnails.  
* Build a history page to view all records stored on the blockchain.  
* Containerize the application using Docker.  
* Clean the dataset by removing identified corrupt video files.