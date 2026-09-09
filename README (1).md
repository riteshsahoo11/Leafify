  🌿 **Leafify: AI-Powered Plant Disease Detector**

    Instantly diagnose crop diseases using image analysis and get actionable treatment recommendations.

    Leafify leverages a PyTorch Convolutional Neural Network (CNN) model running on a FastAPI backend to provide real-time diagnosis and an Explainable AI (XAI) feature using Grad-    CAM heatmaps to show exactly where the model found the disease.

  ✨ **Features**

    Real-time Diagnosis: Upload or snap a picture of an affected leaf for instant disease identification.

    Grad-CAM Heatmaps: See visual confirmation (a heatmap) of the precise area on the leaf the AI is focused on for its prediction.

    Confidence Scoring: Get a confidence percentage for the prediction.

    Treatment Advice: Receive practical, step-by-step recommendations for mitigating the identified disease.

    Responsive Interface: A fast and intuitive web application built with React/Vite.

  🚀 **How to Run the Application Locally**

    This project consists of two parts: a Python backend (FastAPI/PyTorch) and a React frontend (Vite). They must be run simultaneously.

  **Prerequisites**

    You need the following installed on your system:

    Python 3.8+

    Node.js / npm (Node Package Manager)

    Git (for cloning)

  **Setup Instructions**

    Ensure you have your trained model file (fast_plant_model.pth) placed inside the backend directory.

  **Step 1:**
  
    Run the Backend (API Server)

    The backend handles the model loading, inference, Grad-CAM generation, and serves the prediction API.

  **Navigate to the backend directory:**

    cd backend


  **Install the required Python packages (including FastAPI, PyTorch, and OpenCV):**

    pip install -r requirements.txt


  **Start the API server:**

    python -m uvicorn main:app --reload


    The server will start, typically running on http://127.0.0.1:8000. Leave this terminal window open.

  **Step 2:**
                        
    Run the Frontend (Web Interface)

    The frontend provides the user interface for capturing images and displaying results.

  **Navigate to the frontend directory:**

    cd "leafify frontend"


  **Install the Node dependencies:**

    npm install


  **Start the development server:**

    npm run dev


  **Go to the Port:**

    The terminal will output the local URL (e.g., http://localhost:5173/). 
    Open this address in your web browser to access the Leafify application.
