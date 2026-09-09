# 🌱 Leafify - Plant Disease Detector

[![Live Demo](https://img.shields.io/badge/Live%20Demo-Vercel-black?style=for-the-badge&logo=vercel)](https://leafify-ras.vercel.app/)

Leafify is an AI powerful, full-stack web application designed to help farmers, gardeners, and plant enthusiasts instantly identify plant diseases. By simply capturing or uploading a photo of a leaf, Leafify uses a state-of-the-art Deep Learning model to diagnose the disease and provide tailored treatment recommendations.

---

## ✨ Features

- **Snap & Upload:** Instantly take a photo using your device's camera or upload an existing image.
- **Image Cropper:** Built-in cropping tool ensuring the AI focuses exclusively on the affected area of the leaf.
- **Deep Learning AI:** Powered by a custom-trained **MobileNetV2** (PyTorch) categorizing among **38 different classes** of plant diseases and healthy leaves.
- **Visual Explainability (Grad-CAM):** Generates an interactive heatmap overlying the leaf, highlighting exactly which spots caused the AI to make its diagnosis!
- **Treatment Recommendations:** Provides curated, actionable steps to treat the specific disease diagnosed.
- **Beautiful UI:** A responsive, glassmorphic design crafted with Tailwind CSS and Framer Motion for buttery-smooth animations.

---

## 🛠️ Tech Stack

**Frontend:**
- **React 19 & Vite:** For blazing-fast frontend rendering and build tooling.
- **Tailwind CSS:** Modern, utility-first styling for a beautiful, responsive layout.
- **Framer Motion:** High-performance layout animations and page transitions.
- **React Router:** For seamless single-page application routing.
- **React Image Crop:** Specialized cropping interface for preparing images before AI inference.

**Backend:**
- **FastAPI:** High-performance Python web framework serving the AI inference endpoints.
- **PyTorch & Torchvision:** Core machine learning libraries running the MobileNetV2 architecture.
- **OpenCV (Headless):** Used to compute and overlap the GradCAM heatmaps dynamically onto the original images.
- **Uvicorn:** ASGI web server for running the FastAPI application.

---

## 🚀 Live Deployment

The application is deployed as a decoupled full-stack architecture:

- **Frontend:** Hosted on **Vercel** (`https://leafify-frontend.vercel.app`) – providing edge caching and instant global loading times.
- **Backend:** Hosted on **Render.com** (Web Service) – providing the environment space necessary for the heavy PyTorch dependencies.

---

## ⚙️ Running Locally

If you want to run the project on your own machine:

### 1. Start the Backend API
Navigate to the root directory and install the Python dependencies:
```bash
pip install -r backend/requirements.txt
```
Start the FastAPI server:
```bash
python -m uvicorn backend.main:app --host 0.0.0.0 --port 8000
```
> The API will be available at `http://localhost:8000`

### 2. Start the Frontend
In a new terminal window, navigate into the frontend folder:
```bash
cd "leafify frontend"
```
Install the Node dependencies:
```bash
npm install
```
Run the Vite development server:
```bash
npm run dev
```

*Note: The frontend will automatically detect the backend is running locally at port 8000 (unless overridden by the `.env.production` file).*
