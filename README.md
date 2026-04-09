# 🌿 JuteGuard AI — Jute Leaf Disease Detection

A machine-learning-powered web application that helps farmers and agronomists detect diseases in jute leaves by analyzing uploaded photos in real time. The app classifies leaf images into three categories and provides actionable remedy recommendations for each diagnosis.

---

## Table of Contents

1. [Project Overview](#project-overview)
2. [How It Works](#how-it-works)
3. [Features](#features)
4. [Prerequisites & Requirements](#prerequisites--requirements)
5. [Setup & Installation](#setup--installation)
6. [Usage Guide](#usage-guide)
7. [Project Structure](#project-structure)
8. [Verdict & Assessment](#verdict--assessment)
9. [License & Attribution](#license--attribution)

---

## Project Overview

| Property | Value |
|---|---|
| **Application Name** | JuteGuard AI |
| **Purpose** | Automated jute leaf disease detection via image classification |
| **Target Users** | Farmers, agronomists, and agricultural extension workers (primary focus: Bangladesh) |
| **Backend** | Python / Flask |
| **ML Framework** | TensorFlow Lite (MobileNetV2) |
| **Frontend** | HTML + Tailwind CSS + JavaScript |
| **Deployment** | Heroku (Gunicorn) |

### Use Cases

- A farmer photographs a diseased jute leaf with their phone, uploads it to the app, and instantly receives the disease name, a description, and four specific treatment steps.
- An agricultural researcher uses the API endpoint to integrate predictions into a larger crop-monitoring pipeline.
- An extension officer demonstrates disease identification techniques during a field training session.

---

## How It Works

### Architecture

```
Browser (User)
    │
    │  HTTP GET /          → serves index_v2.html
    │  HTTP POST /predict  → accepts image file, returns JSON
    ▼
Flask Application  (web_app_v2.py)
    │
    ├─ Image Preprocessing
    │     • PIL opens the uploaded file
    │     • Resizes to 224 × 224 pixels
    │     • Normalizes pixel values to [0, 1]
    │
    ├─ TFLite Inference
    │     • Loads mobilenetv2_balanced.tflite
    │     • Runs a single forward pass
    │     • Returns a 3-element probability vector
    │
    └─ Response Assembly
          • Picks the highest-probability class
          • Attaches disease description & remedies from DISEASE_INFO dict
          • Returns JSON to the browser
```

### ML Model Details

| Attribute | Value |
|---|---|
| **Base Architecture** | MobileNetV2 |
| **Format** | TensorFlow Lite (.tflite) |
| **File** | `mobilenetv2_balanced.tflite` (~9.5 MB) |
| **Input Tensor** | `[1, 224, 224, 3]` — float32, normalised to [0, 1] |
| **Output Tensor** | `[1, 3]` — softmax probabilities |

MobileNetV2 is a lightweight convolutional neural network designed for mobile and embedded applications. Its inverted residual blocks with linear bottlenecks allow high accuracy at low computational cost, making it well-suited for a web-hosted inference server.

### Disease Classification System

The model outputs a confidence score for each of the three classes:

| Index | Class | Type |
|---|---|---|
| 0 | **Cercospora Leaf Spot** | Fungal |
| 1 | **Golden Mosaic** | Viral |
| 2 | **Healthy Leaf** | — |

The class with the highest softmax probability is selected as the prediction and paired with a curated description and four remedy steps stored in the `DISEASE_INFO` dictionary inside `web_app_v2.py`.

### Data Flow

```
[User uploads image]
        │
        ▼
POST /predict  ←─── multipart/form-data (field name: "file")
        │
        ├── Read bytes from request
        ├── PIL.Image.open → convert to RGB
        ├── Resize to 224×224
        ├── Normalise: pixel / 255.0  →  np.float32
        ├── Add batch dimension: shape [1, 224, 224, 3]
        │
        ├── interpreter.set_tensor(input)
        ├── interpreter.invoke()
        ├── preds = interpreter.get_tensor(output)  →  [p0, p1, p2]
        │
        ├── predicted_class = CLASS_NAMES[argmax(preds)]
        ├── confidence = max(preds) × 100  (percentage)
        │
        └── JSON response:
              {
                "class":       "<disease name>",
                "confidence":  <float>,
                "description": "<disease description>",
                "remedies":    ["step1", "step2", "step3", "step4"]
              }
```

---

## Features

- **Real-time disease prediction** — instant JSON response after image upload, no page reload required.
- **Confidence score** — shows the model's certainty as a percentage.
- **Disease descriptions** — plain-language explanation of each diagnosed condition.
- **Actionable remedies** — four specific treatment or management steps per disease.
- **Drag-and-drop upload** — the frontend supports drag-and-drop as well as click-to-browse.
- **Responsive UI** — Tailwind CSS layout works on desktop and mobile browsers.
- **Glassmorphism design** — modern frosted-glass aesthetic ("JuteGuard AI" brand).
- **Heroku-ready deployment** — `Procfile` and `runtime.txt` are pre-configured.

---

## Prerequisites & Requirements

| Requirement | Version |
|---|---|
| Python | 3.10.13 (specified in `runtime.txt`) |
| pip | latest recommended |
| Virtual environment tool | `venv` (built-in) or `conda` |
| Git | any recent version |
| Heroku CLI | latest (for cloud deployment only) |

### Python Dependencies

```
flask
gunicorn
numpy
pillow
tensorflow-cpu==2.13.0
```

> **Note:** `tensorflow-cpu` is used deliberately to keep the Heroku slug size manageable and because GPU acceleration is not required for single-image inference.

---

## Setup & Installation

### 1. Clone the Repository

```bash
git clone https://github.com/sadman7202/Deployment-of-Jute-leaf-disease-detection.git
cd Deployment-of-Jute-leaf-disease-detection
```

### 2. Create and Activate a Virtual Environment

**Linux / macOS**
```bash
python3 -m venv venv
source venv/bin/activate
```

**Windows (Command Prompt)**
```cmd
python -m venv venv
venv\Scripts\activate
```

**Windows (PowerShell)**
```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
```

### 3. Install Dependencies

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

> Installing `tensorflow-cpu==2.13.0` may take several minutes on first run.

### 4. Verify the Model File is Present

```bash
ls -lh mobilenetv2_balanced.tflite
# Expected: ~ 9.5 MB
```

### 5. Run Locally

```bash
python web_app_v2.py
```

The application starts on **http://localhost:5001**.

To use a different port, edit the last line of `web_app_v2.py`:

```python
app.run(debug=True, port=YOUR_PORT)
```

### 6. Deploy to Heroku

```bash
# Log in to Heroku
heroku login

# Create a new Heroku app
heroku create your-app-name

# Push to Heroku
git push heroku main

# Open the deployed app
heroku open
```

> Heroku will automatically detect the `Procfile` and use `gunicorn web_app_v2:app` as the web process command.

---

## Usage Guide

### Running the Application

```bash
# Activate virtual environment first
source venv/bin/activate          # Linux/macOS
# or
venv\Scripts\activate             # Windows

# Start the server
python web_app_v2.py
```

### Accessing the Web Interface

Open your browser and navigate to:

```
http://localhost:5001
```

You will see the **JuteGuard AI** interface with an upload panel on the left and a results panel on the right.

### Uploading an Image and Getting a Prediction

1. Click the dashed upload area **or** drag and drop a leaf image onto it.
2. Select a `.jpg`, `.jpeg`, `.png`, or other common image format.
3. A preview of the image appears inside the upload card.
4. Click the **"Analyze Leaf"** button.
5. The results panel displays:
   - **Disease name** (or "Healthy Leaf")
   - **Confidence percentage**
   - **Disease description**
   - **Recommended remedies** (numbered list)

### Using the API Directly

You can call the `/predict` endpoint programmatically:

```bash
curl -X POST http://localhost:5001/predict \
     -F "file=@/path/to/your/leaf_image.jpg"
```

**Example JSON response:**

```json
{
  "class": "Cercospora Leaf Spot",
  "confidence": 94.73,
  "description": "Cercospora leaf spot is a fungal disease that causes circular to oval spots with gray centers and dark brown borders on leaves. It can lead to significant defoliation and yield loss if left untreated.",
  "remedies": [
    "Remove and destroy infected plant debris.",
    "Apply fungicides containing copper or chlorothalonil.",
    "Ensure proper spacing between plants to improve air circulation.",
    "Rotate crops to prevent disease buildup in the soil."
  ]
}
```

### Example Workflow

```
1. Farmer notices unusual spots on jute leaves
2. Takes a photo with a smartphone
3. Opens JuteGuard AI in the mobile browser
4. Uploads the photo via drag-and-drop or file picker
5. Receives: "Cercospora Leaf Spot — 94.7% confidence"
6. Reads remedy #2: "Apply fungicides containing copper or chlorothalonil"
7. Takes action in the field
```

---

## Project Structure

```
Deployment-of-Jute-leaf-disease-detection/
│
├── web_app_v2.py                  # Main Flask application
│                                  #   - TFLite model loading & inference
│                                  #   - /predict POST endpoint
│                                  #   - DISEASE_INFO remedies database
│
├── mobilenetv2_balanced.tflite    # Trained TFLite model (~9.5 MB)
│                                  #   - MobileNetV2 backbone
│                                  #   - Input: 224×224×3 float32
│                                  #   - Output: 3-class softmax
│
├── templates/
│   ├── index_v2.html              # ✅ Active frontend template
│   │                              #   - Tailwind CSS + glassmorphism UI
│   │                              #   - Drag-and-drop image upload
│   │                              #   - Async fetch → JSON results display
│   ├── index.html                 # Earlier template (not in use)
│   └── new_index.html             # Draft template (not in use)
│
├── requirements.txt               # Python dependencies
├── Procfile                       # Heroku process definition
│                                  #   web: gunicorn web_app_v2:app
├── runtime.txt                    # Heroku Python version (3.10.13)
└── .gitignore                     # Excludes __pycache__, *.pyc, *.tflite
                                   #   (model file is explicitly re-included)
```

---

## Verdict & Assessment

### Strengths ✅

| Area | Detail |
|---|---|
| **Lightweight model** | MobileNetV2 TFLite keeps inference fast and the deployment footprint small |
| **Clean separation of concerns** | Backend logic, ML inference, and frontend are clearly delineated |
| **Actionable output** | Each prediction includes specific remedy steps, not just a label |
| **Production-ready configuration** | Procfile and runtime.txt remove friction from Heroku deployment |
| **Modern UI** | Glassmorphism design with drag-and-drop provides a good user experience |
| **Balanced dataset** | The `_balanced` suffix in the model filename suggests class-balanced training |

### Areas for Improvement ⚠️

| Area | Issue | Suggested Fix |
|---|---|---|
| **File validation** | No server-side check that the upload is actually an image | Add a MIME-type/extension whitelist before calling `predict_image` |
| **Error handling** | A corrupt or non-image file causes an unhandled exception | Wrap `Image.open` in a try/except and return a user-friendly JSON error |
| **Only 3 classes** | Jute is susceptible to more diseases (e.g., stem rot, root rot) | Expand the training dataset and retrain with additional classes |
| **No rate limiting** | The `/predict` endpoint has no throttling | Add Flask-Limiter to prevent abuse on the public Heroku URL |
| **Unused templates** | `index.html` and `new_index.html` are stale | Remove or move to an `archive/` folder to reduce confusion |
| **No tests** | There are no unit or integration tests | Add `pytest` tests for `predict_image()` and the `/predict` route |
| **Confidence threshold** | Low-confidence predictions are shown without a warning | Display a "low confidence" notice when confidence < 60% |

### Recommendations for Enhancement 🚀

1. **Expand disease classes** — collect and label images for additional jute diseases (stem rot, anthracnose, root rot) and retrain the model.
2. **Add a confidence threshold UI warning** — display an amber banner when confidence is below 60% to encourage the user to retake the photo.
3. **Multilingual support** — add Bengali language support since the primary target users are Bangladeshi farmers.
4. **Offline-capable PWA** — package as a Progressive Web App (PWA) with a service worker so farmers can use it in areas with poor connectivity.
5. **Batch upload** — allow multiple images to be analysed in one submission for field surveys.
6. **Model versioning** — implement an API version prefix (`/api/v1/predict`) and store model metadata (training date, accuracy, classes) in a `model_info.json` file.
7. **Logging & monitoring** — integrate a lightweight logging solution (e.g., Sentry free tier) to track prediction errors in production.
8. **Dataset documentation** — add a `DATA.md` or `TRAINING.md` explaining the dataset source, split ratios, and training accuracy for reproducibility.

### Summary

JuteGuard AI is a **well-structured, functional proof-of-concept** that successfully deploys a real ML model in a production-grade web environment. The core inference pipeline works correctly, the UI is polished, and Heroku deployment is pre-configured.

The main gaps are around robustness (input validation, error handling) and scope (only three disease classes). With the enhancements listed above it could become a genuinely useful tool for jute farmers in Bangladesh.

---

## License & Attribution

| Property | Value |
|---|---|
| **Repository** | [sadman7202/Deployment-of-Jute-leaf-disease-detection](https://github.com/sadman7202/Deployment-of-Jute-leaf-disease-detection) |
| **Author** | sadman7202 |
| **License** | Not specified — all rights reserved by default |

> If you use this project or the model in your own work, please give credit to the original repository.

For questions or issues, please open a GitHub Issue in the repository.
