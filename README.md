# Vehicle-Damage-
# 🛡️ AutoGuard AI — Vehicle Damage Assessment System

> **ResNet-50 powered deep learning application for automated vehicle damage classification with severity estimation and repair cost prediction.**



---

## 📌 Overview

AutoGuard AI is an end-to-end machine learning application that classifies vehicle damage from images into **6 categories**, estimates damage severity, and provides repair cost ranges. The system uses a fine-tuned **ResNet-50** backbone and is deployed as a **FastAPI** REST API with a **Streamlit** frontend.

This project was built as a complete ML pipeline — from model training in a Jupyter notebook to a production-ready, full-stack web application.

---

## 🎯 Classes & Damage Categories

| Class | Description | Severity | Est. Repair Cost |
|-------|-------------|----------|-----------------|
| `F_Normal` | Front — No Damage | Minor | ₹0 – ₹5,000 |
| `R_Normal` | Rear — No Damage | Minor | ₹0 – ₹5,000 |
| `F_Breakage` | Front — Glass/Panel Breakage | Moderate | ₹5,000 – ₹15,000 |
| `R_Breakage` | Rear — Glass/Panel Breakage | Moderate | ₹5,000 – ₹15,000 |
| `F_Crushed` | Front — Structural Crush Damage | Critical | ₹15,000 – ₹40,000 |
| `R_Crushed` | Rear — Structural Crush Damage | Critical | ₹15,000 – ₹40,000 |

---

## 🏗️ Architecture

```
autoguard-ai/
│
├── main.py                  # FastAPI backend — inference API
├── app.py                   # Streamlit frontend — UI client
├── model_resnetup.pth       # Trained model weights (ResNet-50)
├── vehicledamage.ipynb      # Training notebook
└── requirements.txt         # Python dependencies
```

### System Flow

```
User Uploads Image
       │
       ▼
 Streamlit Frontend (app.py)
       │  HTTP POST /predict
       ▼
 FastAPI Backend (main.py)
       │
       ├── EXIF correction (PIL)
       ├── ImageNet normalization
       ├── ResNet-50 inference (PyTorch)
       └── Softmax → Top class + all probabilities
       │
       ▼
 JSON Response → Streamlit renders:
   • Predicted class & confidence
   • Severity badge & repair cost
   • Probability dashboard (all 6 classes)
```

---

## 🧠 Model Details

- **Backbone:** ResNet-50 (pretrained on ImageNet)
- **Fine-tuning strategy:** Frozen base — only `layer4` + custom FC head trained
- **Head:** `Dropout(0.615) → Linear(2048 → 6)`
- **Input size:** 224 × 224 (ImageNet standard)
- **Normalization:** mean `[0.485, 0.456, 0.406]`, std `[0.229, 0.224, 0.225]`
- **Output:** 6-class softmax probabilities

---

## 🚀 Getting Started

### Prerequisites

- Python 3.10+
- `model_resnetup.pth` weights file (place in project root)

### Installation

```bash
# Clone the repository
git clone https://github.com/your-username/autoguard-ai.git
cd autoguard-ai

# Install dependencies
pip install -r requirements.txt
```

### Running the Application

**Step 1 — Start the FastAPI backend:**
```bash
uvicorn main:app --host 0.0.0.0 --port 8000 --reload
```

**Step 2 — Start the Streamlit frontend** (in a new terminal):
```bash
streamlit run app.py --server.port 8501
```

**Step 3 — Open your browser:**
```
http://localhost:8501
```

### API Usage (Direct)

Once the backend is running, you can also call the API directly:

```bash
curl -X POST "http://localhost:8000/predict" \
  -H "accept: application/json" \
  -F "file=@your_car_image.jpg"
```

**Sample Response:**
```json
{
  "label": "F_Crushed",
  "confidence": 94.72,
  "all_probs": {
    "F_Breakage": 1.23,
    "F_Crushed": 94.72,
    "F_Normal": 0.41,
    "R_Breakage": 0.88,
    "R_Crushed": 2.31,
    "R_Normal": 0.45
  },
  "severity": "Critical",
  "cost": "₹15,000 – ₹40,000",
  "icon": "🚨"
}
```

**Interactive API docs:** `http://localhost:8000/docs`

---

## 🖥️ UI Features

- **Glassmorphism dark UI** with gradient accents
- **Bento Grid layout** — upload panel + results panel side-by-side
- **Live probability dashboard** — animated bars for all 6 classes
- **Severity-coded results** — color-coded by damage level (green / amber / red)
- **Low-confidence detection** — warns user when model confidence < 50%
- **EXIF orientation correction** — handles smartphone photos correctly

---

## 📦 Tech Stack

| Layer | Technology |
|-------|-----------|
| Deep Learning | PyTorch, TorchVision |
| Model Architecture | ResNet-50 (transfer learning) |
| Backend API | FastAPI + Uvicorn |
| Frontend | Streamlit |
| Image Processing | Pillow (PIL) |
| Data Validation | Pydantic v2 |

---

## ⚠️ Disclaimer

Estimated repair costs are based on average industry standards and are for **informational purposes only**. Actual costs may vary significantly based on vehicle make, model, and local labor rates.

---

## 📄 License

This project is licensed under the MIT License — see the [LICENSE](LICENSE) file for details.

---

*Built with PyTorch · FastAPI · Streamlit*
