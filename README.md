# 🪖 Safety PPE Detection System

An automated Computer Vision and Deep Learning system designed to monitor safety compliance in industrial and construction environments by detecting Personal Protective Equipment (PPE) in real-time.

---

## 📌 Project Overview

Ensuring worker safety in hazardous environments is critical. The **Safety PPE Detection System** leverages state-of-the-art Deep Learning object detection models (YOLO) to automatically identify whether individuals are wearing mandatory safety gear, helping prevent workplace injuries and streamline safety audits.

---

## ✨ Key Features

- 🔍 **Real-Time PPE Detection:** Detects personal protective gear in static images and live video streams.
- 🦺 **Multi-Class Recognition:**
  - ⛑️ **Helmets / Hard Hats**
  - 🦺 **Safety Vests**
  - 😷 **Face Masks / Goggles**
  - 🚫 **Non-Compliance Detection** (highlights workers missing safety gear)
- ⚡ **High Precision & Performance:** Optimized for fast inference speeds and low false-positive rates.
- 📓 **Google Colab Support:** Model training and evaluation scripts prepared for GPU acceleration.

---

## 🛠️ Tech Stack & Frameworks

- **Programming Language:** Python 🐍
- **Computer Vision:** OpenCV
- **Deep Learning Frameworks:** PyTorch, Ultralytics YOLO
- **Environment:** Jupyter Notebooks / Google Colab

---

## 📁 Repository Structure

```text
├── Colab_Training.ipynb   # Model training notebook & experiments
├── models/                 # Pre-trained and custom fine-tuned weights
├── data/                   # Sample images and dataset configuration
└── README.md               # Project documentation
