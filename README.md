---

# 🩺 Skin Lesion Classification System (AI Based)

An AI-powered web application that analyzes dermoscopic skin images using a deep learning model to classify skin lesions and provide an indicative clinical category (benign, pre-malignant, or malignant) for early awareness.

> ⚠️ **Disclaimer:** This system is for educational and research purposes only and is not a medical diagnosis tool.

---

## 📌 Project Overview

Skin cancer is one of the most common cancers worldwide. Early detection significantly improves treatment success.
This project uses a **Convolutional Neural Network (CNN)** trained on dermoscopic images to automatically classify skin lesions and display the predicted lesion type along with a confidence score.

The trained model is integrated into a **Streamlit web application** so users can upload an image and instantly receive a prediction.

---

## 🧠 Classes Predicted

| Code  | Lesion Type              | Clinical Category |
| ----- | ------------------------ | ----------------- |
| akiec | Actinic Keratosis        | Pre-Malignant     |
| bcc   | Basal Cell Carcinoma     | Malignant         |
| bkl   | Benign Keratosis         | Benign            |
| df    | Dermatofibroma           | Benign            |
| mel   | Melanoma                 | Malignant         |
| nv    | Melanocytic Nevus (Mole) | Benign            |
| vasc  | Vascular Lesion          | Benign            |

---

## 📂 Dataset Used

**HAM10000 – Human Against Machine with 10000 training images**

Dataset Link:
[https://www.kaggle.com/datasets/kmader/skin-cancer-mnist-ham10000](https://www.kaggle.com/datasets/kmader/skin-cancer-mnist-ham10000)

The dataset contains dermoscopic images of pigmented skin lesions belonging to seven diagnostic categories.

---

## 🧪 Methodology

### 1. Data Preprocessing

* Image resizing
* RGB normalization (0-1 scaling)
* Label encoding
* Train-test split
* Handling class imbalance

### 2. Data Augmentation

* Rotation
* Flipping
* Zooming
* Shifting

### 3. Model Training

A Convolutional Neural Network (CNN) was trained using TensorFlow/Keras including:

* Convolution layers
* ReLU activation
* Max pooling
* Dropout (overfitting prevention)
* Dense layers
* Softmax output

### 4. Evaluation

* Accuracy
* Loss curves
* Confusion matrix

### 5. Deployment

The trained model is integrated into a Streamlit web app where users can upload images and receive predictions instantly.

---

## 🖥️ Application Features

* Upload JPG/PNG dermoscopic image
* Automatic lesion classification
* Confidence percentage
* Clinical category (Benign / Pre-Malignant / Malignant)
* Detailed probability breakdown

---

## 📁 Project Folder Structure

```
skin-lesion-classifier
│── app.py                  # Streamlit web application
│── requirements.txt        # Python dependencies
│── skin_lesion_model.h5   # Trained CNN model
│── README.md
│
└── training/
     └── skin_lesion_training.ipynb   # Model training notebook
```

---

## ⚙️ Installation (Run Locally)

### Step 1 — Clone Repository

```bash
git clone https://github.com/ANGAVI-KRISH/skin-lesion-classifier.git
cd skin-lesion-classifier
```

### Step 2 — Install Dependencies

```bash
pip install -r requirements.txt
```

### Step 3 — Run Application

```bash
streamlit run app.py
```

Open browser:

```
http://localhost:8501
```

---

## 🌐 Deployment

The project is deployed using **Streamlit Cloud** and **Render** for online access.

Users can upload dermoscopic images directly from a browser without installing any software.

---

## 🛠️ Technologies Used

* Python
* TensorFlow / Keras
* OpenCV
* NumPy
* Streamlit
* CNN (Deep Learning)

---

## 📊 Future Improvements

* Mobile application integration
* Clinical dataset expansion
* Explainable AI (Grad-CAM visualization)
* Real-time dermatologist assistance

---

## 📜 License

This project is developed for academic purposes.
Not intended for real medical use.