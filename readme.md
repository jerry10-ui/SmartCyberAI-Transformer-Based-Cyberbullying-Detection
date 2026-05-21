# SmartCyberAI

AI-Driven Multi-Label Cyberbullying Detection System using Transformer-Based Deep Learning and Classical Machine Learning Baselines.

---

# 📌 Project Overview

SmartCyberAI is an intelligent cyberbullying and hate speech detection system developed using Natural Language Processing (NLP), Machine Learning, and Transformer-based Deep Learning techniques.

The system is capable of detecting multiple forms of harmful online content from textual input, including:

- Hate Speech
- Toxic Content
- Aggression
- Bullying
- Racism
- Sexism
- Religious Hate
- Offensive Language

The project also includes a comparative experimental study between classical machine learning algorithms and a Transformer-based DistilBERT model.

---

# 🎯 Objectives

- Detect cyberbullying automatically from user text
- Perform multi-label classification on harmful content
- Compare classical ML baselines with transformer-based deep learning
- Build a full-stack web application for real-time prediction
- Reproduce and analyze baseline methodology from a Springer research paper

---

# 🧠 Features

## 🔹 AI Features

- Multi-label cyberbullying classification
- Transformer-based DistilBERT model
- TF-IDF based classical ML baselines
- Stacked Ensemble implementation
- Real-time prediction system
- Confidence score visualization

---

## 🔹 Web Application Features

- User Registration & Login
- User-specific dashboards
- Profile management
- Activity history tracking
- Risk-level detection
- Model information page
- REST API endpoint
- Modern responsive UI

---

# 🏷 Labels Supported

The system supports the following categories:

| Label |
|------|
| hate |
| offensive |
| aggression |
| bullying |
| toxic |
| racism |
| sexism |
| religious_hate |

---

# 📂 Datasets Used

Three Kaggle datasets were used:

1. Cyberbullying Dataset
2. CyberBullying Detection Dataset
3. Hate Speech Detection Curated Dataset

After preprocessing and balancing:

```text
Final Dataset Size: 281,197 samples
```

---

# ⚙️ Dataset Processing Pipeline

The following preprocessing steps were performed:

- Data cleaning
- Label standardization
- Multi-label conversion
- Dataset merging
- Dataset balancing
- Train-validation split

---

# 🧪 Experimental Setup

## 📊 Train / Validation Split

```python
test_size = 0.1
random_state = 42
```

## 📈 Evaluation Metrics

- Macro F1 Score
- Micro F1 Score
- Precision
- Recall

Primary Metric:

```text
Macro F1 Score
```

---

# 🤖 Models Implemented

## 🔹 Classical Machine Learning Baselines

### 1️⃣ Logistic Regression

Pipeline:

```text
TF-IDF → Logistic Regression
```

---

### 2️⃣ Support Vector Machine (SVM)

Pipeline:

```text
TF-IDF → Linear SVM
```

---

### 3️⃣ Random Forest

Pipeline:

```text
TF-IDF → SVD → Random Forest
```

---

### 4️⃣ XGBoost

Pipeline:

```text
TF-IDF → XGBoost
```

---

### 5️⃣ Stacked Ensemble

Base Models:

- Logistic Regression
- SVM
- Random Forest
- XGBoost

Meta Model:

- Logistic Regression

Pipeline:

```text
TF-IDF
   ↓
LR + SVM + RF + XGB
   ↓
Meta Classifier
```

---

# 🧠 Proposed Deep Learning Model

## DistilBERT

Model:

```text
distilbert-base-uncased
```

Configuration:

| Parameter | Value |
|------|------|
| Max Length | 128 |
| Batch Size | 8 |
| Epochs | 2 |
| Optimizer | AdamW |
| Learning Rate | 2e-5 |

Task Type:

```text
Multi-label Classification
```

---

# 📊 Final Results

| Model | Macro F1 |
|------|------|
| Random Forest | 0.2666 |
| Logistic Regression | 0.3767 |
| XGBoost | 0.3838 |
| SVM | 0.4097 |
| Stacked Ensemble | 0.3978 |
| **DistilBERT (Proposed)** | **0.7446** |

---

# 🔍 Key Findings

- Transformer models significantly outperform classical ML approaches.
- TF-IDF based models struggle with contextual understanding.
- SVM performed best among traditional classifiers.
- Random Forest performed poorly on sparse high-dimensional TF-IDF vectors.
- DistilBERT achieved the highest Macro F1 score due to contextual embedding capabilities.

---

# 🖥 System Architecture

```text
User Input
    ↓
Text Preprocessing
    ↓
DistilBERT Prediction Engine
    ↓
Multi-label Classification
    ↓
Risk Analysis
    ↓
Web Dashboard Visualization
```

---

# 🌐 Web Application Modules

## 🔹 Authentication System

- User registration
- User login
- Session management
- Logout functionality

---

## 🔹 Dashboard

- Risk statistics
- Activity history
- Recent analyses
- User-specific logs

---

## 🔹 Profile Page

- User details
- Total analyses performed

---

## 🔹 Model Information Page

Displays:

- Model architecture
- Dataset size
- Evaluation metrics
- Hardware details
- Supported labels

---

# 🛠 Tech Stack

## Backend

- Python
- Flask
- SQLite

---

## Machine Learning / Deep Learning

- PyTorch
- HuggingFace Transformers
- Scikit-learn
- XGBoost

---

## Frontend

- HTML
- CSS
- Bootstrap 5

---

# 📁 Project Structure

```text
SmartCyberAI/
│
├── data/
│   ├── raw/
│   └── processed/
│
├── models/
│
├── src/
│   ├── app/
│   ├── training/
│   └── preprocessing/
│
├── templates/
│
├── static/
│
├── README.md
├── requirements.txt
└── .gitignore
```

---

# 🚀 Installation & Setup

## 1️⃣ Clone Repository

```bash
git clone https://github.com/YOUR_USERNAME/SmartCyberAI.git
```

---

## 2️⃣ Navigate to Project

```bash
cd SmartCyberAI
```

---

## 3️⃣ Install Dependencies

```bash
pip install -r requirements.txt
```

---

## 4️⃣ Run Flask Application

```bash
python app.py
```

---

# 📡 API Endpoint

## Prediction API

```http
POST /api/predict
```

Example Request:

```json
{
    "text": "You are stupid"
}
```

---

# 📚 Research Paper Reference

This project reproduces and extends the baseline methodology from the following Springer research paper:

> AI-based detection of hate speech on social media using stacked ensemble machine learning techniques.

The project further extends the work by introducing:

- Multi-label classification
- Transformer-based deep learning
- Comparative experimental analysis

---

# 🏆 Conclusion

The proposed DistilBERT-based cyberbullying detection framework achieved significantly better performance than classical machine learning approaches.

The experimental study demonstrates that transformer-based contextual embeddings are highly effective for multi-label cyberbullying detection tasks compared to TF-IDF based traditional models.

---

# 👨‍💻 Author

Abhay Garg

---

# 📜 License

This project is developed for academic and educational purposes.