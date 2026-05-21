# SmartCyberAI - Transformer Based Cyberbullying Detection

Transformer-Based Multi-Label Cyberbullying Detection System using DistilBERT and Natural Language Processing.

---

# Project Overview

SmartCyberAI is an AI-powered cyberbullying detection system developed using Transformer-based Deep Learning and Natural Language Processing (NLP).

The system is capable of identifying multiple forms of harmful online behavior from textual content in real time using a fine-tuned DistilBERT model.

The project focuses on detecting cyberbullying across multiple categories including hate speech, toxic language, aggression, racism, sexism, and offensive content.

---

# Objectives

- Detect cyberbullying and hate speech automatically
- Perform multi-label text classification
- Build a real-time AI-powered prediction system
- Improve contextual understanding using transformer architecture
- Create a user-friendly web application for analysis and monitoring

---

# Key Features

## 🔹 AI Features

- Transformer-based DistilBERT model
- Multi-label cyberbullying classification
- Context-aware text understanding
- Real-time prediction system
- Confidence score analysis
- Risk-level detection

---

## 🔹 Web Application Features

- User Registration & Login
- User-specific dashboards
- Profile management
- Activity history tracking
- Risk analytics
- Model information page
- REST API integration
- Responsive modern UI

---

# Supported Labels

The system supports detection of the following categories:

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

# Dataset Information

The model was trained on a custom merged dataset created from multiple publicly available cyberbullying and hate speech datasets.

### Dataset Processing Included

- Data cleaning
- Label standardization
- Multi-label conversion
- Dataset balancing
- Train-validation splitting

### Final Dataset Size

```text
281,197 samples
```

---

# Deep Learning Model

## DistilBERT

The project uses:

```text
distilbert-base-uncased
```

DistilBERT is a lightweight transformer architecture derived from BERT that provides strong NLP performance with reduced computational cost.

---

# Model Configuration

| Parameter | Value |
|------|------|
| Max Length | 128 |
| Batch Size | 8 |
| Epochs | 2 |
| Optimizer | AdamW |
| Learning Rate | 2e-5 |
| Task Type | Multi-label Classification |

---

# Model Performance

## Final Evaluation Result

| Metric | Score |
|------|------|
| Macro F1 Score | 0.7446 |

The transformer-based approach achieved strong performance in detecting contextual and semantically complex cyberbullying patterns.

---

# Why Transformers?

Traditional NLP techniques often fail to understand contextual meaning in text.

Transformer architectures like DistilBERT provide:

- Context-aware understanding
- Semantic representation learning
- Better handling of informal social media language
- Improved multi-label classification performance

---

# System Architecture

```text
User Input
    ↓
Text Preprocessing
    ↓
DistilBERT Tokenization
    ↓
Transformer Prediction Engine
    ↓
Multi-label Classification
    ↓
Risk Analysis
    ↓
Dashboard Visualization
```

---

# Web Application Modules

## 🔹 Authentication System

- User Registration
- User Login
- Session Management
- Logout Functionality

---

## 🔹 Dashboard

- Risk statistics
- Recent analyses
- User activity tracking
- Prediction history

---

## 🔹 Profile Section

- User details
- Total analyses performed
- Account management

---

## 🔹 Model Information Page

Displays:

- Model architecture
- Supported labels
- Evaluation metrics
- Dataset details
- Training configuration

---

# Tech Stack

## Backend

- Python
- Flask
- SQLite

---

## Deep Learning & NLP

- PyTorch
- HuggingFace Transformers
- DistilBERT
- Scikit-learn

---

## Frontend

- HTML
- CSS
- Bootstrap 5

---

# Project Structure

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
|      ├── templates/
│   ├── training/
│   └── preprocessing/
│
├── README.md
├── requirements.txt
└── .gitignore
```

---

# Installation & Setup

## 1️⃣ Clone Repository

```bash
git clone https://github.com/YOUR_USERNAME/SmartCyberAI.git
```

---

## 2️⃣ Navigate to Project Directory

```bash
cd SmartCyberAI
```

---

## 3️⃣ Install Dependencies

```bash
pip install -r requirements.txt
```

---

## 4️⃣ Run Application

```bash
python app.py
```

---

# Conclusion

SmartCyberAI demonstrates the effectiveness of Transformer-based Deep Learning for cyberbullying detection.

The DistilBERT model successfully captures contextual and semantic information from text, enabling accurate multi-label classification of harmful online behavior.

The project combines AI, NLP, and web technologies to create a practical and intelligent cyberbullying detection platform.

---

# Author

Developed by Abhay Garg 
(Pursuing B.Tech CSE – AI/ML)
