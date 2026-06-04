# Hybrid Intrusion Detection System using Ensemble Trees and BiLSTM

## Overview

This project presents a **Hybrid Intrusion Detection System (IDS)** that combines **Random Forest (Ensemble Learning)** and **CNN-BiLSTM (Deep Learning)** through a fusion mechanism for enhanced cyber threat detection and classification.

The system is designed to address challenges in modern network security, particularly the detection of **rare and low-frequency attacks** within highly imbalanced network traffic. By leveraging both machine learning and deep learning techniques, the proposed architecture improves detection accuracy while reducing false positives.

This work has been accepted at:

**2026 2nd International Conference on Intelligent Systems for Communication, IoT and Security (ICISCoIS 2026)**

**Paper Title:**
*A Hybrid Intrusion Detection System for Integrating Ensemble Trees with BiLSTM for Enhanced Threat Classification*

---

## Key Features

* Hybrid Machine Learning and Deep Learning Architecture
* Binary Intrusion Detection (Benign vs Malicious)
* Multiclass Attack Classification
* Ensemble Random Forest Model
* CNN-BiLSTM Deep Learning Model
* Meta-Fusion Decision Engine
* Network Flow-Based Analysis
* Scalable and IoT-Oriented Security Framework
* High Accuracy and Low False Positive Rate

---

## System Architecture

```text
Network Traffic Data
        │
        ▼
Data Preprocessing
        │
        ▼
Feature Engineering
        │
 ┌──────┴──────┐
 │             │
 ▼             ▼
Random      CNN-BiLSTM
Forest      Deep Model
 │             │
 └──────┬──────┘
        ▼
 Meta-Fusion Layer
        ▼
 Final Intrusion Decision
```

---

## Project Workflow

### 1. Data Collection

Network traffic data is collected from benchmark intrusion detection datasets containing both benign and malicious traffic samples.

### 2. Data Preprocessing

* Missing value handling
* Feature selection
* Label encoding
* Data normalization
* Dataset balancing

### 3. Feature Engineering

Important network flow characteristics such as:

* Flow Duration
* Packet Statistics
* Flow Bytes/s
* Flow Packets/s
* Inter Arrival Time (IAT)
* Protocol Information
* Port Information

are extracted and transformed for model training.

### 4. Random Forest Model

The Random Forest model performs:

* Binary classification
* Feature importance analysis
* Fast intrusion detection

### 5. CNN-BiLSTM Model

The CNN-BiLSTM architecture:

* Extracts spatial traffic patterns using CNN
* Learns temporal traffic behavior using BiLSTM
* Performs multiclass attack classification

### 6. Fusion Layer

The outputs from both models are combined using a meta-learning strategy to improve:

* Detection accuracy
* Robustness
* Rare attack identification

### 7. Prediction and Alert Generation

The final system predicts:

* Benign Traffic
* DDoS Attacks
* Port Scan Attacks
* Botnet Activity
* Web Attacks
* Other Network Intrusions

---

## Technologies Used

### Programming Languages

* Python

### Machine Learning

* Scikit-Learn
* Random Forest

### Deep Learning

* TensorFlow
* Keras
* CNN
* BiLSTM

### Data Processing

* Pandas
* NumPy

### Visualization

* Matplotlib
* Seaborn

### Deployment

* Flask
* HTML
* CSS

---

## Performance Results

| Model               | Classification | Accuracy |
| ------------------- | -------------- | -------- |
| Random Forest       | Binary         | 99.20%   |
| Random Forest       | Multiclass     | 97.59%   |
| CNN-BiLSTM          | Binary         | 97.67%   |
| CNN-BiLSTM          | Multiclass     | 97.82%   |
| Hybrid Fusion Model | Binary         | 99.49%   |
| Hybrid Fusion Model | Multiclass     | 99.52%   |

### Additional Metrics

* Precision: 99.49%
* Recall: 99.52%
* F1-Score: 98.48%
* AUC: 1.000

---

## Project Structure

```text
intrusion_detection/
│
├── app/
│   ├── app.py
│   └── templates/
│       └── index.html
│
├── data/
│   ├── raw/
│   ├── processed_data.csv
│   └── encoded_data.csv
│
├── models/
│   ├── random_forest_binary.pkl
│   ├── cnn_bilstm_multiclass.h5
│   ├── scaler.pkl
│   └── label_encoder.pkl
│
├── predictions/
│
├── scripts/
│   ├── process_raw.py
│   ├── prepare_data.py
│   ├── train_binary_rf.py
│   ├── train_multiclass_cnn_bilstm.py
│   └── meta_fusion.py
│
└── README.md
```

---

## Applications

* Enterprise Network Security
* IoT Security Monitoring
* Smart City Infrastructure Protection
* Cloud Security Analytics
* Critical Infrastructure Protection
* Cyber Threat Intelligence

---

## Future Enhancements

* Real-Time Packet Capture
* Explainable AI for Intrusion Detection
* Federated Learning-Based IDS
* Edge and Fog Deployment
* Zero-Day Attack Detection
* Adaptive Online Learning

---

## Research Contribution

This project demonstrates that combining ensemble learning with deep learning through a hybrid fusion architecture significantly improves intrusion detection performance, especially for rare and low-frequency attacks that are often missed by standalone models.

---

## Authors

* Charitha K R
* P. Mano Paul
* Soujanya V
* Pragati Sanjay Hubballi
* Deeksha B Poojary

---

## Citation

If you use this work, please cite:

**Paul, P. Mano., Soujanya, V., Poojary, D.B., Charitha, K.R., Hubballi, P.S.**

*"A Hybrid Intrusion Detection System for Integrating Ensemble Trees with BiLSTM for Enhanced Threat Classification."*

**2026 2nd International Conference on Intelligent Systems for Communication, IoT and Security (ICISCoIS 2026).**

DOI: 10.1109/ICISCoIS62701.2026.11447984
