#Hybrid Functional Link Neural Network for Early Detection of Chronic Diseases
This repository contains the implementation of a hybrid machine learning model that combines Functional Link Artificial Neural Networks (FLANN) with a shallow neural network to predict chronic diseases such as Heart Disease and Chronic Kidney Disease (CKD) using structured tabular data.

The project focuses on evaluating different FLANN expansion functions (Legendre, Trigonometric, Chebyshev, Fourier, and Power series) to determine the best-performing expansion for medical diagnosis. The final hybrid model utilizes the Legendre expansion function for optimal accuracy.

📌 Table of Contents

Overview

Objectives

Datasets Used

Technologies and Methods

FLANN Variants Compared

Hybrid Model Architecture

Model Evaluation Metrics

Project Results

Folder Structure

How to Run

Future Scope

References

📖 Overview

This project aims to build lightweight, accurate, and interpretable models for early-stage disease detection using only tabular clinical data. It compares multiple FLANN variants and implements hybrid architectures to improve performance for heart and kidney disease datasets.

🎯 Objectives

Evaluate and compare the performance of various FLANN variants.

Identify the most suitable expansion function for disease prediction.

Build hybrid models using the best FLANN variant with a single hidden layer.

Analyze results through confusion matrix, ROC curves, loss curves, and accuracy metrics.

Apply the hybrid model to both Heart Disease and CKD datasets.

🧠 Datasets Used

Heart Disease Dataset

Size: ~300,000 samples (synthetically scaled)

Features: age, sex, blood pressure, cholesterol, etc.

Target: 0 (no disease), 1 (disease)

Chronic Kidney Disease Dataset

Size: ~300,000 samples (balanced and preprocessed)

Features: 15 key clinical indicators + 3 custom symptoms

Target: 0 (no disease), 1 (disease)

🧪 Technologies and Methods

Python (NumPy, Pandas, Matplotlib, Seaborn)

Manual implementation of Neural Networks (no TensorFlow/PyTorch)

Functional Expansion: Legendre, Trigonometric, Chebyshev, Fourier

Optimization: Mini-batch Gradient Descent with L2 Regularization

Activation Functions: Leaky ReLU, Sigmoid

Dropout: 35% for hidden layer

ROC Curve, Confusion Matrix, F1-score, Precision, Recall

🧮 FLANN Variants Compared

Trigonometric FLANN (Order 2 and 3)

Legendre FLANN (Order 2 and 3)

Chebyshev FLANN (Order 2 and 3)

Fourier Series FLANN

Power Series FLANN

The Legendre FLANN showed the most stable and generalizable results across both datasets.

🧱 Hybrid Model Architecture

Functional Expansion (Legendre Order 2)
→ Fully Connected Hidden Layer (Leaky ReLU + Dropout)
→ Output Layer (Sigmoid)

Features:

Weight regularization (L2)

Dropout during training

Adaptive learning using gradient descent

📊 Evaluation Metrics

Accuracy

Precision

Recall (Sensitivity)

F1 Score

ROC-AUC

Loss Curves (Train vs Test)

Accuracy Curves (Train vs Test)

📈 Project Results

Heart Disease Hybrid Model

Accuracy: 98.7%

F1 Score: 98.79%

ROC-AUC: > 0.99

CKD Hybrid Model

Accuracy: 98.76%

F1 Score: 98.74%

ROC-AUC: > 0.99

📁 Folder Structure

├── heart_disease_model/
│ └── hybrid_legendre_heart.py
├── kidney_disease_model/
│ └── hybrid_legendre_kidney.py
├── data/
│ ├── heart_disease_dataset.csv
│ └── chronic_kidney_disease_dataset.csv
├── images/
│ ├── flann_architecture.png
│ ├── hybrid_architecture.png
│ ├── loss_curves.png
│ └── roc_curves.png
├── report/
│ └── final_report.pdf
├── presentation/
│ └── final_presentation.pptx
└── README.md

▶️ How to Run

Install dependencies

pip install numpy pandas matplotlib seaborn scikit-learn

Clone repository

git clone https://github.com/your-username/hybrid-flann-disease-prediction.git
cd hybrid-flann-disease-prediction

Run models

python heart_disease_model/hybrid_legendre_heart.py
python kidney_disease_model/hybrid_legendre_kidney.py

📌 Future Scope

Expand to other diseases like Diabetes, Liver Disease, Parkinson’s.

Integrate image-based models (e.g., CNN for X-ray or MRI).

Deploy as a medical diagnostic web/mobile application.

Real-time monitoring and alerts with live data input.

Optimize for edge devices and cloud deployment.

📚 References

[1] P. Mitra, M. Mondal, and S. Saha, "Legendre Neural Network for the Early Detection of Chronic Kidney Disease," IEEE Access, 2023.

[2] P. Mitra, M. Mondal, and S. Saha, "Legendre Neural Network for the Early Detection of Cardiovascular Disease," Springer Nature, 2023.

[3] Xin, Q., et al. “Wavelet Convolution Neural Network for Epilepsy Classification,” IEEE TNSRE, 2022.

[4] Jalodia, N., et al. “DNN Multi-label Classifier for SLA Violation,” IEEE Open Journal, 2021.

[5] Geng, Y., et al. “Prediction Using Multi-Model Fusion Neural Network,” IEEE Access, 2020.
