# Disease Prediction using FLANN & Hybrid Neural Networks
A machine learning project focused on early detection of chronic diseases (Heart Disease and Chronic Kidney Disease) using Functional Link Artificial Neural Networks (FLANN) and Hybrid Neural Network architectures. This work compares various FLANN expansions and introduces hybrid models with improved predictive performance.

📖 Overview
This project implements and compares:

Pure FLANN models (Trigonometric, Legendre, Chebyshev, etc.)

Hybrid FLANN models with a single hidden layer and functional expansions

Datasets used: Heart Disease & Chronic Kidney Disease

Goal: Improve early disease detection through shallow but expressive neural architectures

🎯 Objectives
Evaluate performance of multiple FLANN expansions on medical tabular data.

Develop hybrid models that combine the best FLANN with a neural layer.

Compare models using metrics like Accuracy, F1-Score, and ROC AUC.

Reduce overfitting while maintaining computational efficiency.

🧠 Model Architectures
🔹 Pure FLANN
Functional expansions applied directly to input features without hidden layers.

🔹 Hybrid FLANN (Proposed)
Input → Legendre Expansion → Hidden Layer (Leaky ReLU + Dropout) → Output (Sigmoid)

🖼️ See: /images/hybrid_legendre_architecture.png

🧪 Datasets
Heart Disease Dataset: ~100,000+ rows (balanced & preprocessed)

Chronic Kidney Disease Dataset: Expanded to 300,000 rows from curated data

All datasets use normalized, numeric values only.

⚙️ Preprocessing
Standardization: z = (x - μ) / σ

Train-Test Split: 60–40 or 80–20 (manual shuffling)

Label encoding: 'target' → 0 (no disease), 1 (disease)

🔢 Expansion Techniques Used
Trigonometric Expansion (sin, cos terms)

Legendre Polynomial Expansion

Chebyshev Polynomial Expansion

📊 Evaluation Metrics
Accuracy

Precision

Recall

F1-Score

Confusion Matrix

ROC Curve / AUC

📈 Results (Hybrid Legendre Example)
Accuracy: 98.75%

Precision: 0.9837

Recall: 0.9920

F1 Score: 0.9879

AUC: > 0.99

📊 See performance graphs in: /images/results_heart_model.png

🌐 Future Scope
Expand to other diseases (e.g., Diabetes, Liver, Parkinson’s)

Integrate image-based diseases using CNN+FLANN hybrids

Build cloud/edge deployable medical apps

Use advanced optimizers like Adam, RMSProp

📂 Repository Structure
/
├── data/ → CSV datasets
├── images/ → Architectures, graphs, ROC curves
├── notebooks/ → Jupyter or Python code files
├── README.md → Project summary
├── requirements.txt → Python dependencies
└── main_model.py → Final hybrid model script

📚 References
A. Mahanta and R. K. Agrawal, “Legendre Neural Network for Function Approximation,” Neurocomputing, vol. 74, no. 16, 2011, doi: 10.1016/j.neucom.2011.05.008.

P. Kumar and D. K. Lobiyal, “A Soft Computing Based Hybrid Legendre Neural Network Model for Solving Classification Problems,” Procedia Computer Science, vol. 167, 2020, pp. 1980–1989, doi: 10.1016/j.procs.2020.03.219.

