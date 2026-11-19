# Final-Data-Science-Capstone-Project1
Developed an end-to-end machine learning project using Python to classify breast cancer diagnoses, featuring data analysis, preprocessing, model training, evaluation, visualization, and performance comparison.
Data Science Capstone Project
Breast Cancer Diagnosis Classification
1. Project Overview
This capstone demonstrates an end-to-end data science and machine learning workflow. It uses the Wisconsin Diagnostic Breast Cancer dataset bundled with scikit-learn to classify observations as benign or malignant based on numeric measurements computed from digitized cell-nucleus images.
Important: This project is for educational purposes only. It is not a medical device and must not be used for diagnosis, screening, or treatment decisions.

2. Problem Statement
Can supervised machine learning models identify patterns in the dataset that distinguish benign from malignant observations?
3. Objectives
- Load and inspect a real-world dataset.
- Perform exploratory data analysis (EDA).
- Prepare data using an ML pipeline.
- Train and compare multiple classification algorithms.
- Evaluate models with suitable metrics.
- Save the selected model and reproducible outputs.
- Communicate results and limitations.
4. Dataset
- Source: sklearn.datasets.load_breast_cancer
- Rows: 569
- Input features: 30 numeric features
- Target classes: benign and malignant
- External download: Not required; scikit-learn supplies the dataset.
5. Tools and Technologies
- Python
- pandas and NumPy
- Matplotlib and Seaborn
- scikit-learn
- joblib
6. Machine Learning Workflow
1. Load the dataset.
2. Inspect class distribution, missing values, and descriptive statistics.
3. Visualize class distribution and feature correlations.
4. Split the data into training and test sets using stratification.
5. Build preprocessing/model pipelines.
6. Compare Logistic Regression, Random Forest, and Gradient Boosting using 5-fold cross-validation.
7. Select the model with the highest mean cross-validation ROC-AUC.
8. Evaluate the selected model on the held-out test set.
9. Save the model, metrics, reports, plots, and metadata.
7. Evaluation Metrics
- Accuracy: overall fraction of correct predictions.
- Precision (malignant): among observations predicted malignant, the fraction that are malignant.
- Recall (malignant): among actual malignant observations, the fraction detected.
- F1-score (malignant): harmonic mean of malignant precision and recall.
- ROC-AUC (malignant): ability to rank malignant observations above benign observations.
Recall is particularly important to discuss in this educational example because false negatives can be consequential. No single metric makes a model clinically suitable.
8. Project Structure
data_science_capstone/
├── capstone.py
├── README.md
├── requirements.txt
└── artifacts/                 # created after running
    ├── best_model.joblib
    ├── model_comparison.csv
    ├── dataset_summary.txt
    ├── model_metadata.json
    ├── *_report.txt
    ├── class_distribution.png
    ├── correlation_heatmap.png
    ├── best_model_confusion_matrix.png
    └── best_model_roc_curve.png
9. Installation and Execution
Requirements: Python 3.9 or later.
# Optional: create and activate a virtual environment
python -m venv .venv

# Windows
.venv\Scripts\activate

# macOS/Linux
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Run the project
python capstone.py
The script creates an artifacts/ directory containing the analysis outputs and trained model.
10. Results
The script prints a model comparison table and identifies the model selected by cross-validation ROC-AUC. Actual scores are generated when you run the code; they are intentionally not hard-coded because results should come from execution.
11. Limitations
- This is a small, prepackaged dataset and may not represent real-world populations.
- The model is not externally validated.
- A random train/test split is not a substitute for prospective or clinical validation.
- Dataset features are derived measurements; the project does not process raw medical images.
- Feature relationships and model performance may change across populations and data collection settings.
12. Future Improvements
- Add hyperparameter tuning with GridSearchCV or RandomizedSearchCV.
- Add calibration analysis and threshold selection.
- Add SHAP-based model explanations.
- Package inference behind a Streamlit interface or REST API.
- Add unit tests, logging, and experiment tracking.
- Validate on an independent dataset before making any real-world claims.
13. Author
Name: [Your Name]
Internship: [Organization / Program]
Project: Final Data Science Capstone
14. Conclusion
This project demonstrates a reproducible machine learning workflow—from data loading and EDA through model comparison, evaluation, and model persistence. The emphasis is on sound methodology, transparent metrics, and responsible interpretation rather than claiming that a high score alone proves real-world usefulness.
