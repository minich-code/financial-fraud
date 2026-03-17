# Submission Checklist

## AI Engineering Challenge - Fraud Detection Pipeline

Use this checklist to ensure your submission is complete before submitting.

---

## ⚠️ Important Reminders

> **Self-Reliant Development:** All work must be completed on your own personal laptop or PC. Sybyl does not provide any development environment or infrastructure.

> **Demo Presentation:** After submission, you will be scheduled for a 30-45 minute demo (in-person or online) to present your solution.

---

## ✅ Pre-Submission Confirmations

Before submitting, confirm the following:

- [ ] **All code runs successfully on my own machine** (laptop/PC)
- [ ] I have tested the complete workflow from data loading to model evaluation
- [ ] My environment can be reproduced using the provided `environment.yml` or `requirements.txt`
- [ ] I am prepared to give a live demo of my solution
- [ ] I can explain my feature engineering decisions and model choices

---

## 📋 Phase 1: Data Science Deliverables

### Jupyter Notebook(s)
- [ ] EDA with clear visualizations and insights
- [ ] Feature engineering with documented rationale
- [ ] Model training and comparison
- [ ] Hyperparameter tuning
- [ ] Final model evaluation with metrics
- [ ] Model interpretation (feature importance, SHAP)
- [ ] Clean, well-commented code
- [ ] Markdown cells explaining your thought process

### Model Artifacts
- [ ] Trained model saved (pickle, joblib, or ONNX)
- [ ] Feature scaler/encoder saved (if applicable)
- [ ] Feature list/schema documented

### Documentation
- [ ] Key EDA findings documented
- [ ] Feature engineering decisions explained
- [ ] Model selection rationale provided
- [ ] Performance metrics reported (AUC, Precision, Recall, F1)

### Test Set Predictions (REQUIRED)
- [ ] `test_predictions.csv` file generated
- [ ] Predictions made for ALL 10,000 transactions in `test_transactions.json`
- [ ] File contains exactly two columns: `transaction_id` and `predicted_fraud`
- [ ] `predicted_fraud` values are binary (0 or 1)
- [ ] Transaction IDs match those in the test file

**Format Example:**
```csv
transaction_id,predicted_fraud
6739205AC695C940,0
0891D338A86BFD86,0
A3F2B1C9D8E7F6A5,1
```

---

## 🔧 Phase 2: MLOps Deliverables

### Source Code (`src/` directory)
- [ ] `validate_data.py` - Data validation logic
- [ ] `feature_engineering.py` - Feature transformation pipeline
- [ ] `train_model.py` - Model training script
- [ ] `evaluate_model.py` - Model evaluation script
- [ ] `export_model.py` - Model export/serialization
- [ ] `utils.py` - Common utilities (if needed)
- [ ] All imports are valid and scripts are runnable

### Docker Files
- [ ] Dockerfile(s) for pipeline stages
- [ ] `requirements.txt` with pinned versions
- [ ] `.dockerignore` file
- [ ] Docker image builds successfully
- [ ] Images are reasonably sized

### Argo Workflow
- [ ] Complete workflow YAML file
- [ ] All pipeline steps defined
- [ ] Proper artifact passing between steps
- [ ] Workflow parameters documented
- [ ] Error handling (retryStrategy) configured
- [ ] Resource limits specified
- [ ] Workflow validates with `argo lint` (if running locally) OR well-documented design

**Note:** If you cannot run Argo locally, ensure your YAML files are well-documented with clear explanations of each step and how they connect.

### Pipeline Documentation
- [ ] Pipeline architecture diagram or description
- [ ] Instructions for running the pipeline
- [ ] Environment variables documented
- [ ] Known limitations listed

---

## 📁 Submission Structure

Your submission should have this structure:

```
submission/
├── README.md                 # Overview of your solution
├── test_predictions.csv      # REQUIRED: Your predictions on test set
├── notebooks/
│   └── fraud_detection.ipynb # Your completed notebook
├── src/
│   ├── __init__.py
│   ├── validate_data.py
│   ├── feature_engineering.py
│   ├── train_model.py
│   ├── evaluate_model.py
│   └── utils.py
├── models/
│   ├── fraud_model.joblib
│   └── feature_scaler.joblib
├── argo/
│   └── fraud-detection-workflow.yaml
├── dockerfiles/
│   ├── Dockerfile
│   └── requirements.txt
└── docs/
    ├── pipeline_architecture.md
    └── feature_documentation.md
```

---

## ✅ Final Checks

Before submitting:

- [ ] All code runs without errors **on my own machine**
- [ ] Notebook cells execute in order
- [ ] No hardcoded absolute paths
- [ ] No sensitive data or credentials
- [ ] README explains how to run everything
- [ ] **`test_predictions.csv` is included and contains 10,000 predictions**
- [ ] Submitted as a single ZIP file

---

## 📧 Submission

Submit your completed work as:
- **Filename:** `[YourName]_AI_Challenge_Submission.zip`
- **Contents:** All directories and files as structured above

---

## 🎤 Demo Preparation

After submission, you will be contacted to schedule your demo presentation.

### Demo Scheduling
- You will receive an email to schedule your demo session
- Demo can be conducted **in-person** or **online** based on your preference
- Duration: **30-45 minutes**

### What to Prepare for the Demo

1. **Live Jupyter Notebook Demonstration**
   - [ ] Be ready to run your notebook live
   - [ ] Prepare to walk through key EDA visualizations
   - [ ] Demonstrate model training and evaluation

2. **Feature Engineering Explanation**
   - [ ] Prepare to explain your rationale for each feature
   - [ ] Be ready to discuss what worked and what didn't

3. **Argo Workflow Walkthrough**
   - [ ] Prepare to present your pipeline architecture
   - [ ] Be ready to explain design decisions and trade-offs

4. **Model Performance Discussion**
   - [ ] Have your metrics ready (AUC, Precision, Recall, F1)
   - [ ] Be prepared to discuss business implications

5. **Q&A Preparation**
   - [ ] Review your code and be ready to explain any part
   - [ ] Think about potential improvements and scalability

---

**Good luck! We look forward to seeing your solution and discussing your approach during the demo.**
