# AI Engineering Challenge: End-to-End ML Fraud Detection Pipeline

## 🎯 Challenge Overview

Welcome to the AI Intern Technical Challenge! You will build a **complete machine learning fraud detection system** for mobile money transactions, taking it from exploratory data analysis to a deployable automated pipeline.

**Duration:** 1 Week  
**Total Points:** 100

---

## 💻 Environment & Requirements

> **Important:** This is a self-reliant challenge. Sybyl will **NOT** provide any development environment, cloud resources, or infrastructure.

### What You Need to Provide
- **Your own personal laptop or PC** for all development work
- All development, testing, and execution must be done on your own machine
- You are responsible for setting up your local development environment

### Minimum Hardware Requirements

| Component | Minimum | Recommended |
|-----------|---------|-------------|
| **RAM** | 8 GB | 16 GB |
| **Storage** | 10 GB free space | 20 GB free space |
| **CPU** | Modern multi-core processor (Intel i5/AMD Ryzen 5 or equivalent) | Intel i7/AMD Ryzen 7 or better |
| **GPU** | Not required | Optional (CPU-based training is sufficient) |
| **OS** | Windows 10/11, macOS 10.15+, or Linux (Ubuntu 20.04+) | Any of the above |

### Dataset Feasibility
The provided dataset (~50,000 rows) is specifically designed to be manageable on personal laptops:
- **Memory footprint:** ~50-100 MB when loaded into pandas
- **Training time:** Minutes, not hours, on a standard laptop CPU
- **No GPU required:** All models can be trained efficiently on CPU

### Argo Workflows Options
For the MLOps phase, you have two options depending on your machine's capabilities:

**Option A: Local Kubernetes (if your machine supports it)**
- Use Minikube or Kind to run a local Kubernetes cluster
- Install Argo Workflows locally
- Test your workflows end-to-end

**Option B: Design-Only Approach (recommended for most laptops)**
- Create well-documented Argo Workflow YAML files
- Provide clear explanations of your workflow design
- Include a detailed walkthrough of how each step would execute
- This approach is fully acceptable and will be evaluated on design quality

---

## 🎤 Demo Presentation

After the 1-week deadline, you will present your solution to the Sybyl team.

### Demo Format
- **Duration:** 30-45 minutes
- **Mode:** In-person or online (to be scheduled)
- **Scheduling:** You will be contacted to arrange a convenient time

### What to Prepare
1. **Live Jupyter Notebook Demonstration**
   - Walk through your EDA with key visualizations
   - Demonstrate model training and evaluation
   
2. **Feature Engineering Explanation**
   - Explain your rationale for each feature
   - Discuss what worked and what didn't

3. **Argo Workflow Walkthrough**
   - Present your pipeline architecture
   - Explain the design decisions

4. **Model Performance Presentation**
   - Present your metrics (AUC, Precision, Recall, F1)
   - Discuss trade-offs and business implications

5. **Q&A Session**
   - Be prepared to answer questions about your approach
   - Discuss potential improvements and scalability

---

## 📋 Scenario

You are an ML Engineer at a fintech company that operates a mobile money platform similar to M-Pesa. Your task is to develop a fraud detection system that can identify suspicious transactions in real-time. The system needs to be both accurate and production-ready.

The challenge is divided into two phases:
1. **Phase 1: Data Science** - Explore the data, engineer features, and develop a fraud detection model
2. **Phase 2: MLOps** - Package your ML pipeline into Argo Workflows for automated deployment

---

## 📁 Package Contents

```
AI_Challenge_Candidate_Package/
├── README.md                    # This file
├── CHALLENGE_BRIEF.pdf          # Detailed challenge document
├── data/
│   ├── transactions.csv         # Training dataset (~50,000 records, labeled)
│   └── test_transactions.json   # Test dataset (~10,000 records, UNLABELED)
├── environment.yml              # Conda environment specification
├── requirements.txt             # Pip requirements
├── notebooks/
│   └── starter_template.ipynb   # Jupyter notebook template
├── src/
│   └── __init__.py              # Source code directory
├── argo/
│   └── workflow-template.yaml   # Argo workflow template
├── dockerfiles/
│   └── Dockerfile.template      # Docker template
└── submission_checklist.md      # What to submit
```

---

## 🛠️ Setup Instructions

### Prerequisites
- Python 3.9+
- Conda (Miniconda or Anaconda)
- Docker Desktop (for containerization)
- Minikube or Kind (optional, for local Argo Workflows testing)

### Environment Setup

```bash
# Create conda environment
conda env create -f environment.yml

# Activate environment
conda activate fraud-detection

# Verify installation
python -c "import pandas; import sklearn; import torch; print('Setup successful!')"
```

### Start Jupyter

```bash
jupyter lab
# Or
jupyter notebook
```

---

## 📊 Dataset Description

The dataset `data/transactions.csv` contains mobile money transactions with the following columns:

| Column | Description |
|--------|-------------|
| `transaction_id` | Unique identifier for each transaction |
| `timestamp` | Date and time of transaction |
| `sender_id` | Unique identifier for the sender |
| `receiver_id` | Unique identifier for the receiver |
| `amount` | Transaction amount in KES |
| `transaction_type` | Type: send_money, pay_bill, buy_goods, withdraw, deposit |
| `sender_balance_before` | Sender's balance before transaction |
| `sender_balance_after` | Sender's balance after transaction |
| `receiver_balance_before` | Receiver's balance before transaction |
| `receiver_balance_after` | Receiver's balance after transaction |
| `device_id` | Device used for the transaction |
| `location_lat` | Latitude of transaction location |
| `location_lon` | Longitude of transaction location |
| `is_fraud` | Target variable (0 = legitimate, 1 = fraud) |

---

## 🔀 Train/Test Split

Your package includes two separate datasets:

### Training Data: `transactions.csv`
- **Records:** ~50,000 transactions
- **Format:** CSV
- **Labels:** ✅ Includes `is_fraud` column
- **Purpose:** Use this for training and validating your model
- **Date Range:** January 2024 - June 2024

### Test Data: `test_transactions.json`
- **Records:** ~10,000 transactions
- **Format:** JSON (array of objects)
- **Labels:** ❌ Does NOT include `is_fraud` column
- **Purpose:** Generate predictions for final evaluation
- **Date Range:** July 2024 - September 2024

### Important Notes
1. **Train your model** using only `transactions.csv`
2. **Generate predictions** for all transactions in `test_transactions.json`
3. Your predictions will be evaluated against the ground truth (which you don't have access to)
4. The test set has similar characteristics to training data (~4% fraud rate)

### Loading Test Data

```python
import json
import pandas as pd

# Load test data
with open('data/test_transactions.json', 'r') as f:
    test_data = json.load(f)
    
test_df = pd.DataFrame(test_data)
print(f"Test transactions: {len(test_df)}")
```

---

## 📝 Phase 1: Data Science (Days 1-4)

### 1.1 Exploratory Data Analysis (EDA)
- Analyze the distribution of transactions
- Identify patterns in fraudulent vs legitimate transactions
- Visualize temporal, geographical, and behavioral patterns
- Document your key insights

### 1.2 Feature Engineering
- Create meaningful features from raw data
- Consider:
  - Transaction velocity features
  - Amount-based features
  - Time-based features
  - Behavioral patterns
  - Device/location features
- Document your feature engineering rationale

### 1.3 Model Development
- Handle class imbalance appropriately
- Train and compare multiple models
- Perform hyperparameter tuning
- Evaluate using appropriate metrics (Precision, Recall, F1, AUC-ROC)
- Interpret model predictions

### Deliverables for Phase 1
- Completed Jupyter notebook with EDA, feature engineering, and modeling
- Trained model saved as a pickle/joblib file
- Summary of key findings and model performance

---

## 🔧 Phase 2: MLOps with Argo Workflows (Days 5-7)

### 2.1 Pipeline Design
Design an automated ML pipeline with these stages:
1. **Data Validation** - Validate incoming data quality
2. **Feature Engineering** - Apply feature transformations
3. **Model Training** - Train the fraud detection model
4. **Model Evaluation** - Evaluate model performance
5. **Model Export** - Save artifacts for deployment

### 2.2 Containerization
- Create Docker images for each pipeline stage
- Ensure reproducibility with proper dependency management

### 2.3 Argo Workflow Implementation
- Implement the pipeline as an Argo Workflow
- Define proper artifact passing between stages
- Add appropriate error handling
- Include workflow parameters for configuration

### Deliverables for Phase 2
- Dockerfile(s) for pipeline stages
- Argo Workflow YAML file(s)
- Documentation of pipeline architecture

---

## ✅ Submission Requirements

Please submit your work as a compressed archive containing:

1. **Jupyter Notebook(s)** - Your complete analysis and modeling work
2. **Source Code** - Modular Python code in `src/` directory
3. **Model Artifacts** - Trained model file(s)
4. **Docker Files** - All Dockerfiles created
5. **Argo Workflows** - Complete workflow YAML files
6. **Documentation** - README explaining your approach
7. **Test Predictions** - Predictions on the test dataset

### Test Predictions Format

You MUST submit a file named `test_predictions.csv` with your predictions for the test set:

| Column | Description |
|--------|-------------|
| `transaction_id` | The transaction ID from `test_transactions.json` |
| `predicted_fraud` | Your prediction: 0 (legitimate) or 1 (fraud) |

**Example:**
```csv
transaction_id,predicted_fraud
6739205AC695C940,0
0891D338A86BFD86,0
A3F2B1C9D8E7F6A5,1
...
```

See `submission_checklist.md` for detailed requirements.

---

## ⏰ Timeline

| Day | Focus |
|-----|-------|
| Day 1 | Setup, EDA |
| Day 2 | Feature Engineering |
| Day 3 | Model Development |
| Day 4 | Model Tuning & Documentation |
| Day 5 | Docker & Pipeline Design |
| Day 6 | Argo Workflow Implementation |
| Day 7 | Testing, Documentation, Submission |

---

## 📈 Evaluation Criteria

| Criteria | Weight |
|----------|--------|
| Data Understanding & EDA | 20% |
| Feature Engineering | 25% |
| Model Development | 25% |
| Argo Workflow Implementation | 20% |
| Code Quality & Documentation | 10% |

---

## ❓ Questions?

If you have questions about the challenge, please contact the hiring team.

---

**Good luck! We look forward to seeing your solution.**
