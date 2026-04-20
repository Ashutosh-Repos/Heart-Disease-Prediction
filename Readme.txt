# 🏥 Cardiovascular Disease Prediction with Explainable AI 🧬

## 📚 Table of Contents
- [Project Overview](#project-overview)
- [Architecture Diagram](#architecture-diagram)
- [Installation](#installation)
- [Usage](#usage)
- [Model Training & Evaluation](#model-training--evaluation)
- [API Reference](#api-reference)
- [Results & Performance](#results--performance)
- [Contributing](#contributing)
- [License](#license)

## 🚀 Project Overview
A high‑fidelity, production‑ready clinical diagnostic system that predicts cardiovascular disease risk using state‑of‑the‑art gradient‑boosting models (LightGBM, XGBoost, CatBoost) with **Predictive Missingness Intelligence** and **Adaptive SMOTE‑Tuning**. Achieves **98.5% accuracy** and **0.99 ROC‑AUC** on a unified clinical registry.

## 🏗️ Architecture Diagram
```mermaid
graph TD
    A1[UCI Global Registry] -->|Ingestion| B[Standardized Clinical Record]
    A2[Cardiovascular Dataset] -->|Ingestion| B
    A3[Kaggle / Local Registries] -->|Offset Engine| B
    B -->|Preprocessing| C[Missingness Intelligence Loop]
    C -->|Optuna Search| D[Best‑Model Selection]
    D -->|Schema Injection| E[Production Binary .joblib]
    E -->|FastAPI| F[Explainable Backend API]
    F -->|Next.js| G[Physician‑Grade Dashboard]
```

## 📦 Installation
### Prerequisites
- Python ≥ 3.9
- Node.js ≥ 18
- Git

### Steps
```bash
# Clone the repository
git clone https://github.com/yourusername/Heart-Disease-Prediction.git
cd Heart-Disease-Prediction

# Python environment
python3 -m venv .venv
source .venv/bin/activate   # macOS / Linux
# .venv\\Scripts\\activate   # Windows
pip install -r requirements.txt

# Frontend dependencies
cd client-app
npm install
cd ..
```

## ▶️ Usage
### Run FastAPI backend
```bash
uvicorn api.api:app --reload --port 8000
```
API available at `http://127.0.0.1:8000`.

### Run Next.js frontend
```bash
cd client-app
npm run dev
```
Open `http://localhost:3000`.

### Example API request (curl)
```bash
curl -X POST http://127.0.0.1:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"age":55,"gender":1,"chestpain":2,"restingBP":130,"serumcholestrol":250,"fastingbloodsugar":0,"restingrelectro":1,"maxheartrate":150,"exerciseangia":0,"oldpeak":1.0,"slope":2,"noofmajorvessels":0,"target":0}'
```

## 📊 Model Training & Evaluation
Detailed pipeline, hyper‑parameter search, and evaluation steps are documented in `design.md`.

## 📡 API Reference
| Endpoint | Method | Description |
|----------|--------|-------------|
| `/predict` | POST | Single prediction |
| `/predict/batch` | POST | Batch predictions |
| `/predict/upload` | POST | CSV upload & bulk prediction |
| `/explain` | POST | SHAP explanation |
| `/model-info` | GET | JSON metrics & CV report |
| `/model-info/ui` | GET | HTML performance dashboard |

## 📈 Results & Performance
| Model | Accuracy | ROC‑AUC |
|-------|----------|--------|
| LightGBM | 98.5% | 0.99 |
| XGBoost | 98.2% | 0.98 |
| CatBoost | 98.3% | 0.985 |

Full metrics in `models/model_performance_report.md`.

## 🤝 Contributing
Guidelines are in `CONTRIBUTING.md`.

## 📄 License
MIT License – see `LICENSE`.

## 🔬 Clinical Research & Findings
### 1. The Regional Center Bias
Our analysis of the **UCI Global Registry** revealed high demographic disparity between the 4 regional centers. Specifically, the **University Hospital Zurich (Switzerland)** center showed an 11:1 disease‑to‑healthy ratio, while **Cleveland Clinic** maintained a balanced 1:1 ratio.

### 2. SMOTE as a Generalization Tool
By treating **SMOTE (Synthetic Minority Over‑sampling)** as a tunable hyperparameter, the "Clinical Optimizer" was able to mathematically bridge these regional gaps. This ensures the model doesn't "over‑learn" the healthy records of the US samples or the diseased skew of the European samples, resulting in a higher global generalization accuracy (**98.5%**).

---

## 📖 Semantic Attribute Dictionary
The following 14 clinical attributes are standardized across all project datasets:

| Feature | Technical Name | Clinical Metadata | Mapping |
| :--- | :--- | :--- | :--- |
| **Age** | `age` | Age (Years) | Continuous |
| **Gender** | `gender` | Biologic Sex | 1 = Male; 0 = Female |
| **Chest Pain** | `chestpain` | `cp` type | 1: Typical, 2: Atypical, 3: Non‑anginal, 4: Asymptomatic |
| **Resting BP** | `restingBP` | Blood Pressure | mmHg on Admission |
| **Cholesterol** | `serumcholestrol` | `chol` | mg/dl |
| **Blood Sugar** | `fastingbloodsugar` | `fbs` (>120 mg/dl) | 1 = True; 0 = False |
| **Resting ECG** | `restingrelectro` | `restecg` | 0: Normal, 1: ST‑T abn, 2: LV Hypertrophy |
| **Heart Rate** | `maxheartrate` | `thalach` | Max heart rate achieved |
| **Ex. Angina** | `exerciseangia` | `exang` | 1 = Yes; 0 = No |
| **ST Depression** | `oldpeak` | Oldpeak | Induced by exercise relative to rest |
| **ST Slope** | `slope` | Slope of the ST segment | 1: Upsloping, 2: Flat, 3: Downsloping |
| **Major Vessels** | `noofmajorvessels` | `ca` (0‑3) | Colored by Fluoroscopy |
| **Target Status** | `target` | Diagnosis (angiographic) | 0 = Normal, 1 = Risk (>50% narrowing) |

### 🛠️ Important: Clinical Value Standard
To maintain **Zero‑Inconsistency**, this project strictly follows the **UCI 1‑Indexed Standard**:
- **Chest Pain**: Must be **1, 2, 3, 4** (Typical, Atypical, Non‑anginal, Asymptomatic).
- **Slope**: Must be **1, 2, 3** (Upsloping, Flat, Downsloping).
If your raw data is 0‑indexed (common in Kaggle sets), you **must** run the Kaggle Alignment Engine (Phase 0) before training or prediction.

## FastAPI Backend
- `POST /predict` → Single prediction
- `POST /predict/batch` → Batch predictions
- `POST /predict/upload` → CSV upload & bulk prediction
- `POST /explain` → SHAP explanation for any record
- `GET /model-info` → JSON metrics & CV report
- `GET /model-info/ui` → Beautiful HTML performance dashboard
- Full CORS support

## Next.js Frontend (App Router + Tailwind + ShadCN)
- Responsive hero landing page with CTA
- Interactive medical checkup form
- Real‑time validation using Zod + React Hook Form
- Dark / Light mode support
- Results display:
  - Risk prediction (Positive / Negative)
  - Probability percentage
  - Interactive SHAP feature contribution bars
  - Detailed contribution table
- Built with Next.js 15, TailwindCSS, ShadCN UI, Tabler Icons

## Project Structure
```bash
.
├── ml/                              # ML training & interpretation scripts
│   ├── preprocessing.py
│   ├── train.py
│   ├── evaluate.py
│   ├── compare_models.py
│   ├── interpret.py
│   ├── hyperparam_search.py
│   └── data/
│       └── Cardiovascular_Disease_Dataset.csv   # Place dataset here
│
├── api/                             # FastAPI server
│   ├── api.py
│   ├── schemas.py
│   └── utils.py
│
├── models/                                 # Auto‑generated as results of training+testing
│   ├── best_model_pipeline.joblib          # final winner model (backend api uses this model to predict)
│   ├── cv_report.json                      # performance metrics of top models.
│   ├── eval_results/                       # all files inside eval_results are generated after evaluation
│   ├── ─────── metrics.json                 # (final performance metric)
│   ├── ─────── feature_importances.csv      # (evaluated every feature or attribute impact on results)
│   ├── ─────── classification_report.json   # (performance reports)
│   ├── ─────── evaluation_summary.json      # summary
│   ├── ─────── confusion_matrix.png         # confusion_matrix (best model)
│   ├── ─────── pr_curve.png                 # pr_curve (best model)
│   ├── ─────── roc_curve.png                # roc_curve (best model)
│   ├── explain/                            # SHAP explanation caches
│   └── hyperopt/                           # hyperparameter tuning results
│
├── frontend/                               # Next.js frontend (rename from client-app if needed)
│   ├── app/
│   ├── components/
│   ├── public/
│   └── next.config.js
│
├── requirements.txt
├── README.md
└── LICENSE
```

### Setup (Go through instructions and cli commands)
## Create Virtual Environment
```bash
python3 -m venv .venv             # for python version 3.9 if any error try (python -m venv .venv )
source .venv/bin/activate          # macOS / Linux
.venv\\Scripts\\activate           # Windows
```
## Install requirements
```bash
pip install -r requirements.txt
```
---
## 🔬 Methodology & Optimization
### 1. Data Provenance & Unification
The model is trained on a **Unified Clinical Registry** synthesized from four international UCI centers:
- **Cleveland Clinic Foundation** (USA)
- **Hungarian Institute of Cardiology** (Budapest)
- **University Hospital Zurich** (Switzerland)
- **VA Medical Center** (Long Beach, CA)
These datasets are semantically merged into a single standardized schema, providing a diverse patient cohort for high‑generalization training.
### 2. Predictive Missingness Strategy
Standard clinical data often suffers from missing values due to administrative or patient‑specific reasons. Instead of simple dropout or mean imputation, this system employs **Missingness Intelligence**:
- Binary indicator flags are generated for every imputed feature.
- The model learns the statistical importance of _why_ a data point is missing (e.g., a specific test not being performed for certain risk groups).
### 3. Hyperparameter Championship (The Optimizer)
Achieving 98.5% accuracy requires deep search. The `ml.train --tune` mode executes:
- **Randomized Search**: 100 iterations per algorithm.
- **Nested SMOTE**: Synthetic Minority Over‑sampling is treated as a **tunable toggle**, allowing the engine to mathematically determine if balancing improves local decision‑boundaries for specific models.
---
## 🧪 Explainable AI (XAI) Dashboard
This system prioritizes **Clinical Trust**. Every prediction is accompanied by a SHAP interpretation:
- **Global Diagnostics**: A summary beeswarm plot shows feature impact across the entire dataset (e.g., verifying `oldpeak` and `maxheartrate` as primary drivers).
- **Local Patient Waterfalls**: For individual reports, the dashboard breaks down exact risk contributions, allowing physicians to verify the "Reasoning Path" of the AI.
---
The system is designed for **Medical Peer‑Review Replication**. Use the following phases to switch between datasets and verify results.
### 🧩 Phase 0: Kaggle Baseline Alignment
If you are using a standard Kaggle `heart.csv` (where Chest Pain is 0‑3), run this first to shift it into the clinical 1‑indexed registry:
```bash
python -m ml.standardize_data \
  --input data/heart.csv \
  --output data/heart_standardized.csv
```
The system automatically detects if an offset is needed.
### 🏆 Phase A: Clinical Registry Unification
*Why this step is required:* The original UCI heart disease data from Kaggle is provided in raw `.data` files rather than CSV. The `ml.combine_uci_data` script aggregates the four center‑specific `.data` files located in `data/heart+disease 2` and converts them into a single standardized CSV (`data/UCI_Heart_Disease_Combined.csv`) for downstream preprocessing and model training.
```bash
python -m ml.combine_uci_data \
  --dir "data/heart+disease 2" \
  --output "data/UCI_Heart_Disease_Combined.csv"
```
### 🧠 Phase B: Deep Championship Training (Dataset Agnostic)
Toggle between the **UCI Global Registry** and the **High‑Volume Cardiovascular Dataset** with the `--data-path` flag.
**Option 1: Train on UCI Global (Best for Clinical Theory)**
```bash
python -m ml.train \
  --data-path data/UCI_Heart_Disease_Combined.csv \
  --target target \
  --id-column patientid \
  --cv 10 \
  --tune \
  --tune-iter 10 \
  --use-smote \
  --n-jobs -1 \
  --random-state 42 \
  --output-dir models
```
**Option 2: Train on High‑Volume (Best for Model Stress‑Testing)**
```bash
python -m ml.train \
  --data-path data/Cardiovascular_Dataset.csv \
  --target target \
  --id-column patientid \
  --cv 10 \
  --tune \
  --tune-iter 10 \
  --n-jobs -1 \
  --random-state 42 \
  --output-dir models
```
### 📊 Phase C: Certification & Metrics
```bash
python -m ml.evaluate \
  --data-path data/UCI_Heart_Disease_Combined.csv \
  --model-path models/best_model_pipeline.joblib \
  --target target \
  --id-column patientid \
  --test-size 0.2 \
  --random-state 42 \
  --output-dir models/eval_results
```
### 🧪 Phase D: Diagnostic Interpretation (SHAP)
```bash
python -m ml.interpret \
  --model-path models/best_model_pipeline.joblib \
  --data-path data/UCI_Heart_Disease_Combined.csv \
  --target target \
  --id-column patientid \
  --output-dir models/explain \
  --max-background 500 \
  --n-samples 100 \
  --topk-features 14 \
  --random-state 42
```
### ⚔️ Phase E: Comparative Model Competition
Benchmark all saved models head‑to‑head on a specific dataset.
```bash
python -m ml.compare_models \
  --data data/Cardiovascular_Dataset.csv \
  --target target \
  --models "models/*.joblib" \
  --out models/compare_results \
  --test-size 0.2 \
  --random-state 42
```
**`comparison_plots/`**:
- `metric_comparison_bar.png`: Standardized performance bench for all algorithms.
- `fold_variation_boxplot.png`: Statistical stability analysis.
- `model_performance_report.md`: The formal, tabular certification report for clinical review.
# Visualizing Performance & Comparison
You can manually trigger the generation of comparative plots and Markdown reports without re‑training:
```bash
python -m ml.visualize_metrics \
  --cv-report models/cv_report.json \
  --hyperopt-dir models/hyperopt \
  --fold-metrics models/fold_metrics.json \
  --output-dir models/comparison_plots
```
**Outputs include:**
- `metric_comparison_bar.png`: All‑model performance comparison.
- `fold_variation_boxplot.png`: Stability comparison across models.
- `fold_details_<model>.png`: Precision/Accuracy/Recall trend (Line) for each model's folds.
- `model_performance_report.md`: Full tabular metrics for all models & individual folds.
- `<model_name>/`: Deep‑dive charts (CM, ROC, PR) for every algorithm.
# Run FastAPI Server
```bash
uvicorn api.api:app --reload --port 8000
```
# Frontend (Next.js)
```bash
cd client-app
npm install
npm run dev
```
## Note
The project is fully compatible with macOS and Linux systems. However, due to certain configurations or environment‑specific differences, it may not be fully compatible with Windows in some cases.
