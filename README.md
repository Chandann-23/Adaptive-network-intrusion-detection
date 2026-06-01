# Network Intrusion Detection & Threat Analytics Platform

A production-grade, modular Machine Learning project structure and data analysis pipeline for Network Intrusion Detection Systems (NIDS) using the **NSL-KDD dataset**.

---

## 1. Project Directory Structure & Design System

The platform is designed around a modular, clean pipeline pattern. This ensures a clear separation of concerns, keeping raw data, configuration parameters, data transformation logic, modeling scripts, and deployment configurations completely distinct.

```text
network-intrusion-detection/
├── configs/                    # Core configuration files (YAML format)
│   ├── data_config.yaml        # Configurations for preprocessing, drop columns, and columns classification
│   └── model_config.yaml       # Hyperparameters, random state, and training configurations
│
├── data/                       # Structured data directories (gitignored except placeholders)
│   ├── raw/                    # Original, immutable datasets (e.g. KDDTrain+.txt)
│   ├── processed/              # Out-of-distribution processed matrices
│   └── external/               # Supplementary log mappings / intelligence lists
│
├── models/                     # Serialized model bins and pipelines (gitignored)
│
├── notebooks/                  # Jupyter notebooks for active prototyping and profiling
│   └── 1.0_exploratory_data_analysis.ipynb
│
├── reports/                    # Generated documentation, audit sheets, and visual metrics
│   ├── figures/                # Visual charts, ROC curves, confusion matrices
│   ├── nsl_kdd_cybersecurity_audit.md       # Audit of cybersecurity attack profiles
│   ├── nsl_kdd_feature_analysis.md          # Feature-by-feature dictionary analysis
│   └── nsl_kdd_data_quality_assessment.md   # Data quality readiness assessment
│
├── src/                        # Production-ready source code library
│   ├── __init__.py
│   ├── data/                   # Data fetch and load pipelines
│   │   ├── __init__.py
│   │   └── make_dataset.py
│   ├── features/               # Columns mapping and scaling pipeline
│   │   ├── __init__.py
│   │   └── build_features.py
│   ├── models/                 # Training and evaluation routines
│   │   ├── __init__.py
│   │   └── train.py
│   └── utils/                  # Unified logging configurations
│       ├── __init__.py
│       └── logger.py
│
├── tests/                      # Testing suite
│   ├── __init__.py
│   ├── conftest.py             # Mock dataframe fixtures
│   ├── test_data.py            # Data pipeline unit tests
│   └── test_features.py        # Transformation pipeline unit tests
│
├── deployment/                 # Deployment and orchestration configurations
│   ├── Dockerfile              # Multi-stage production container
│   └── docker-compose.yml      # Orchestration for model serving and tracking
│
├── README.md                   # Main setup instructions
├── ROADMAP.md                  # Development and scaling roadmap
├── requirements.txt            # Python dependencies lists
└── pyproject.toml              # Formatting and styling setup (Ruff, pytest)
```

---

## 2. Environment Setup

To run the model training pipeline or run tests locally, set up a virtual environment and install the required dependencies:

```bash
# 1. Clone or navigate to the project directory
cd "Network ML"

# 2. Instantiate python virtual environment
python -m venv .venv

# 3. Activate the environment
# On Windows PowerShell:
.venv\Scripts\Activate.ps1
# On Linux/macOS:
source .venv/bin/activate

# 4. Install dependencies
pip install --upgrade pip
pip install -r requirements.txt
```

---

## 3. Running the Pipeline & Execution

### A. Run Automated Training Pipeline
Running the training script will automatically download the NSL-KDD dataset to `data/raw/` (if not present), fit the preprocessing pipeline, train the XGBoost classifier, evaluate it on a validation split, and save the serialized model outputs to the `models/` directory:

```bash
python src/models/train.py
```

### B. Run Automated Unit Tests
We use `pytest` for pipeline validation. We run `pytest` through the python module interface (`python -m pytest`) to ensure the current workspace directory is correctly added to `sys.path`. Tests run offline using mock fixtures to verify feature dimensions and target mapping logic:

```bash
# Run pytest tests
python -m pytest

# Verify code style and formatting using Ruff
ruff check .
```

### C. Run via Docker Compose
To build and execute the training pipeline inside an isolated container alongside a local **MLflow Tracking Server**:

```bash
# Navigate to the deployment folder
cd deployment

# Spin up pipeline and tracking database
docker-compose up --build
```
Once started, the MLflow dashboard will be accessible at `http://localhost:5000` to inspect training runs and metric trends.
