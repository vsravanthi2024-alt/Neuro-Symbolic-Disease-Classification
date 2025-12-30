# Neuro-symbolic Model for disease classification
This repository contains the implementation of a framework that combines neural and symbolic reasoning components using an attention-based fusion mechanism, enabling effective linkage between unstructured clinical narratives and structured medical knowledge. This design supports interpretable and reliable clinical decision-support modeling.
# Project Title
Neuro-Symbolic Disease Classification Framework

## Description
This repository contains a neuro-symbolic framework for disease classification using emulated electronic medical records (EMRs). The approach integrates structured medical knowledge with contextual language representations to support interpretable clinical modeling.

## Dataset Information
- Type: Electronic Health Records/ clinical records
- Content: Symptom-level textual descriptions and associated labels
- Availability: Kaggle Repository
- Link: https://www.kaggle.com/datasets/davidvictor297/electronic-medical-records-emr

## Dataset Organization
The dataset used in this study is a self-curated synthetic dataset designed to emulate clinical records. The data are organized into separate directories for input samples and corresponding labels to support reproducible experimentation.
## Dataset Structure

The dataset is organized to reflect the data processing pipeline. Raw EMR data are stored separately from processed text and extracted symptom representations to support reproducibility.

dataset/

├── raw/

│   └── emr_data.csv

├── processed/

│   └── cleaned_emr_data.csv

└── symptoms/

    └── extracted_symptoms.csv


## Code Information
- Programming language: Python
- Model types: Neural, symbolic, and hybrid architectures
- Purpose: Clinical text processing and disease classification and prediction

## Usage Instructions
Step-by-Step Execution

Step 1: Clone the repository and open the project directory.

Step 2: Install the required dependencies specified in the Requirements.txt file.

Step 3: Execute Baseline models.py to run the proposed neuro-symbolic model.

Step 4: Run Neuro Neuro-Symbolic Disease classification.ipynb to obtain baseline model results.

Step 5: Execute Comparision among models.ipynb to perform comparative analysis.

Step 6: Run Visualization.ipynb to generate result visualisations.

Step 7: The complete reference code is available in Neuro Symbolic Disease Classification.py (Step by step code verification)

## Requirements
- Python 3.8+
- Required libraries are listed in `requirements.txt`

## Methodology
1. Text preprocessing and normalization
2. Concept mapping using medical ontologies
3. Feature representation using language models
4. Neuro-symbolic fusion and classification
5. Model evaluation

## Evaluation Method
The proposed methodology is evaluated using comparative analysis with baseline models. Performance validation is conducted using standard classification metrics to assess predictive effectiveness and robustness.
| Model                                              | Classification Accuracy (%) |
|----------------------------------------------------|-----------------------------|
| Deep Neural Network (Baseline)                      | ~90.0                       |
| Feed-Forward Neural Network                         | 91.5                        |
| Hybrid Neuro-Symbolic Model (NN + Random Forest)    | 94.5                        |


## Computing Infrastructure
- Operating System: Windows / Linux
- Hardware: Standard CPU-based system
- Frameworks: Python, deep learning libraries

## Citations
[1]Singhal, K., et al. (2023). Large language models encode clinical knowledge. Nature, 620(7972), 172–180. https://doi.org/10.1038/s41586-023-06291-2

[2]Zhang, X., et al. (2024). Development and evaluation of a neuro-symbolic NLP pipeline for automated clinical trial eligibility screening. Journal of the American Medical Informatics Association, 31(8), 1642–1651. https://doi.org/10.1093/jamia/ocae124

[3]Prenosil, G. A., et al. (2024). Neuro-symbolic AI for auditable cognitive information extraction from medical reports. Scientific Reports, 14(1), 26145. https://doi.org/10.1038/s41598-024-77535-y

## License
Confidential – for academic and research purposes only.
