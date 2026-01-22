Causality-Aware Hybrid Deep Learning for Multivariate Financial Time Series Forecasting
Overview

This repository contains the implementation and experimental framework for the project "Causality-Aware Hybrid Deep Learning for Multivariate Financial Time Series Forecasting". The project explores how causal reasoning, operationalised through Directed Acyclic Graphs (DAGs), can be integrated with deep learning architectures to improve predictive accuracy, robustness, and interpretability in financial time series forecasting.

Unlike purely correlation-driven models, this work explicitly incorporates domain-informed causal structure into the learning pipeline, addressing key challenges such as spurious correlations, non-stationarity, and poor generalisation under regime shifts.

Key Contributions

Integration of causal priors (DAGs) into deep learning models for multivariate forecasting

A hybrid architecture combining causal structure with sequence-based neural networks

Empirical evaluation on multivariate financial time series data

Improved interpretability and robustness compared to standard deep learning baselines

Project Structure
├── data/                   # Financial time series datasets (or data loaders)
├── notebooks/              # Jupyter notebooks for experiments and analysis
│   └── Causality-Aware Hybrid Deep Learning.ipynb
├── models/                 # Model architectures (LSTM / hybrid models)
├── causal/                 # DAG definitions and causal utilities
├── utils/                  # Preprocessing, evaluation, helper functions
├── results/                # Saved outputs, metrics, and figures
├── README.md               # Project documentation
└── requirements.txt        # Python dependencies

Note: Dataset files may be excluded due to licensing or size constraints. Scripts assume appropriate financial time series data formatted as multivariate sequences.

Methodology Summary

Causal Modelling
A theory-driven Directed Acyclic Graph (DAG) encodes assumed causal relationships between financial variables (e.g. macroeconomic indicators, asset returns, volatility indices).

Hybrid Deep Learning Architecture
Causal structure is incorporated into deep learning via masked connections, causal weighting, or structured attention mechanisms.

Forecasting Task
The model performs multivariate time series forecasting using sequence models (e.g. LSTM-based architectures), trained with temporally consistent splits.

Evaluation
Performance is evaluated against standard deep learning baselines using error-based metrics and qualitative robustness analysis.

Installation

Clone the repository:

git clone https://github.com/your-username/causality-aware-financial-forecasting.git
cd causality-aware-financial-forecasting

Create a virtual environment (recommended):

python -m venv venv
source venv/bin/activate  # Linux / macOS
venv\Scripts\activate     # Windows

Install dependencies:

pip install -r requirements.txt
Usage

Run the main experimental notebook:

jupyter notebook notebooks/Causality-Aware\ Hybrid\ Deep\ Learning.ipynb

Alternatively, model training and evaluation scripts can be executed directly if provided in the models/ or utils/ directories.

Results

The causality-aware hybrid model demonstrates:

Lower forecasting error compared to purely data-driven baselines

Greater stability during volatile market periods

Improved interpretability through alignment with causal assumptions

Detailed results and figures can be found in the results/ directory.

Research Context

This project is suitable for:

PhD-level research in Causal AI, Financial Machine Learning, and Time Series Analysis

Extension into causal discovery, stress testing, and policy-intervention analysis

Submission to conferences and journals in machine learning and quantitative finance

Limitations

The causal DAG is partially theory-driven and may reflect subjective assumptions

Latent confounders in financial markets remain a challenge

Computational overhead increases with causal constraints

Future Work

Automated or hybrid causal discovery methods

Extension to Transformer-based architectures

Counterfactual and intervention-based forecasting

Evaluation across multiple financial markets and regimes

References

Pearl, J. (2009). Causality: Models, Reasoning, and Inference. Cambridge University Press.

Peters, J., Janzing, D., & Schölkopf, B. (2017). Elements of Causal Inference. MIT Press.

Author

Christian Somtoo Obiechina
MSc Artificial Intelligence & Data Science
Research Interests: Causal AI, Financial Machine Learning, Time Series Forecasting

License

This project is released for academic and research purposes. Please cite appropriately if used in scholarly work.
