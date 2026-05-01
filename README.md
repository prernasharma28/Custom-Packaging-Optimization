# Custom Packaging Optimization for Cardboard Waste Reduction Using AI

**Author:** Prerna Sharma, Chitkara University

## Overview
This project implements an AI-driven optimization framework for generating custom-fit packaging designs that reduce cardboard waste in e-commerce operations.

## Key Features
- **ML Clustering:** K-Means clustering of products by dimensional similarity
- **Dynamic Programming:** Optimal box size computation
- **Constrained Optimization:** L-BFGS-B algorithm for real-world constraints
- **Sustainability Analysis:** Carbon footprint and material conservation metrics

## Results
- **68.94%** reduction in packaging waste
- **13.26%** cost reduction
- **13.26%** carbon emission reduction
- **51.67%** improvement in volume utilization

## Installation
```bash
pip install -r requirements.txt
## Create python venv
python3 -m venv .venv
## Running the Script
python3 src/research_clean_script_v3.py --data data/ecommerce_product_dimension.csv
