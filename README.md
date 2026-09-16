# AI-Retail-Predictors
This repository contains the contents of AI-Powered Sales and Demand Prediction System for filling stations stores

## Module Information

- **Module:** AI for Business Analysis with Python (AIBUY3A)
- **Institution:** Vaal University of Technology (VUT)
- **Submission Date:** 02 November 2026
- **Presentations:** 09–13 November 2026

## Team Members

##  Team Members & Responsibilities

|  Name |  Student Number |  Component(s) Responsible |
|:---|:---:|:---|
| **Ntando Mahlaba** | `240605128` | Machine Learning |
| **Simeli Ndodzi** | `240117484` | Time Series Analysis |
| **Lebogang Sebela** | `224488341` | NLP / Speech Processing |
| **Ndabezinhle Nxumalo** | `225083728` | Deep Learning |
| **Simphiwe Nkabinde** | `224973762` | Chatbot / Softbot |
| **Sandiso Shabalala** | `224245309` | Data Preprocessing |
| **Nkosinathi Mahlangu** | `224215809` | Documentation |
| **SC Buthelezi** | `224118692` | Testing & Evaluation |

## Project Overview

This project implements an AI-powered sales and demand prediction system for filling station stores. It uses the Kaggle dataset *"Retail Store Inventory and Demand Forecasting"* (atomicd) to forecast product demand, detect stockout risk, and provide an interactive chatbot interface for store managers.

> **Note on theme adaptation:** The source dataset's original product categories (Groceries, Electronics, Clothing, Furniture, Toys) were relabeled to align with a filling station store context. The full justification for this decision is documented in [`docs/business_objectives.txt`](docs/business_objectives.txt).

## AI Components Addressed

This project addresses all six required AI components, each isolated in its own module for traceability:

| Component | Description | Location |
|---|---|---|
| *Machine Learning* | Regression-based demand forecasting models | [src/ml_models/](src/ml_models/) |
| *Time Series Analysis* | Decomposition and trend/seasonality analysis of sales data | [src/time_series/](src/time_series/) |
| *NLP / Speech Processing* | Natural-language query processing for the chatbot | [src/nlp/](src/nlp/) |
| *Deep Learning* | Neural network-based forecasting model | [src/deep_learning/](src/deep_learning/) |
| *Chatbot / Softbot* | Conversational interface for store managers | [src/chatbot/](src/chatbot/) |
| *Data Preprocessing* | Cleaning, relabeling, and feature engineering underpinning all components | [src/data_preprocessing/](src/data_preprocessing/) |

## Repository Structure

```
ai-retail-predictors/
├── README.md
├── requirements.txt
├── docs/
│   ├── requirements_spec.txt
│   ├── business_objectives.txt
│   └── final_report.docx
├── data/
│   ├── raw/                        
│   └── processed/                  
├── src/
│   ├── data_preprocessing/        
│   ├── ml_models/                  
│   ├── time_series/                
│   ├── nlp/                        
│   ├── deep_learning/              
│   └── chatbot/                    
├── config/
│   └── category_mapping.json       
├── outputs/                        
├── logs/                           
├── notebooks/                      
└── tests/                          
```

---

## Installation

```bash
pip install -r requirements.txt
```

Requirements: Python 3.11.x, Windows. No GPU required.

---

## Running the Pipeline

```bash
python main.py
```

Or run the data cleaner in isolation:

```bash
python src/data_preprocessing/data_cleaner.py "data/raw/sales_data.csv"
```

---

## Documentation
The full written submission, including requirements specification, system design, model development, and results, will be available at [docs/final_report.docx](docs/final_report.docx).

## Academic Note

This repository is part of the AIBUY3A module requirements at VUT and is intended for academic evaluation purposes.

