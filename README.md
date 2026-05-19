# 💳 PaySense AI

**Smart Payment Failure Prediction for Nigerian Fintech**

![Python](https://img.shields.io/badge/Python-3.11-blue)
![Streamlit](https://img.shields.io/badge/Streamlit-Latest-red)
![XGBoost](https://img.shields.io/badge/XGBoost-Latest-green)

## Overview
PaySense AI predicts payment failures before they happen using XGBoost trained on Nigerian payment behavioral patterns across all major banks and payment methods.

## Pages
| Page | Description |
|---|---|
| ⚡ Payment Scanner | Score individual payments instantly |
| 📦 Batch Analyzer | Upload CSV, flag high-risk payments |
| 📊 Model Insights | Feature importance & failure patterns |
| 📈 Risk Report | Failure rates by bank, method & category |

## Setup
```bash
pip install -r requirements.txt
python model/train_model.py
streamlit run app.py
```

## Author
**Okparaji Wisdom** · [@Santandave961](https://github.com/Santandave961)