# HR Attrition Intelligence Platform

End-to-end people analytics project combining **exploratory data analysis** with 
a **Gradient Boosting predictive model** — all inside a Streamlit app.

## Features
| Tab | Contents |
|-----|----------|
| 📊 Overview | Attrition by dept/age, income distribution, overtime impact |
| 🔬 Deep Dive | Satisfaction scores, tenure scatter, marital status analysis |
| 🤖 ML Model | ROC curve, confusion matrix, feature importance, CV scores |
| 🔮 Risk Predictor | Real-time risk score + retention recommendations |

## Stack
| Layer       | Technology               |
|-------------|--------------------------|
| App         | Streamlit                |
| ML Model    | scikit-learn GBM         |
| Charts      | Plotly Express           |
| Data        | Pandas + NumPy           |
| Deployment  | Streamlit Community Cloud (free) |

## Run Locally
```bash
pip install -r requirements.txt
streamlit run app.py
```

## Free Deployment (Streamlit Community Cloud)
1. Push to a **public** GitHub repo
2. Go to share.streamlit.io → New app
3. Point to `app.py` → Deploy

## Model Performance
- ROC-AUC ~0.87 on held-out test set
- 5-fold CV AUC: ~0.86
- Features: 18 HR attributes including overtime, income, satisfaction scores

## Skills Demonstrated
- People analytics / HR domain knowledge
- EDA storytelling with Plotly
- ML classification with class imbalance awareness
- Feature importance interpretation (model explainability)
- Production Streamlit app design
