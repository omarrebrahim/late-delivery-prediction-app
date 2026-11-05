# 📦 Late-Delivery Predictor for an E-commerce Marketplace

Late deliveries cause bad reviews and lost customers. This project predicts, **at purchase time**, which orders on the Olist marketplace will arrive after their estimated delivery date, so operations can warn customers and chase risky orders early.

**🔗 Live app:**https://late-delivery-prediction-app-aqeqgjexoxgtxyt5ua6p8v.streamlit.app/

![App ss1](<Screenshot 2026-11-05 210301.png>)
![ss2](<Screenshot 2026-11-05 210359.png>)
![ss3](<Screenshot 2026-11-05 210427.png>)
![ss4](<Screenshot 2026-11-05 210508.png>)
## Results

| Model | ROC-AUC | PR-AUC | Recall (late) | Precision (late) |
|---|---|---|---|---|
| Baseline (majority class) | 0.500 | 0.068 | 0.000 | 0.000 |
| Logistic Regression | 0.710 | 0.158 | 0.628 | 0.122 |
| Gradient Boosting (tuned) | 0.793 | 0.255 | 0.662 | 0.174 |
| **Random Forest (tuned, threshold 0.39) – final** | **0.790** | **0.258** | **0.612** | **0.195** |

*Test-set scores on 19,294 held-out orders (6.8% late). The final model catches about 61% of late orders; roughly 1 in 5 alerts is a true late order (2.9x the base rate). Random Forest and Gradient Boosting were close in cross-validation (PR-AUC 0.252 vs 0.248).*

## What's inside

- `notebook.ipynb` – full workflow: framing, data understanding, EDA, leakage-free pipeline, 3 models + baseline, tuning, evaluation, error analysis, business insight.
- `app.py` – Streamlit app: order form (optional seller/customer zip prefixes for exact distance) → late-delivery risk and suggested action, plus a seller late-rate page.
- `artifacts/` – saved pipeline (`model.joblib`), metadata, seller stats and zip-code coordinates (`zip_coords.csv`), created by the notebook.

## Data

[Brazilian E-Commerce Public Dataset by Olist](https://www.kaggle.com/datasets/olistbr/brazilian-ecommerce) (Kaggle, ~100K orders). The project joins all 9 tables (orders, items, customers, sellers, products, payments, reviews, geolocation, category translation). Seller-to-customer distance is the straight-line distance between the median coordinates of their zip-code prefixes from the geolocation table (the small share of orders without a match falls back to the state capital). Not included in the repo: download the CSV files into `data/`.

## How to run

```bash
pip install -r requirements.txt
# 1. put the Olist CSVs in ./data
# 2. run the notebook top to bottom (creates ./artifacts)
jupyter notebook notebook.ipynb
# 3. launch the app
streamlit run app.py
```

## Key design decisions

- **Target:** late = delivered date later than the estimated delivery date (delivered orders only).
- **No leakage:** only information available at purchase time is used. Approval, carrier and delivery timestamps and review data are excluded.
- **Metric:** recall and PR-AUC on the late class, because missing a late order costs more than a false warning. Accuracy is misleading on imbalanced data.
- **Threshold:** chosen on training-set cross-validated predictions to hit a target recall, then evaluated once on the untouched test set.

## Limitations

Orders are from 2016–2018 in Brazil. Distance is a straight line between zip-prefix areas, not the carrier's real route. Much of the signal comes from purchase month (specific 2017–2018 congestion periods): in a time-based check (train on older orders, test on the newest 20%) ROC-AUC falls to 0.70 and PR-AUC to 0.07, so the model needs regular retraining. Using exact zip-code distance instead of state-capital distance did not change test performance noticeably (PR-AUC 0.258 vs 0.260). The model gives probabilities, not guarantees.

## Credits

Dataset: Olist & André Sionek, Kaggle. Built for the NextGen Institute Machine Learning final project.
