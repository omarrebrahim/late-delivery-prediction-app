"""Olist Late-Delivery Risk app (Streamlit).

Tab 1: order form -> late-delivery risk + suggested action.
Tab 2: seller page -> late rate per seller.

Needs the files created by the last cells of the notebook:
    artifacts/model.joblib, artifacts/meta.json, artifacts/seller_stats.csv
"""
import json
import math
from datetime import date, time
from pathlib import Path
import joblib
import pandas as pd
import streamlit as st

ART = Path(__file__).parent / "artifacts"

st.set_page_config(page_title="Late-Delivery Risk", page_icon="📦", layout="wide")


@st.cache_resource
def load_model():
    model = joblib.load(ART / "model.joblib")
    meta = json.loads((ART / "meta.json").read_text())
    return model, meta


@st.cache_data
def load_sellers():
    return pd.read_csv(ART / "seller_stats.csv")


@st.cache_data
def load_zip_coords():
    path = ART / "zip_coords.csv"
    if not path.exists():
        return {}
    z = pd.read_csv(path, dtype={"zip_prefix": str})
    z["zip_prefix"] = z["zip_prefix"].str.zfill(5)
    return {r.zip_prefix: (r.lat, r.lng) for r in z.itertuples(index=False)}


def locate(zip_text, state, centroids, zip_coords):
    """Coordinates for a 5-digit zip prefix; falls back to the state capital. Returns (lat, lng, used_zip)."""
    key = zip_text.strip().zfill(5) if zip_text.strip().isdigit() else None
    if key and key in zip_coords:
        lat, lng = zip_coords[key]
        return lat, lng, True
    lat, lng = centroids[state]
    return lat, lng, False


def haversine_km(lat1, lon1, lat2, lon2):
    r = 6371.0
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dphi = p2 - p1
    dl = math.radians(lon2 - lon1)
    a = math.sin(dphi / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dl / 2) ** 2
    return 2 * r * math.asin(math.sqrt(a))


if not (ART / "model.joblib").exists():
    st.error("Model files not found. Run the notebook first so it creates the `artifacts/` folder.")
    st.stop()

model, meta = load_model()
sellers = load_sellers()
zip_coords = load_zip_coords()
defaults = meta["defaults"]
centroids = meta["state_centroids"]
threshold = meta["threshold"]

st.title("📦 Late-Delivery Risk Checker")
st.caption(
    "Predicts, at purchase time, how likely an order is to arrive after its estimated "
    "delivery date. Trained on the public Olist Brazilian e-commerce dataset."
)

tab_order, tab_seller = st.tabs(["Order risk", "Seller late rates"])

# ---------------------------------------------------------------- order form
with tab_order:
    with st.form("order_form"):
        c1, c2, c3 = st.columns(3)
        with c1:
            category = st.selectbox("Product category", meta["categories"])
            price = st.number_input("Item price (BRL)", min_value=1.0, max_value=10000.0,
                                    value=float(defaults["price"]), step=5.0)
            freight = st.number_input("Freight (BRL)", min_value=0.0, max_value=1000.0,
                                      value=float(defaults["freight_value"]), step=1.0)
            n_items = st.number_input("Number of items", min_value=1, max_value=20, value=1)
        with c2:
            seller_state = st.selectbox("Seller state", meta["states"],
                                        index=meta["states"].index("SP") if "SP" in meta["states"] else 0)
            customer_state = st.selectbox("Customer state", meta["states"],
                                          index=meta["states"].index("RJ") if "RJ" in meta["states"] else 0)
            seller_zip = st.text_input("Seller zip prefix (5 digits, optional)", max_chars=5,
                                       help="Gives a more accurate distance. Empty = state capital is used.")
            customer_zip = st.text_input("Customer zip prefix (5 digits, optional)", max_chars=5,
                                         help="Gives a more accurate distance. Empty = state capital is used.")
            weight = st.number_input("Weight (grams)", min_value=1.0, max_value=40000.0,
                                     value=float(defaults["product_weight_g"]), step=50.0)
            volume = st.number_input("Volume (cm³)", min_value=1.0, max_value=200000.0,
                                     value=float(defaults["product_volume_cm3"]), step=500.0)
        with c3:
            est_days = st.number_input("Estimated delivery time shown to customer (days)",
                                       min_value=1.0, max_value=120.0,
                                       value=float(defaults["est_delivery_days"]), step=1.0)
            payment_type = st.selectbox("Payment type", meta["payment_types"])
            installments = st.number_input("Installments", min_value=1, max_value=24, value=1)
            purchase_date = st.date_input("Purchase date", value=date.today())
            purchase_time = st.time_input("Purchase time", value=time(12, 0))
        submitted = st.form_submit_button("Check risk", type="primary")

    if submitted:
        s_lat, s_lon, s_zip_ok = locate(seller_zip, seller_state, centroids, zip_coords)
        c_lat, c_lon, c_zip_ok = locate(customer_zip, customer_state, centroids, zip_coords)
        row = {
            "price": price,
            "freight_value": freight,
            "freight_ratio": freight / price if price > 0 else None,
            "n_items": n_items,
            "product_weight_g": weight,
            "product_volume_cm3": volume,
            "distance_km": haversine_km(s_lat, s_lon, c_lat, c_lon),
            "same_state": int(seller_state == customer_state),
            "est_delivery_days": est_days,
            "installments": installments,
            "purchase_month": purchase_date.month,
            "purchase_dow": purchase_date.weekday(),
            "purchase_hour": purchase_time.hour,
            "product_category": category,
            "seller_state": seller_state,
            "customer_state": customer_state,
            "payment_type": payment_type,
        }
        X = pd.DataFrame([row])[meta["features"]]
        proba = float(model.predict_proba(X)[0, 1])

        if proba >= threshold:
            level, action = "HIGH", "Warn the customer, prioritise this order with the carrier, or upgrade shipping."
        elif proba >= threshold / 2:
            level, action = "MEDIUM", "Monitor this order; no action needed unless it is delayed at the carrier."
        else:
            level, action = "LOW", "No action needed."

        m1, m2, m3 = st.columns(3)
        m1.metric("Late-delivery probability", f"{proba:.1%}")
        m2.metric("Risk level", level)
        m3.metric("Seller → customer distance", f"{row['distance_km']:,.0f} km")
        st.info(f"**Suggested action:** {action}")
        st.caption(
            f"HIGH means the probability is at or above the model's alert threshold ({threshold:.2f}). "
            + ("Distance is computed between the two zip-code areas."
               if (s_zip_ok and c_zip_ok) else
               "Distance is approximate (a state capital is used for any side without a valid zip prefix).")
        )

# --------------------------------------------------------------- seller page
with tab_seller:
    platform_rate = meta.get("platform_late_rate")
    min_orders = st.slider("Minimum orders per seller", 1, 200, 30)
    view = sellers[sellers["n_orders"] >= min_orders].sort_values("late_rate", ascending=False)
    st.write(f"{len(view):,} sellers with at least {min_orders} orders"
             + (f" · platform late rate {platform_rate:.1%}" if platform_rate is not None else ""))
    st.dataframe(
        view.assign(late_rate=(view["late_rate"] * 100).round(1)).rename(columns={"late_rate": "late_rate_%"}),
        width="stretch",
        hide_index=True,
    )
    seller_id = st.text_input("Look up a seller ID")
    if seller_id:
        hit = sellers[sellers["seller_id"] == seller_id.strip()]
        if hit.empty:
            st.warning("Seller not found.")
        else:
            r = hit.iloc[0]
            st.success(f"{int(r['n_orders'])} orders · late rate {r['late_rate']:.1%} · state {r['seller_state']}")

st.divider()
st.caption("Model limitations: trained on 2016–2018 Brazilian orders; state-level distance only; "
           "month effects reflect past peaks such as Black Friday and early 2018; "
           "predictions are probabilities, not guarantees.")
