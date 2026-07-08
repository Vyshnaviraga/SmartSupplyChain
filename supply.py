"""
📦 Smart Supply Chain Analytics
--------------------------------
Late Delivery Risk Prediction using Machine Learning (XGBoost)

A professional, portfolio-quality Streamlit dashboard for predicting the
risk of a late delivery in a supply chain, built for resume / interview
demonstration purposes.

Run with:
    streamlit run supply.py

Requires (same folder):
    xgboost_model.pkl
    label_encoders.pkl
"""

import hashlib
import pickle
import warnings
import numpy as np
import pandas as pd
import streamlit as st
import xgboost as xgb
import plotly.graph_objects as go
import plotly.express as px
from sklearn.exceptions import InconsistentVersionWarning

# Suppress unpickling version warnings for clean deployment logs
warnings.filterwarnings("ignore", category=InconsistentVersionWarning)

# ==============================================================================
# CONFIG
# ==============================================================================

MODEL_PATH = "xgboost_model.pkl"
ENCODERS_PATH = "label_encoders.pkl"

# Exact feature order the model was trained on. DO NOT REORDER, DO NOT RENAME.
FEATURE_ORDER = [
    "Type",
    "Sales per customer",
    "Category Id",
    "Category Name",
    "Customer City",
    "Customer Country",
    "Customer Id",
    "Customer Segment",
    "Customer State",
    "Department Id",
    "Department Name",
    "Market",
    "Order City",
    "Order Country",
    "Order Customer Id",
    "Order Id",
    "Order Item Cardprod Id",
    "Order Item Discount",
    "Order Item Discount Rate",
    "Order Item Id",
    "Order Item Product Price",
    "Order Item Profit Ratio",
    "Order Item Quantity",
    "Sales",
    "Order Item Total",
    "Order Profit Per Order",
    "Order Region",
    "Order State",
    "Product Card Id",
    "Product Category Id",
    "Product Name",
    "Product Price",
    "Product Status",
    "Order Year",
    "Order Month",
    "Order Day",
    "Shipping Month",
]

# ID / bookkeeping columns that are meaningless to a human user filling out a
# form. The model still needs a value for them, so a deterministic,
# reasonable placeholder is generated automatically instead of asking the
# user to type in raw database identifiers.
AUTO_ID_FIELDS = [
    "Customer Id",
    "Order Customer Id",
    "Order Id",
    "Order Item Id",
    "Order Item Cardprod Id",
    "Product Card Id",
    "Category Id",
    "Product Category Id",
    "Department Id",
]

# Sensible default numeric ranges/steps for a cleaner sidebar.
NUMERIC_HINTS = {
    "Sales per customer": {"min_value": 0.0, "value": 200.0, "step": 1.0},
    "Order Item Discount": {"min_value": 0.0, "value": 10.0, "step": 1.0},
    "Order Item Discount Rate": {"min_value": 0.0, "max_value": 1.0, "value": 0.1, "step": 0.01},
    "Order Item Product Price": {"min_value": 0.0, "value": 100.0, "step": 1.0},
    "Order Item Profit Ratio": {"min_value": -1.0, "max_value": 1.0, "value": 0.2, "step": 0.01},
    "Order Item Quantity": {"min_value": 1, "value": 1, "step": 1},
    "Sales": {"min_value": 0.0, "value": 200.0, "step": 1.0},
    "Order Item Total": {"min_value": 0.0, "value": 200.0, "step": 1.0},
    "Order Profit Per Order": {"min_value": -1000.0, "value": 20.0, "step": 1.0},
    "Product Price": {"min_value": 0.0, "value": 100.0, "step": 1.0},
    "Order Year": {"min_value": 2015, "max_value": 2030, "value": 2018, "step": 1},
    "Order Month": {"min_value": 1, "max_value": 12, "value": 1, "step": 1},
    "Order Day": {"min_value": 1, "max_value": 31, "value": 1, "step": 1},
    "Shipping Month": {"min_value": 1, "max_value": 12, "value": 1, "step": 1},
}

# Sidebar grouping (only fields meaningful to a human are shown here;
# AUTO_ID_FIELDS are generated automatically and never rendered).
INPUT_GROUPS = {
    "👤 Customer": ["Customer City", "Customer State", "Customer Country", "Customer Segment"],
    "📦 Product": ["Category Name", "Department Name", "Product Name", "Product Price", "Product Status"],
    "🧾 Order": ["Type", "Order Item Quantity", "Order Item Discount", "Order Item Discount Rate",
                "Order Item Product Price", "Order Item Profit Ratio"],
    "💰 Pricing": ["Sales per customer", "Sales", "Order Item Total", "Order Profit Per Order"],
    "🌍 Region": ["Market", "Order City", "Order State", "Order Country", "Order Region"],
    "📅 Time": ["Order Year", "Order Month", "Order Day", "Shipping Month"],
}


# ==============================================================================
# LOADING (cached so the model/encoders are only read from disk once)
# ==============================================================================

@st.cache_resource(show_spinner="Loading machine learning model...")
def load_model():
    """
    Load the trained XGBoost model from disk.

    Handles multiple possible save formats so a mismatched extension doesn't
    crash the app: a standard pickle, a joblib dump, or XGBoost's native
    save_model format (sklearn wrapper or raw Booster).
    """
    try:
        with open(MODEL_PATH, "rb") as f:
            return pickle.load(f)
    except Exception:
        pass

    try:
        import joblib
        return joblib.load(MODEL_PATH)
    except Exception:
        pass

    try:
        clf = xgb.XGBClassifier()
        clf.load_model(MODEL_PATH)
        return clf
    except Exception:
        pass

    try:
        booster = xgb.Booster()
        booster.load_model(MODEL_PATH)
        return booster
    except Exception as e:
        raise RuntimeError(
            f"Could not load '{MODEL_PATH}' as a pickle, joblib file, or "
            f"native XGBoost model. The file may be corrupted or saved in an "
            f"unsupported format. Original error: {e}"
        )


@st.cache_resource(show_spinner="Loading label encoders...")
def load_encoders():
    """Load the dictionary of fitted LabelEncoders used during training."""
    try:
        with open(ENCODERS_PATH, "rb") as f:
            return pickle.load(f)
    except Exception:
        pass

    try:
        import joblib
        return joblib.load(ENCODERS_PATH)
    except Exception as e:
        raise RuntimeError(
            f"Could not load '{ENCODERS_PATH}' as a pickle or joblib file. "
            f"The file may be corrupted or saved in an unsupported format. "
            f"Original error: {e}"
        )


# ==============================================================================
# UI HELPERS
# ==============================================================================

def inject_custom_css():
    """Inject a dark, premium custom theme with Inter font and glassmorphic card styles."""
    st.markdown(
        """
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
        
        /* Apply font to the entire Streamlit app */
        .stApp {
            font-family: 'Inter', sans-serif !important;
            background-color: #0b0d13;
            color: #e2e8f0;
        }
        
        /* Sidebar styling */
        section[data-testid="stSidebar"] {
            background-color: #0f121a;
            border-right: 1px solid #1e293b;
        }
        
        /* Headings */
        h1, h2, h3, h4, h5, h6 {
            font-family: 'Inter', sans-serif !important;
            font-weight: 700 !important;
            color: #f8fafc !important;
            letter-spacing: -0.02em;
        }
        
        .subtitle-text {
            color: #94a3b8;
            font-size: 1.1rem;
            margin-top: -12px;
            margin-bottom: 24px;
            font-weight: 400;
        }
        
        /* Glassmorphic cards */
        .glass-card {
            background: rgba(23, 28, 41, 0.6);
            backdrop-filter: blur(12px);
            -webkit-backdrop-filter: blur(12px);
            border: 1px solid rgba(255, 255, 255, 0.05);
            border-radius: 16px;
            padding: 24px;
            box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.37);
            margin-bottom: 20px;
        }
        
        /* Metric cards */
        .kpi-container {
            display: flex;
            gap: 16px;
            flex-wrap: wrap;
            margin-bottom: 24px;
        }
        
        .kpi-card {
            flex: 1;
            min-width: 220px;
            background: rgba(30, 41, 59, 0.4);
            border: 1px solid rgba(255, 255, 255, 0.05);
            border-radius: 12px;
            padding: 16px 20px;
            box-shadow: 0 4px 20px 0 rgba(0, 0, 0, 0.2);
            transition: transform 0.2s ease, border-color 0.2s ease;
        }
        
        .kpi-card:hover {
            transform: translateY(-2px);
            border-color: rgba(37, 99, 235, 0.4);
        }
        
        .kpi-label {
            font-size: 0.85rem;
            color: #94a3b8;
            font-weight: 500;
            text-transform: uppercase;
            letter-spacing: 0.05em;
        }
        
        .kpi-value {
            font-size: 1.8rem;
            font-weight: 700;
            color: #f1f5f9;
            margin-top: 4px;
        }
        
        /* Prediction cards */
        .result-card-safe {
            background: linear-gradient(135deg, rgba(16, 185, 129, 0.15) 0%, rgba(5, 150, 105, 0.05) 100%);
            border: 1px solid rgba(16, 185, 129, 0.3);
            border-radius: 16px;
            padding: 24px;
            text-align: center;
            box-shadow: 0 8px 32px 0 rgba(16, 185, 129, 0.1);
        }
        
        .result-card-safe .status-title {
            font-size: 1.6rem;
            font-weight: 700;
            color: #10b981;
            margin-bottom: 8px;
        }
        
        .result-card-risk {
            background: linear-gradient(135deg, rgba(239, 68, 68, 0.15) 0%, rgba(220, 38, 38, 0.05) 100%);
            border: 1px solid rgba(239, 68, 68, 0.3);
            border-radius: 16px;
            padding: 24px;
            text-align: center;
            box-shadow: 0 8px 32px 0 rgba(239, 68, 68, 0.1);
        }
        
        .result-card-risk .status-title {
            font-size: 1.6rem;
            font-weight: 700;
            color: #ef4444;
            margin-bottom: 8px;
        }
        
        /* Recommendations box */
        .reco-container {
            margin-top: 20px;
        }
        
        .reco-card {
            background: rgba(30, 41, 59, 0.3);
            border-left: 4px solid #3b82f6;
            border-radius: 4px 12px 12px 4px;
            padding: 14px 20px;
            margin-bottom: 12px;
            font-size: 0.95rem;
            color: #e2e8f0;
            display: flex;
            align-items: center;
            gap: 12px;
        }
        
        .reco-card.high-risk {
            border-left-color: #ef4444;
        }
        
        .reco-card.low-risk {
            border-left-color: #10b981;
        }
        
        /* Streamlit overrides */
        .stButton>button {
            background: linear-gradient(135deg, #2563eb 0%, #1d4ed8 100%) !important;
            color: white !important;
            border-radius: 10px !important;
            border: none !important;
            padding: 12px 24px !important;
            font-weight: 600 !important;
            transition: all 0.3s ease !important;
            box-shadow: 0 4px 12px rgba(37, 99, 235, 0.2) !important;
            width: 100%;
        }
        
        .stButton>button:hover {
            transform: translateY(-1px) !important;
            box-shadow: 0 6px 20px rgba(37, 99, 235, 0.4) !important;
        }
        
        /* Expander styling */
        div[data-testid="stExpander"] {
            background-color: rgba(30, 41, 59, 0.2) !important;
            border: 1px solid rgba(255, 255, 255, 0.05) !important;
            border-radius: 12px !important;
        }
        
        /* Tab styling */
        button[data-baseweb="tab"] {
            font-weight: 600 !important;
            color: #94a3b8 !important;
        }
        
        button[data-baseweb="tab"][aria-selected="true"] {
            color: #3b82f6 !important;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def render_header():
    """Render the app title, subtitle and short project description."""
    st.markdown("## 📦 Smart Supply Chain Analytics")
    st.markdown(
        '<p class="subtitle-text">Late Delivery Risk Prediction using Machine Learning</p>',
        unsafe_allow_html=True,
    )
    st.markdown(
        """
        <div class="glass-card">
        This dashboard uses a trained <b>XGBoost</b> classifier to estimate the risk of a
        <b>late delivery</b> for an order, based on customer, product, order, pricing,
        regional, and timing attributes. Configure the order details in the sidebar and
        click <b>Predict</b> to view the model's risk assessment, KPIs, and business
        recommendations.
        </div>
        """,
        unsafe_allow_html=True,
    )


def _stable_hash(seed_str: str) -> int:
    """Deterministic integer hash used to build reasonable placeholder IDs."""
    return int(hashlib.md5(seed_str.encode()).hexdigest(), 16)


def _render_single_input(col: str, encoders: dict):
    """Render one sidebar widget for a given human-meaningful feature."""
    if col in encoders:
        options = list(encoders[col].classes_)
        return st.selectbox(col, options, key=f"input_{col}")

    if col == "Product Status":
        # Small binary toggle instead of a raw number.
        choice = st.selectbox("Product Status", ["Active", "Discontinued"], key="input_Product Status")
        return 0 if choice == "Active" else 1

    hint = NUMERIC_HINTS.get(col, {})
    is_int_like = isinstance(hint.get("value"), int)

    if is_int_like:
        return st.number_input(
            col,
            min_value=int(hint.get("min_value", 0)),
            max_value=int(hint.get("max_value", 10_000_000)) if "max_value" in hint else None,
            value=int(hint.get("value", 0)),
            step=int(hint.get("step", 1)),
            key=f"input_{col}",
        )

    return st.number_input(
        col,
        min_value=float(hint.get("min_value", -1_000_000.0)),
        max_value=float(hint.get("max_value", 1_000_000.0)) if "max_value" in hint else None,
        value=float(hint.get("value", 0.0)),
        step=float(hint.get("step", 1.0)),
        format="%.4f",
        key=f"input_{col}",
    )


def create_sidebar(encoders: dict):
    """
    Build a grouped, uncluttered sidebar containing only business-meaningful
    inputs. Internal database IDs are never shown to the user.

    Returns a dict of {feature_name: raw_user_value} and whether Predict was clicked.
    """
    st.sidebar.markdown("### 🎛️ Order Configuration")
    st.sidebar.caption("Fill in the order details below, grouped by category.")

    raw_inputs = {}

    for group_name, columns in INPUT_GROUPS.items():
        with st.sidebar.expander(group_name, expanded=(group_name == "👤 Customer")):
            for col in columns:
                if col not in FEATURE_ORDER:
                    continue
                raw_inputs[col] = _render_single_input(col, encoders)

    st.sidebar.markdown("---")
    predict_clicked = st.sidebar.button("🔮 Predict Late Delivery Risk", use_container_width=True)

    return raw_inputs, predict_clicked


def fill_auto_id_fields(raw_inputs: dict, encoders: dict) -> dict:
    """
    Populate the internal bookkeeping/ID columns that were intentionally
    hidden from the user with deterministic, reasonable placeholder values
    that match the training distribution of the DataCo dataset.
    """
    # 1. Category Id & Product Category Id (synchronized and mapped to Category Name)
    category_name = raw_inputs.get("Category Name")
    if category_name and "Category Name" in encoders:
        try:
            cat_classes = list(encoders["Category Name"].classes_)
            category_idx = cat_classes.index(category_name)
            category_id = category_idx + 2  # range is [2, 51]
        except ValueError:
            category_id = 2 + (_stable_hash(category_name) % 50)
    else:
        category_id = 2 + (_stable_hash(str(category_name)) % 50)
    raw_inputs["Category Id"] = int(category_id)
    raw_inputs["Product Category Id"] = int(category_id)

    # 2. Product Card Id & Order Item Cardprod Id (synchronized and mapped to Product Name)
    product_name = raw_inputs.get("Product Name")
    if product_name and "Product Name" in encoders:
        try:
            prod_classes = list(encoders["Product Name"].classes_)
            product_idx = prod_classes.index(product_name)
            product_id = product_idx + 19  # range is [19, 137]
        except ValueError:
            product_id = 19 + (_stable_hash(product_name) % 118)
    else:
        product_id = 19 + (_stable_hash(str(product_name)) % 118)
    raw_inputs["Product Card Id"] = int(product_id)
    raw_inputs["Order Item Cardprod Id"] = int(product_id)

    # 3. Department Id (mapped to Department Name)
    dept_name = raw_inputs.get("Department Name")
    if dept_name and "Department Name" in encoders:
        try:
            dept_classes = list(encoders["Department Name"].classes_)
            dept_idx = dept_classes.index(dept_name)
            dept_id = dept_idx + 2  # range is [2, 12]
        except ValueError:
            dept_id = 2 + (_stable_hash(dept_name) % 11)
    else:
        dept_id = 2 + (_stable_hash(str(dept_name)) % 11)
    raw_inputs["Department Id"] = int(dept_id)

    # 4. Customer Id & Order Customer Id (synchronized and mapped to demographics)
    cust_seed = "|".join(str(raw_inputs.get(k, "")) for k in 
                          ["Customer City", "Customer State", "Customer Segment", "Customer Country"])
    customer_id = 1000 + (_stable_hash(cust_seed) % 19000)
    raw_inputs["Customer Id"] = int(customer_id)
    raw_inputs["Order Customer Id"] = int(customer_id)

    # 5. Order Id & Order Item Id (mapped to customer and order date)
    order_seed = f"{customer_id}|{raw_inputs.get('Order Year', '')}|{raw_inputs.get('Order Month', '')}|{raw_inputs.get('Order Day', '')}"
    order_id = 10000 + (_stable_hash(order_seed) % 67000)
    raw_inputs["Order Id"] = int(order_id)
    raw_inputs["Order Item Id"] = int(order_id + 1)

    return raw_inputs


# ==============================================================================
# PREPROCESSING
# ==============================================================================

def preprocess_input(raw_inputs: dict, encoders: dict) -> pd.DataFrame:
    """
    Convert raw sidebar inputs (plus auto-generated ID fields) into a
    single-row DataFrame matching the exact feature order/names the model
    was trained on. Categorical values are label-encoded using the fitted
    encoders; unseen labels raise a clear, catchable error.
    """
    row = {}

    for col in FEATURE_ORDER:
        value = raw_inputs.get(col)

        if col in encoders:
            encoder = encoders[col]
            try:
                row[col] = encoder.transform([value])[0]
            except ValueError as exc:
                raise ValueError(
                    f"The value '{value}' for feature '{col}' was not seen during "
                    f"training and cannot be encoded."
                ) from exc
        else:
            row[col] = value

    return pd.DataFrame([row], columns=FEATURE_ORDER)


# ==============================================================================
# PREDICTION
# ==============================================================================

def predict(model, input_df: pd.DataFrame):
    """
    Run prediction and return (predicted_class, probability_array).

    Supports both the sklearn-style XGBClassifier API and the raw
    xgb.Booster API.
    """
    if isinstance(model, xgb.Booster):
        dmatrix = xgb.DMatrix(input_df)
        prob_late = float(model.predict(dmatrix)[0])
        prob_on_time = 1.0 - prob_late
        return int(prob_late >= 0.5), np.array([prob_on_time, prob_late])

    prediction = model.predict(input_df)[0]
    probabilities = model.predict_proba(input_df)[0]
    return int(prediction), probabilities


def get_feature_importance(model, columns):
    """Return a sorted pandas Series of feature importances, or None."""
    try:
        if isinstance(model, xgb.Booster):
            scores = model.get_score(importance_type="gain")
            if not scores:
                return None
            series = pd.Series(scores)
        else:
            importances = getattr(model, "feature_importances_", None)
            if importances is None:
                return None
            series = pd.Series(importances, index=columns)
        return series.sort_values(ascending=False).head(15)
    except Exception:
        return None


# ==============================================================================
# RESULT / DASHBOARD RENDERING
# ==============================================================================

def render_prediction_result(prediction: int, probabilities: np.ndarray):
    """Display the prediction outcome with interactive gauge and metrics."""
    prob_on_time = float(probabilities[0]) * 100
    prob_late = float(probabilities[1]) * 100
    confidence = max(prob_on_time, prob_late)

    st.markdown("### 🧠 Risk Assessment")

    # Render glassmorphic card based on prediction
    if prediction == 0:
        st.markdown(
            f"""
            <div class="result-card-safe">
                <div class="status-title">🟢 Low Late Delivery Risk</div>
                <p style="margin:0; color:#cbd5e1; font-size:0.95rem;">
                    The model predicts this shipment will arrive <b>on time</b> with <b>{confidence:.1f}%</b> confidence.
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            f"""
            <div class="result-card-risk">
                <div class="status-title">🔴 High Late Delivery Risk</div>
                <p style="margin:0; color:#cbd5e1; font-size:0.95rem;">
                    The model predicts this shipment is at <b>high risk of delay</b> with <b>{confidence:.1f}%</b> confidence.
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.write("")

    # Columns for Gauge and Probability Bars
    col1, col2 = st.columns([1, 1.2])

    with col1:
        # Beautiful Plotly Gauge Chart for risk indicator
        fig = go.Figure(
            go.Indicator(
                mode="gauge+number",
                value=prob_late,
                domain={"x": [0, 1], "y": [0, 1]},
                title={"text": "Late Delivery Probability", "font": {"size": 14, "color": "#94a3b8"}},
                number={"suffix": "%", "font": {"size": 36, "color": "#ef4444" if prediction == 1 else "#10b981", "family": "Inter"}},
                gauge={
                    "axis": {"range": [0, 100], "tickwidth": 1, "tickcolor": "#94a3b8"},
                    "bar": {"color": "#ef4444" if prediction == 1 else "#10b981", "thickness": 0.25},
                    "bgcolor": "rgba(30, 41, 59, 0.3)",
                    "borderwidth": 1,
                    "bordercolor": "rgba(255, 255, 255, 0.1)",
                    "steps": [
                        {"range": [0, 30], "color": "rgba(16, 185, 129, 0.05)"},
                        {"range": [30, 70], "color": "rgba(245, 158, 11, 0.05)"},
                        {"range": [70, 100], "color": "rgba(239, 68, 68, 0.05)"},
                    ],
                    "threshold": {
                        "line": {"color": "#ef4444", "width": 3},
                        "thickness": 0.75,
                        "value": 70,
                    },
                },
            )
        )
        fig.update_layout(
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font={"color": "#f8fafc", "family": "Inter"},
            height=220,
            margin=dict(l=20, r=20, t=30, b=10),
        )
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        # Side metrics using clean progress bars
        st.markdown("<p style='font-size:0.95rem; font-weight:600; color:#f8fafc; margin-bottom:8px;'>Probability Analysis</p>", unsafe_allow_html=True)
        
        # Show custom styled horizontal progress bars
        st.markdown(
            f"""
            <div style="background: rgba(30, 41, 59, 0.3); border-radius: 8px; padding: 12px 16px; border: 1px solid rgba(255, 255, 255, 0.05);">
                <div style="display:flex; justify-content:space-between; font-size:0.85rem; color:#94a3b8; font-weight:500;">
                    <span>On-Time Delivery</span>
                    <span>{prob_on_time:.1f}%</span>
                </div>
                <div style="background:rgba(255,255,255,0.05); border-radius:4px; height:8px; margin-top:6px; overflow:hidden;">
                    <div style="background:#10b981; width:{prob_on_time}%; height:100%;"></div>
                </div>
            </div>
            <div style="background: rgba(30, 41, 59, 0.3); border-radius: 8px; padding: 12px 16px; border: 1px solid rgba(255, 255, 255, 0.05); margin-top: 12px;">
                <div style="display:flex; justify-content:space-between; font-size:0.85rem; color:#94a3b8; font-weight:500;">
                    <span>Late Delivery Risk</span>
                    <span>{prob_late:.1f}%</span>
                </div>
                <div style="background:rgba(255,255,255,0.05); border-radius:4px; height:8px; margin-top:6px; overflow:hidden;">
                    <div style="background:#ef4444; width:{prob_late}%; height:100%;"></div>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    return prob_on_time, prob_late


def render_recommendations(prediction: int, prob_late: float):
    """Show actionable business recommendations based on the prediction and probability score."""
    st.markdown("### 📋 Business Action Plan")

    risk_class = "high-risk" if prediction == 1 else "low-risk"

    if prediction == 1:
        if prob_late >= 85:
            recos = [
                ("🚨", "CRITICAL WARNING: Late delivery is highly probable. Freeze standard processing and escalate to expedited dispatch."),
                ("🚚", "Reroute shipment via priority express carrier and flag the order for real-time tracking updates."),
                ("📞", "Proactively notify the account representative and the customer with an updated expected delivery date."),
                ("🛡️", "Review penalty clauses/SLAs associated with late delivery for this customer segment to mitigate financial risk."),
            ]
        else:
            recos = [
                ("⚠️", "MODERATE RISK WARNING: Late delivery risk is elevated. Trigger standard priority handling."),
                ("📣", "Alert warehouse logistics to prioritize picking and packing for this order number."),
                ("🔍", "Monitor carrier tracking updates daily for any transit disruptions or weather alerts."),
                ("🔄", "Ensure that backup inventory is identified in case a replacement order needs to be re-shipped."),
            ]
    else:
        recos = [
            ("✅", "Low risk detected. Proceed with standard logistics workflows."),
            ("📊", "Continue shipping schedule with selected standard carrier class."),
            ("🟢", "Customer expectations are aligned with standard shipping window — no special mitigation required."),
        ]

    st.markdown("<div class='reco-container'>", unsafe_allow_html=True)
    for icon, text in recos:
        st.markdown(
            f"""
            <div class="reco-card {risk_class}">
                <span style="font-size: 1.25rem;">{icon}</span>
                <span>{text}</span>
            </div>
            """,
            unsafe_allow_html=True,
        )
    st.markdown("</div>", unsafe_allow_html=True)


def render_dashboard(prediction: int, prob_late: float, prob_on_time: float, input_df: pd.DataFrame, model):
    """Render beautiful KPI metrics and interactive charts using tabs."""
    st.write("")
    st.markdown("### 📊 Dashboard & Model Insights")

    # Render custom HTML KPI cards
    order_qty = int(input_df["Order Item Quantity"].iloc[0])
    order_total = float(input_df['Order Item Total'].iloc[0])
    sales = float(input_df['Sales'].iloc[0])
    profit = float(input_df['Order Profit Per Order'].iloc[0])

    st.markdown(
        f"""
        <div class="kpi-container">
            <div class="kpi-card">
                <div class="kpi-label">Order Quantity</div>
                <div class="kpi-value">{order_qty}</div>
            </div>
            <div class="kpi-card">
                <div class="kpi-label">Order Total Value</div>
                <div class="kpi-value">${order_total:,.2f}</div>
            </div>
            <div class="kpi-card">
                <div class="kpi-label">Gross Sales</div>
                <div class="kpi-value">${sales:,.2f}</div>
            </div>
            <div class="kpi-card" style="border-left: 3px solid {'#ef4444' if profit < 0 else '#10b981'}">
                <div class="kpi-label">Projected Net Profit</div>
                <div class="kpi-value" style="color: {'#ef4444' if profit < 0 else '#10b981'}">${profit:,.2f}</div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    tab1, tab2 = st.tabs(["📊 Feature Insights", "🧬 Model Explanations (SHAP)"])

    with tab1:
        st.markdown("#### 🔎 Top Factors Influencing Delivery Risk")
        importance = get_feature_importance(model, list(FEATURE_ORDER))
        if importance is not None:
            # Convert series to dataframe for Plotly
            imp_df = pd.DataFrame({
                "Feature": importance.index,
                "Importance (Gain)": importance.values
            }).sort_values("Importance (Gain)", ascending=True)

            fig = px.bar(
                imp_df,
                x="Importance (Gain)",
                y="Feature",
                orientation="h",
                color="Importance (Gain)",
                color_continuous_scale="Blues",
                template="plotly_dark",
            )
            fig.update_layout(
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                coloraxis_showscale=False,
                margin=dict(l=10, r=10, t=10, b=10),
                height=400,
            )
            fig.update_xaxes(showgrid=True, gridcolor="rgba(255,255,255,0.05)")
            fig.update_yaxes(showgrid=False)
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Feature importance is not available for this model type.")

    with tab2:
        render_shap_explanation(model, input_df)


def render_shap_explanation(model, input_df: pd.DataFrame):
    """Display a SHAP explanation for this single prediction if shap is installed."""
    try:
        import shap
        import matplotlib.pyplot as plt

        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(input_df)

        st.markdown("#### 🧬 SHAP Explanation")
        fig, ax = plt.subplots(figsize=(8, 5))
        # Keep plot background dark to match the app theme
        fig.patch.set_facecolor('#0b0d13')
        ax.set_facecolor('#0f121a')
        shap.summary_plot(shap_values, input_df, plot_type="bar", show=False)
        # Style text labels
        for t in ax.get_xticklabels() + ax.get_yticklabels():
            t.set_color('#94a3b8')
        ax.xaxis.label.set_color('#f8fafc')
        ax.title.set_color('#f8fafc')
        
        st.pyplot(fig, use_container_width=True)
        plt.close(fig)
    except Exception:
        # Graceful dashboard placeholder when shap is not available locally
        st.markdown(
            """
            <div style="background: rgba(30, 41, 59, 0.2); border: 1px dashed rgba(255, 255, 255, 0.1); border-radius: 8px; padding: 20px; text-align: center; color: #94a3b8;">
                🧬 <b>Explainable AI (SHAP) is disabled.</b><br>
                To enable SHAP force plots, run <code>pip install shap</code> in your local terminal environment.
            </div>
            """,
            unsafe_allow_html=True,
        )


def render_processed_input(input_df: pd.DataFrame):
    with st.expander("🔍 View processed model input (debug view)"):
        st.dataframe(input_df, use_container_width=True)


# ==============================================================================
# MAIN APP
# ==============================================================================

def main():
    st.set_page_config(
        page_title="Smart Supply Chain Analytics",
        page_icon="📦",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    inject_custom_css()
    render_header()

    try:
        model = load_model()
        encoders = load_encoders()
    except FileNotFoundError as e:
        st.error(
            f"❌ Required file not found: `{e.filename}`. Make sure `xgboost_model.pkl` "
            f"and `label_encoders.pkl` are in the same folder as this app."
        )
        st.stop()
        return
    except Exception as e:
        st.error(f"❌ Failed to load model/encoders: {e}")
        st.stop()
        return

    raw_inputs, predict_clicked = create_sidebar(encoders)

    st.markdown("---")

    # Persistent state logic for Streamlit tabs/widgets interaction
    if "has_prediction" not in st.session_state:
        st.session_state.has_prediction = False
        st.session_state.prediction = None
        st.session_state.probabilities = None
        st.session_state.input_df = None

    if predict_clicked:
        try:
            # Create a copy to prevent side effects
            inputs_copy = raw_inputs.copy()
            inputs_copy = fill_auto_id_fields(inputs_copy, encoders)
            input_df = preprocess_input(inputs_copy, encoders)
            
            prediction, probabilities = predict(model, input_df)
            
            st.session_state.has_prediction = True
            st.session_state.prediction = prediction
            st.session_state.probabilities = probabilities
            st.session_state.input_df = input_df
        except ValueError as e:
            st.error(f"⚠️ Input validation failed: {e}")
            st.session_state.has_prediction = False
        except Exception as e:
            st.error(f"❌ Unexpected error while preparing input: {e}")
            st.session_state.has_prediction = False

    if st.session_state.has_prediction:
        render_processed_input(st.session_state.input_df)
        prob_on_time, prob_late = render_prediction_result(st.session_state.prediction, st.session_state.probabilities)
        render_recommendations(st.session_state.prediction, prob_late)
        render_dashboard(st.session_state.prediction, prob_late, prob_on_time, st.session_state.input_df, model)
    else:
        st.info("👈 Configure the order parameters in the sidebar, then click **Predict Late Delivery Risk**.")

    st.markdown("---")
    st.caption(
        "Model: XGBoost Classifier · Target: Late_delivery_risk · "
        "Built with Python, Streamlit & Machine Learning"
    )


if __name__ == "__main__":
    main()