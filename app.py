import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, confusion_matrix, ConfusionMatrixDisplay,
    classification_report
)
import warnings
warnings.filterwarnings("ignore")

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Wine Quality Classifier",
    page_icon="🍷",
    layout="wide",
)

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    .metric-card {
        background: #f8f7f4;
        border: 1px solid #d3d1c7;
        border-radius: 10px;
        padding: 16px;
        text-align: center;
    }
    .metric-val  { font-size: 28px; font-weight: 700; color: #1a1a18; }
    .metric-lbl  { font-size: 12px; color: #888780; margin-top: 2px; }
    .pred-better { background:#e1f5ee; border:1.5px solid #1D9E75;
                   border-radius:10px; padding:20px; text-align:center; }
    .pred-lower  { background:#faece7; border:1.5px solid #D85A30;
                   border-radius:10px; padding:20px; text-align:center; }
    .pred-title  { font-size:22px; font-weight:700; margin-bottom:4px; color:#555;}
    .pred-sub    { font-size:13px; color:#555; }
    section[data-testid="stSidebar"] { background: #f4f3f0; }

    /* Sidebar — force dark readable text in light mode */
    section[data-testid="stSidebar"],
    section[data-testid="stSidebar"] p,
    section[data-testid="stSidebar"] span,
    section[data-testid="stSidebar"] label,
    section[data-testid="stSidebar"] div,
    section[data-testid="stSidebar"] small { color: #1a1a18 !important; }

    section[data-testid="stSidebar"] h1,
    section[data-testid="stSidebar"] h2,
    section[data-testid="stSidebar"] h3 { color: #1a1a18 !important; font-weight: 600; }

    /* Radio button labels */
    section[data-testid="stSidebar"] .stRadio label,
    section[data-testid="stSidebar"] .stRadio span { color: #1a1a18 !important; font-weight: 500; }

    /* Caption — slightly muted but still readable */
    section[data-testid="stSidebar"] .stCaption,
    section[data-testid="stSidebar"] .stCaption p { color: #444441 !important; }
</style>
""", unsafe_allow_html=True)


# ── Load & prepare data ───────────────────────────────────────────────────────
@st.cache_data
def load_data():
    df = pd.read_csv("winequality-red.csv")
    df["quality_binary"] = (df["quality"] >= 6).astype(int)
    return df

@st.cache_resource
def train_models(df):
    feature_cols = [c for c in df.columns if c not in ("quality", "quality_binary")]
    X = df[feature_cols]
    y = df["quality_binary"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )

    # Logistic Regression
    lr = Pipeline([
        ("scaler", StandardScaler()),
        ("lr", LogisticRegression(penalty="l2", C=1.0, solver="lbfgs",
                                   max_iter=1000, random_state=42))
    ])
    lr.fit(X_train, y_train)

    # Decision Tree
    dt = DecisionTreeClassifier(max_depth=5, min_samples_split=20,
                                 min_samples_leaf=10, random_state=42)
    dt.fit(X_train, y_train)

    # Random Forest
    rf = RandomForestClassifier(n_estimators=300, max_depth=10,
                                 max_features="sqrt", min_samples_leaf=5,
                                 random_state=42, n_jobs=-1)
    rf.fit(X_train, y_train)

    return {
        "Logistic Regression": lr,
        "Decision Tree": dt,
        "Random Forest": rf,
    }, X_train, X_test, y_train, y_test, feature_cols


# ── Sidebar ───────────────────────────────────────────────────────────────────
st.sidebar.image("https://img.icons8.com/color/96/wine-bottle.png", width=60)
st.sidebar.title("🍷 Wine Quality")
st.sidebar.markdown("Red wine quality classifier using three ML models.")
st.sidebar.divider()

page = st.sidebar.radio(
    "Navigate",
    ["🔮 Predict", "📊 Model Performance", "📈 Data Explorer"],
    index=0,
)
st.sidebar.divider()
st.sidebar.caption("Dataset: UCI Wine Quality · 1,599 samples · 11 features")


# ── Load ──────────────────────────────────────────────────────────────────────
try:
    df = load_data()
    models, X_train, X_test, y_train, y_test, feature_cols = train_models(df)
except FileNotFoundError:
    st.error("⚠️  `winequality-red.csv` not found. Place it in the same folder as `app.py` and restart.")
    st.stop()


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 1 — PREDICT
# ══════════════════════════════════════════════════════════════════════════════
if page == "🔮 Predict":
    st.title("🔮 Predict Wine Quality")
    st.markdown("Adjust the sliders to match your wine's physicochemical profile, then click **Predict**.")

    col_input, col_result = st.columns([1.1, 0.9], gap="large")

    with col_input:
        st.subheader("Wine Profile")

        c1, c2 = st.columns(2)
        with c1:
            fixed_acidity      = st.slider("Fixed acidity",       4.0, 16.0, 7.4,  0.1)
            citric_acid        = st.slider("Citric acid",         0.0,  1.0, 0.26, 0.01)
            chlorides          = st.slider("Chlorides",           0.01, 0.62, 0.08, 0.01)
            density            = st.slider("Density",             0.990, 1.004, 0.997, 0.001)
            sulphates          = st.slider("Sulphates",           0.33, 2.0,  0.66, 0.01)
        with c2:
            volatile_acidity   = st.slider("Volatile acidity",    0.12, 1.58, 0.52, 0.01)
            residual_sugar     = st.slider("Residual sugar",      1.0,  15.0, 2.5,  0.1)
            free_sulfur_dio    = st.slider("Free sulfur dioxide", 1.0,  72.0, 15.0, 1.0)
            pH                 = st.slider("pH",                  2.7,  4.0,  3.31, 0.01)
            alcohol            = st.slider("Alcohol (%)",         8.0,  15.0, 10.4, 0.1)

        total_sulfur_dio = st.slider("Total sulfur dioxide", 6.0, 289.0, 46.0, 1.0)

        model_choice = st.selectbox(
            "Model",
            ["Random Forest", "Logistic Regression", "Decision Tree"],
        )

        predict_btn = st.button("Predict Quality", type="primary", use_container_width=True)

    with col_result:
        st.subheader("Prediction Result")

        if predict_btn:
            input_data = pd.DataFrame([[
                fixed_acidity, volatile_acidity, citric_acid, residual_sugar,
                chlorides, free_sulfur_dio, total_sulfur_dio, density,
                pH, sulphates, alcohol
            ]], columns=feature_cols)

            model     = models[model_choice]
            pred      = model.predict(input_data)[0]
            proba     = model.predict_proba(input_data)[0]
            confidence = proba[pred] * 100

            if pred == 1:
                st.markdown(f"""
                <div class="pred-better">
                    <div class="pred-title">✅ Better Quality</div>
                    <div class="pred-sub">Quality score predicted: 6 – 8</div>
                    <br>
                    <div style="font-size:32px; font-weight:700; color:#0F6E56;">{confidence:.1f}%</div>
                    <div style="font-size:12px; color:#555;">model confidence</div>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div class="pred-lower">
                    <div class="pred-title">⚠️ Lower Quality</div>
                    <div class="pred-sub">Quality score predicted: 3 – 5</div>
                    <br>
                    <div style="font-size:32px; font-weight:700; color:#993C1D;">{confidence:.1f}%</div>
                    <div style="font-size:12px; color:#555;">model confidence</div>
                </div>
                """, unsafe_allow_html=True)

            st.divider()
            st.markdown("**Probability breakdown**")
            prob_df = pd.DataFrame({
                "Class": ["Lower (0)", "Better (1)"],
                "Probability": [f"{proba[0]*100:.1f}%", f"{proba[1]*100:.1f}%"]
            })
            st.dataframe(prob_df, hide_index=True, use_container_width=True)

            # Feature importance / coefficients
            st.markdown("**Top influencing features** (this model)")
            if model_choice == "Logistic Regression":
                coefs = pd.Series(
                    model.named_steps["lr"].coef_[0],
                    index=feature_cols
                ).abs().sort_values(ascending=False).head(6)
                fig, ax = plt.subplots(figsize=(5, 2.8))
                ax.barh(coefs.index[::-1], coefs.values[::-1], color="#378ADD")
                ax.set_xlabel("|Coefficient|", fontsize=9)
                ax.tick_params(labelsize=8)
                fig.tight_layout()
                st.pyplot(fig)
                plt.close()
            else:
                attr = "feature_importances_"
                estimator = model if model_choice == "Decision Tree" else model
                imp = pd.Series(
                    estimator.feature_importances_,
                    index=feature_cols
                ).sort_values(ascending=False).head(6)
                fig, ax = plt.subplots(figsize=(5, 2.8))
                ax.barh(imp.index[::-1], imp.values[::-1], color="#378ADD")
                ax.set_xlabel("Importance", fontsize=9)
                ax.tick_params(labelsize=8)
                fig.tight_layout()
                st.pyplot(fig)
                plt.close()
        else:
            st.info("Set the wine profile on the left and click **Predict Quality**.")


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 2 — MODEL PERFORMANCE
# ══════════════════════════════════════════════════════════════════════════════
elif page == "📊 Model Performance":
    st.title("📊 Model Performance")
    st.markdown("Evaluation of all three models on the held-out **test set** (20 % of 1,599 samples).")

    # Compute metrics for all models
    rows = []
    for name, model in models.items():
        y_pred = model.predict(X_test)
        rows.append({
            "Model":     name,
            "Accuracy":  round(accuracy_score(y_test, y_pred), 4),
            "Precision": round(precision_score(y_test, y_pred), 4),
            "Recall":    round(recall_score(y_test, y_pred), 4),
            "F1":        round(f1_score(y_test, y_pred), 4),
        })
    results_df = pd.DataFrame(rows).set_index("Model")

    # ── Metric cards ─────────────────────────────────────────────────────────
    selected_model = st.selectbox("Select model to inspect", list(models.keys()), index=2)
    row = results_df.loc[selected_model]

    m1, m2, m3, m4 = st.columns(4)
    for col, metric in zip([m1, m2, m3, m4], ["Accuracy", "Precision", "Recall", "F1"]):
        with col:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-val">{row[metric]:.3f}</div>
                <div class="metric-lbl">{metric}</div>
            </div>
            """, unsafe_allow_html=True)

    st.divider()

    col_cm, col_bar = st.columns(2, gap="large")

    # ── Confusion matrix ─────────────────────────────────────────────────────
    with col_cm:
        st.subheader("Confusion matrix")
        y_pred = models[selected_model].predict(X_test)
        cm = confusion_matrix(y_test, y_pred)
        fig, ax = plt.subplots(figsize=(4, 3.5))
        ConfusionMatrixDisplay(cm, display_labels=["lower", "better"]).plot(
            ax=ax, colorbar=False, cmap="Blues"
        )
        ax.set_title(selected_model, fontsize=11)
        fig.tight_layout()
        st.pyplot(fig)
        plt.close()

    # ── Comparison bar chart ──────────────────────────────────────────────────
    with col_bar:
        st.subheader("All models — F1 comparison")
        fig, ax = plt.subplots(figsize=(4, 3.5))
        colors = ["#6baed6" if m != selected_model else "#2171b5"
                  for m in results_df.index]
        ax.bar(results_df.index, results_df["F1"], color=colors, edgecolor="white")
        ax.set_ylim(0, 1)
        ax.set_ylabel("F1 Score")
        for i, (model_name, val) in enumerate(results_df["F1"].items()):
            ax.text(i, val + 0.01, f"{val:.3f}", ha="center", fontsize=9)
        ax.tick_params(axis="x", labelsize=8)
        fig.tight_layout()
        st.pyplot(fig)
        plt.close()

    st.divider()

    # ── Full results table ───────────────────────────────────────────────────
    st.subheader("Full results table")
    st.dataframe(
        results_df.style
        .highlight_max(color="#c6e2ff")
        .format(precision=4),
        use_container_width=True
    )

    # ── Classification report ────────────────────────────────────────────────
    with st.expander(f"Classification report — {selected_model}"):
        y_pred = models[selected_model].predict(X_test)
        report = classification_report(y_test, y_pred,
                                        target_names=["lower", "better"])
        st.code(report)


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 3 — DATA EXPLORER
# ══════════════════════════════════════════════════════════════════════════════
elif page == "📈 Data Explorer":
    st.title("📈 Data Explorer")
    st.markdown("Explore the raw dataset and feature distributions.")

    tab1, tab2, tab3 = st.tabs(["Dataset", "Feature distribution", "Correlation heatmap"])

    with tab1:
        st.markdown(f"**{len(df):,} samples · {len(feature_cols)} features · binary target**")
        class_counts = df["quality_binary"].value_counts().sort_index()
        c1, c2, c3 = st.columns(3)
        c1.metric("Total samples", len(df))
        c2.metric("Lower quality (0)", int(class_counts.get(0, 0)))
        c3.metric("Better quality (1)", int(class_counts.get(1, 0)))
        st.dataframe(df[feature_cols + ["quality", "quality_binary"]].head(50),
                     use_container_width=True)

    with tab2:
        feat = st.selectbox("Select feature", feature_cols, index=feature_cols.index("alcohol"))
        fig, axes = plt.subplots(1, 2, figsize=(9, 3.5))

        # Overall distribution
        sns.histplot(df[feat], kde=True, ax=axes[0], color="#378ADD",
                     edgecolor="white", linewidth=0.4)
        axes[0].set_title(f"{feat} — overall", fontsize=10)
        axes[0].set_xlabel("")

        # By class
        for label, color, name in zip([0, 1], ["#6baed6", "#2171b5"], ["lower", "better"]):
            sns.kdeplot(df[df["quality_binary"] == label][feat],
                        ax=axes[1], color=color, fill=True, alpha=0.35, label=name)
        axes[1].set_title(f"{feat} — by class", fontsize=10)
        axes[1].legend(fontsize=9)

        fig.tight_layout()
        st.pyplot(fig)
        plt.close()

    with tab3:
        fig, ax = plt.subplots(figsize=(9, 7))
        corr = df[feature_cols + ["quality_binary"]].corr()
        mask = np.triu(np.ones_like(corr, dtype=bool))
        sns.heatmap(corr, mask=mask, annot=True, fmt=".2f",
                    cmap="coolwarm", center=0, linewidths=0.5,
                    ax=ax, square=True, cbar_kws={"shrink": 0.8})
        ax.set_title("Pearson correlation matrix", fontsize=11)
        fig.tight_layout()
        st.pyplot(fig)
        plt.close()