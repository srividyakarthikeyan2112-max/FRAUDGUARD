import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier, IsolationForest
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
from imblearn.over_sampling import SMOTE
import warnings
import io

warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="FraudGuard · Detection System",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────
# CUSTOM CSS  (dark industrial aesthetic)
# ─────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;600&family=IBM+Plex+Sans:wght@300;400;600;700&display=swap');

html, body, [class*="css"] {
    font-family: 'IBM Plex Sans', sans-serif;
    background-color: #0d0f14;
    color: #e2e8f0;
}

/* Sidebar */
section[data-testid="stSidebar"] {
    background: #111318 !important;
    border-right: 1px solid #1e2330;
}
section[data-testid="stSidebar"] * { color: #cbd5e1 !important; }

/* Header */
.fraud-header {
    background: linear-gradient(135deg, #0f172a 0%, #1e1b4b 50%, #0f172a 100%);
    border: 1px solid #312e81;
    border-radius: 12px;
    padding: 2rem 2.5rem;
    margin-bottom: 1.5rem;
    position: relative;
    overflow: hidden;
}
.fraud-header::before {
    content: '';
    position: absolute; top: 0; left: 0; right: 0; bottom: 0;
    background: repeating-linear-gradient(
        45deg, transparent, transparent 10px,
        rgba(99,102,241,0.03) 10px, rgba(99,102,241,0.03) 20px
    );
}
.fraud-header h1 {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 2rem; font-weight: 600;
    color: #a5b4fc; margin: 0; letter-spacing: -0.5px;
}
.fraud-header p {
    color: #64748b; margin: 0.5rem 0 0; font-size: 0.9rem;
    font-family: 'IBM Plex Mono', monospace;
}

/* Metric cards */
.metric-card {
    background: #111318;
    border: 1px solid #1e2330;
    border-radius: 10px;
    padding: 1.2rem 1.5rem;
    text-align: center;
}
.metric-card .label {
    font-size: 0.7rem; text-transform: uppercase;
    letter-spacing: 2px; color: #475569;
    font-family: 'IBM Plex Mono', monospace;
}
.metric-card .value {
    font-size: 2rem; font-weight: 700;
    font-family: 'IBM Plex Mono', monospace;
    color: #a5b4fc; line-height: 1.1;
}
.metric-card .sub { font-size: 0.75rem; color: #475569; margin-top: 4px; }

/* Risk badge */
.risk-critical { color: #f87171; font-weight: 700; font-family: 'IBM Plex Mono', monospace; }
.risk-high     { color: #fb923c; font-weight: 700; font-family: 'IBM Plex Mono', monospace; }
.risk-medium   { color: #fbbf24; font-weight: 600; font-family: 'IBM Plex Mono', monospace; }
.risk-low      { color: #34d399; font-weight: 600; font-family: 'IBM Plex Mono', monospace; }

/* Section headers */
.section-title {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.75rem; font-weight: 600;
    text-transform: uppercase; letter-spacing: 3px;
    color: #6366f1; border-bottom: 1px solid #1e2330;
    padding-bottom: 0.6rem; margin: 1.5rem 0 1rem;
}

/* Alert box */
.alert-box {
    background: rgba(239,68,68,0.08);
    border: 1px solid rgba(239,68,68,0.3);
    border-left: 4px solid #ef4444;
    border-radius: 8px;
    padding: 1rem 1.2rem;
    margin: 0.5rem 0;
}
.alert-box .tx-id {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.85rem; color: #f87171; font-weight: 600;
}
.alert-box .tx-detail { font-size: 0.8rem; color: #94a3b8; margin-top: 4px; }

/* Feature bar */
.feat-bar-wrap { margin: 6px 0; }
.feat-label {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.75rem; color: #94a3b8;
    display: flex; justify-content: space-between;
    margin-bottom: 3px;
}
.feat-bar-bg {
    background: #1e2330; border-radius: 3px; height: 6px; overflow: hidden;
}
.feat-bar-fill {
    height: 6px; border-radius: 3px;
    background: linear-gradient(90deg, #6366f1, #a5b4fc);
    transition: width 0.5s;
}

/* Dataframe override */
.stDataFrame { border-radius: 10px; overflow: hidden; }

/* Buttons */
.stButton>button {
    background: linear-gradient(135deg, #4f46e5, #6366f1) !important;
    color: white !important; border: none !important;
    border-radius: 8px !important; font-family: 'IBM Plex Mono', monospace !important;
    font-size: 0.8rem !important; letter-spacing: 1px !important;
    padding: 0.5rem 1.2rem !important;
}
.stButton>button:hover { opacity: 0.85; }

/* Tabs */
.stTabs [data-baseweb="tab"] {
    font-family: 'IBM Plex Mono', monospace !important;
    font-size: 0.75rem !important; text-transform: uppercase !important;
    letter-spacing: 2px !important; color: #475569 !important;
}
.stTabs [aria-selected="true"] { color: #a5b4fc !important; }
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────
# GENERATE SYNTHETIC DATASET
# ─────────────────────────────────────────────
@st.cache_data
def generate_synthetic_data(n_samples=5000, fraud_ratio=0.02, seed=42):
    """
    Generates a realistic synthetic credit-card dataset that mimics the
    Kaggle Credit Card Fraud dataset structure (V1-V28 + Amount + Time).
    """
    rng = np.random.default_rng(seed)
    n_fraud = int(n_samples * fraud_ratio)
    n_normal = n_samples - n_fraud

    # --- Normal transactions ---
    normal = pd.DataFrame(
        rng.standard_normal((n_normal, 28)),
        columns=[f"V{i}" for i in range(1, 29)],
    )
    normal["Amount"] = rng.exponential(80, n_normal)
    normal["Time"]   = np.sort(rng.uniform(0, 172800, n_normal))
    normal["Class"]  = 0

    # --- Fraud transactions (shifted distributions) ---
    fraud = pd.DataFrame(
        rng.standard_normal((n_fraud, 28)) * 1.5 + rng.choice([-3, 3], (n_fraud, 28)),
        columns=[f"V{i}" for i in range(1, 29)],
    )
    fraud["Amount"] = rng.exponential(300, n_fraud)  # higher amounts
    fraud["Time"]   = rng.uniform(0, 172800, n_fraud)
    fraud["Class"]  = 1

    df = pd.concat([normal, fraud], ignore_index=True).sample(frac=1, random_state=seed)
    return df


# ─────────────────────────────────────────────
# PREPROCESSING
# ─────────────────────────────────────────────
@st.cache_data
def preprocess(df: pd.DataFrame):
    df = df.copy()
    df.fillna(df.median(numeric_only=True), inplace=True)

    feature_cols = [c for c in df.columns if c != "Class"]
    scaler = StandardScaler()
    df[feature_cols] = scaler.fit_transform(df[feature_cols])

    X = df[feature_cols].values
    y = df["Class"].values if "Class" in df.columns else None

    return df, X, y, feature_cols, scaler


# ─────────────────────────────────────────────
# MODEL TRAINING
# ─────────────────────────────────────────────
@st.cache_resource
def train_models(_X_train, _y_train, feature_cols):
    # SMOTE to handle class imbalance
    smote = SMOTE(random_state=42)
    X_res, y_res = smote.fit_resample(_X_train, _y_train)

    # Random Forest
    rf = RandomForestClassifier(
        n_estimators=150, max_depth=12,
        class_weight="balanced", random_state=42, n_jobs=-1
    )
    rf.fit(X_res, y_res)

    # Isolation Forest (anomaly detection — trained on full data)
    iso = IsolationForest(
        n_estimators=100, contamination=0.02, random_state=42, n_jobs=-1
    )
    iso.fit(_X_train)

    return rf, iso


# ─────────────────────────────────────────────
# SCORING
# ─────────────────────────────────────────────
def compute_fraud_scores(X, rf, iso):
    """Returns a 0-100 risk score by blending RF probability + ISO anomaly score."""
    rf_prob = rf.predict_proba(X)[:, 1]                        # 0-1
    iso_raw = iso.score_samples(X)                              # typically -0.7 to 0
    iso_norm = 1 - (iso_raw - iso_raw.min()) / (iso_raw.max() - iso_raw.min() + 1e-9)

    # Weighted blend: 70 % RF, 30 % ISO
    combined = 0.70 * rf_prob + 0.30 * iso_norm
    return np.clip(combined * 100, 0, 100)


def risk_label(score):
    if score >= 75: return "CRITICAL", "risk-critical"
    if score >= 50: return "HIGH",     "risk-high"
    if score >= 25: return "MEDIUM",   "risk-medium"
    return "LOW", "risk-low"


# ─────────────────────────────────────────────
# FEATURE IMPORTANCE (WHY FLAGGED?)
# ─────────────────────────────────────────────
def explain_transaction(row_values, feature_cols, rf):
    """Returns top-5 feature contributions using mean decrease in impurity."""
    importances = rf.feature_importances_
    contributions = np.abs(row_values) * importances   # feature value × global importance
    top_idx = np.argsort(contributions)[::-1][:5]
    return [(feature_cols[i], contributions[i], row_values[i]) for i in top_idx]


# ─────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────
with st.sidebar:
    st.markdown("### 🛡️ FraudGuard")
    st.markdown("---")

    uploaded = st.file_uploader(
        "Upload CSV dataset", type=["csv"],
        help="Upload a CSV with numeric features + optional 'Class' column (0/1)."
    )

    st.markdown("**Or use synthetic demo data**")
    n_samples = st.slider("Sample size", 500, 10000, 3000, step=500)
    fraud_pct = st.slider("Fraud %", 1, 10, 2)

    st.markdown("---")
    st.markdown("**Risk threshold**")
    threshold = st.slider("Flag if score ≥", 25, 90, 50)

    st.markdown("---")
    st.caption("Models: Random Forest + Isolation Forest")
    st.caption("Imbalance: SMOTE oversampling")


# ─────────────────────────────────────────────
# LOAD DATA
# ─────────────────────────────────────────────
if uploaded:
    raw_df = pd.read_csv(uploaded)
    st.sidebar.success(f"✓ Loaded {len(raw_df):,} rows")
else:
    raw_df = generate_synthetic_data(n_samples, fraud_pct / 100)


# ─────────────────────────────────────────────
# PREPROCESS + TRAIN
# ─────────────────────────────────────────────
with st.spinner("🔄 Preprocessing & training models…"):
    df_proc, X_all, y_all, feature_cols, scaler = preprocess(raw_df)

    if y_all is not None:
        X_train, X_test, y_train, y_test = train_test_split(
            X_all, y_all, test_size=0.2, stratify=y_all, random_state=42
        )
    else:
        X_train, y_train = X_all, None
        X_test, y_test   = X_all, None

    rf_model, iso_model = train_models(X_train, y_train, feature_cols)

    scores = compute_fraud_scores(X_all, rf_model, iso_model)
    rf_preds = rf_model.predict(X_all)


# ─────────────────────────────────────────────
# BUILD RESULT DATAFRAME
# ─────────────────────────────────────────────
result_df = raw_df.copy()
result_df["_RiskScore"] = np.round(scores, 1)
result_df["_RiskLevel"] = [risk_label(s)[0] for s in scores]
result_df["_Flagged"]   = scores >= threshold
result_df["_TxID"]      = [f"TX-{i:05d}" for i in range(len(result_df))]

flagged_df = result_df[result_df["_Flagged"]].copy()
n_flagged  = len(flagged_df)
n_total    = len(result_df)
n_fraud_true = int(result_df["Class"].sum()) if "Class" in result_df.columns else "N/A"

if y_test is not None and len(set(y_test)) > 1:
    test_scores = compute_fraud_scores(X_test, rf_model, iso_model)
    try:
        auc = roc_auc_score(y_test, test_scores / 100)
    except Exception:
        auc = None
else:
    auc = None


# ─────────────────────────────────────────────
# HEADER
# ─────────────────────────────────────────────
st.markdown("""
<div class="fraud-header">
  <h1>🛡️ FRAUDGUARD</h1>
  <p>Real-time transaction risk scoring · Random Forest + Isolation Forest ensemble</p>
</div>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
# KPI ROW
# ─────────────────────────────────────────────
k1, k2, k3, k4, k5 = st.columns(5)

kpis = [
    (k1, "TRANSACTIONS", f"{n_total:,}", "total loaded"),
    (k2, "FLAGGED",      f"{n_flagged:,}", f"≥ {threshold} risk score"),
    (k3, "FLAG RATE",    f"{n_flagged/n_total*100:.1f}%", "of all transactions"),
    (k4, "TRUE FRAUD",   str(n_fraud_true) if isinstance(n_fraud_true, int) else "N/A",
         "labeled in dataset"),
    (k5, "ROC-AUC",      f"{auc:.3f}" if auc else "—", "test set"),
]
for col, label, val, sub in kpis:
    with col:
        st.markdown(f"""
        <div class="metric-card">
            <div class="label">{label}</div>
            <div class="value">{val}</div>
            <div class="sub">{sub}</div>
        </div>""", unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# ─────────────────────────────────────────────
# TABS
# ─────────────────────────────────────────────
tab1, tab2, tab3, tab4 = st.tabs([
    "📋  Transaction Monitor",
    "📊  Analytics",
    "🔍  Why Flagged?",
    "🎯  Model Performance",
])


# ════════════════════════════════════════════
# TAB 1 — TRANSACTION TABLE
# ════════════════════════════════════════════
with tab1:
    st.markdown('<div class="section-title">Transaction Risk Monitor</div>', unsafe_allow_html=True)

    # Filter controls
    c1, c2, c3 = st.columns([1, 1, 2])
    with c1:
        show_only = st.selectbox("Show", ["All", "Flagged only", "Normal only"])
    with c2:
        risk_filter = st.multiselect("Risk level", ["CRITICAL","HIGH","MEDIUM","LOW"],
                                     default=["CRITICAL","HIGH","MEDIUM","LOW"])
    with c3:
        search = st.text_input("Search TX-ID", placeholder="e.g. TX-00042")

    view_df = result_df.copy()
    if show_only == "Flagged only": view_df = view_df[view_df["_Flagged"]]
    if show_only == "Normal only":  view_df = view_df[~view_df["_Flagged"]]
    view_df = view_df[view_df["_RiskLevel"].isin(risk_filter)]
    if search:
        view_df = view_df[view_df["_TxID"].str.contains(search.upper())]

    # Display columns — only include cols that actually exist in the dataframe
    possible_cols = ["_TxID", "Amount", "Time", "_RiskScore", "_RiskLevel", "_Flagged", "Class"]
    display_cols = [c for c in possible_cols if c in view_df.columns]

    rename_map = {
        "_TxID": "TX ID", "Amount": "Amount ($)", "Time": "Time (s)",
        "_RiskScore": "Risk Score", "_RiskLevel": "Risk Level",
        "_Flagged": "🚨 Flagged", "Class": "True Label"
    }
    show = view_df[display_cols].rename(columns=rename_map).head(200)

    def color_risk(val):
        colors = {"CRITICAL": "color:#f87171;font-weight:700",
                  "HIGH":     "color:#fb923c;font-weight:700",
                  "MEDIUM":   "color:#fbbf24",
                  "LOW":      "color:#34d399"}
        return colors.get(val, "")

    # Only apply formatting to columns that exist after rename
    style = show.style.applymap(color_risk, subset=["Risk Level"]) \
                      .background_gradient(subset=["Risk Score"], cmap="RdYlGn_r", vmin=0, vmax=100)
    fmt = {}
    if "Amount ($)" in show.columns: fmt["Amount ($)"] = "{:.2f}"
    if "Risk Score"  in show.columns: fmt["Risk Score"]  = "{:.1f}"
    if "Time (s)"    in show.columns: fmt["Time (s)"]    = "{:.0f}"
    if fmt:
        style = style.format(fmt)

    st.dataframe(style, use_container_width=True, height=420)

    st.caption(f"Showing {len(show):,} of {len(view_df):,} rows")

    # Alerts
    if n_flagged:
        st.markdown('<div class="section-title">🚨 High-Risk Alerts</div>', unsafe_allow_html=True)
        top_alerts = flagged_df.nlargest(5, "_RiskScore")
        for _, row in top_alerts.iterrows():
            label, css = risk_label(row["_RiskScore"])
            amt_str = f"${float(row['Amount']):.2f}" if "Amount" in row.index else "N/A"
            time_str = f"{float(row['Time']):.0f}s" if "Time" in row.index else "N/A"
            true_lbl = f" · True label: {'FRAUD' if row.get('Class')==1 else 'NORMAL'}" \
                       if "Class" in row.index else ""
            st.markdown(f"""
            <div class="alert-box">
                <div class="tx-id">⚠ {row['_TxID']}
                    <span class="{css}" style="margin-left:12px">{label}</span>
                    <span style="float:right;font-size:1.1rem;color:#f87171">
                        {row['_RiskScore']:.1f} / 100
                    </span>
                </div>
                <div class="tx-detail">
                    Amount: {amt_str} · Time: {time_str}{true_lbl}
                </div>
            </div>""", unsafe_allow_html=True)


# ════════════════════════════════════════════
# TAB 2 — ANALYTICS
# ════════════════════════════════════════════
with tab2:
    st.markdown('<div class="section-title">Fraud Analytics Dashboard</div>', unsafe_allow_html=True)

    plt.style.use("dark_background")
    BG  = "#0d0f14"
    BG2 = "#111318"
    ACC = "#6366f1"
    RED = "#f87171"
    GRN = "#34d399"
    TXT = "#94a3b8"

    fig, axes = plt.subplots(2, 3, figsize=(16, 8), facecolor=BG)
    fig.patch.set_facecolor(BG)
    for ax in axes.flat:
        ax.set_facecolor(BG2)
        for spine in ax.spines.values(): spine.set_edgecolor("#1e2330")
        ax.tick_params(colors=TXT, labelsize=8)
        ax.xaxis.label.set_color(TXT)
        ax.yaxis.label.set_color(TXT)
        ax.title.set_color("#e2e8f0")

    # 1. Fraud vs Normal donut
    ax = axes[0, 0]
    if "Class" in result_df.columns:
        counts = result_df["Class"].value_counts()
        wedges, _ = ax.pie(
            [counts.get(0, 0), counts.get(1, 0)],
            colors=[GRN, RED], startangle=90,
            wedgeprops=dict(width=0.55, edgecolor=BG, linewidth=2),
        )
        ax.text(0, 0, f"{counts.get(1,0)/len(result_df)*100:.1f}%\nFRAUD",
                ha="center", va="center", fontsize=11, fontweight="bold",
                color=RED, fontfamily="monospace")
        ax.set_title("Fraud vs Normal", fontsize=10, fontweight="bold")
        ax.legend(["Normal", "Fraud"], loc="lower center", ncol=2,
                  fontsize=8, facecolor=BG2, edgecolor="#1e2330", labelcolor=TXT)
    else:
        ax.text(0.5, 0.5, "No Class\nLabels", ha="center", va="center",
                transform=ax.transAxes, color=TXT, fontsize=12)

    # 2. Risk score distribution
    ax = axes[0, 1]
    ax.hist(scores[result_df.get("Class", pd.Series(0*scores)) == 0],
            bins=40, color=GRN, alpha=0.6, label="Normal", density=True)
    ax.hist(scores[result_df.get("Class", pd.Series(0*scores)) == 1],
            bins=40, color=RED, alpha=0.8, label="Fraud",  density=True)
    ax.axvline(threshold, color="#fbbf24", linestyle="--", linewidth=1.5, label=f"Threshold ({threshold})")
    ax.set_title("Risk Score Distribution", fontsize=10, fontweight="bold")
    ax.set_xlabel("Risk Score (0-100)", fontsize=8)
    ax.legend(fontsize=7, facecolor=BG2, edgecolor="#1e2330", labelcolor=TXT)

    # 3. Risk level breakdown (bar)
    ax = axes[0, 2]
    lvl_counts = result_df["_RiskLevel"].value_counts().reindex(
        ["CRITICAL","HIGH","MEDIUM","LOW"], fill_value=0
    )
    bar_colors = [RED, "#fb923c", "#fbbf24", GRN]
    bars = ax.bar(lvl_counts.index, lvl_counts.values, color=bar_colors, edgecolor=BG, linewidth=1.5)
    for bar, val in zip(bars, lvl_counts.values):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 5,
                str(val), ha="center", fontsize=8, color=TXT)
    ax.set_title("Risk Level Breakdown", fontsize=10, fontweight="bold")
    ax.set_ylabel("Count", fontsize=8)

    # 4. Amount vs Risk Score scatter
    ax = axes[1, 0]
    samp = result_df.sample(min(800, len(result_df)), random_state=1)
    sc = ax.scatter(samp["Amount"], samp["_RiskScore"],
                    c=samp["_RiskScore"], cmap="RdYlGn_r", vmin=0, vmax=100,
                    alpha=0.6, s=12, edgecolors="none")
    ax.axhline(threshold, color="#fbbf24", linestyle="--", linewidth=1, alpha=0.8)
    ax.set_title("Amount vs Risk Score", fontsize=10, fontweight="bold")
    ax.set_xlabel("Transaction Amount ($)", fontsize=8)
    ax.set_ylabel("Risk Score", fontsize=8)
    fig.colorbar(sc, ax=ax, pad=0.02).ax.tick_params(colors=TXT, labelsize=7)

    # 5. Top feature importances
    ax = axes[1, 1]
    imp = pd.Series(rf_model.feature_importances_, index=feature_cols).nlargest(10)
    ax.barh(imp.index[::-1], imp.values[::-1], color=ACC, edgecolor=BG, linewidth=1)
    ax.set_title("Top 10 Feature Importances (RF)", fontsize=10, fontweight="bold")
    ax.set_xlabel("Importance", fontsize=8)

    # 6. Risk score over time
    ax = axes[1, 2]
    time_df = result_df[["Time", "_RiskScore"]].copy()
    time_df = time_df.sort_values("Time")
    # Rolling avg
    roll_n = max(1, len(time_df) // 100)
    ax.plot(time_df["Time"].values, time_df["_RiskScore"].rolling(roll_n).mean().values,
            color=ACC, linewidth=1.2, alpha=0.9)
    ax.fill_between(time_df["Time"].values,
                    time_df["_RiskScore"].rolling(roll_n).mean().fillna(0).values,
                    alpha=0.15, color=ACC)
    ax.axhline(threshold, color="#fbbf24", linestyle="--", linewidth=1, alpha=0.8)
    ax.set_title("Risk Score Over Time", fontsize=10, fontweight="bold")
    ax.set_xlabel("Time (s)", fontsize=8)
    ax.set_ylabel("Avg Risk Score", fontsize=8)

    plt.tight_layout(pad=1.5)
    st.pyplot(fig, use_container_width=True)
    plt.close()


# ════════════════════════════════════════════
# TAB 3 — WHY FLAGGED?
# ════════════════════════════════════════════
with tab3:
    st.markdown('<div class="section-title">Explainability — Why Was This Flagged?</div>',
                unsafe_allow_html=True)

    if n_flagged == 0:
        st.info("No transactions flagged at the current threshold.")
    else:
        tx_options = flagged_df["_TxID"].tolist()
        selected_tx = st.selectbox("Select flagged transaction", tx_options)

        tx_row = flagged_df[flagged_df["_TxID"] == selected_tx].iloc[0]
        tx_idx = result_df[result_df["_TxID"] == selected_tx].index[0]
        tx_X   = X_all[tx_idx]

        score  = tx_row["_RiskScore"]
        label, css = risk_label(score)

        c1, c2, c3 = st.columns(3)
        with c1:
            st.markdown(f"""
            <div class="metric-card">
                <div class="label">Risk Score</div>
                <div class="value" style="color:#f87171">{score:.1f}</div>
                <div class="sub">out of 100</div>
            </div>""", unsafe_allow_html=True)
        with c2:
            st.markdown(f"""
            <div class="metric-card">
                <div class="label">Risk Level</div>
                <div class="value {css}" style="font-size:1.4rem">{label}</div>
                <div class="sub">classification</div>
            </div>""", unsafe_allow_html=True)
        with c3:
            rf_p = rf_model.predict_proba(tx_X.reshape(1, -1))[0][1] * 100
            st.markdown(f"""
            <div class="metric-card">
                <div class="label">RF Fraud Prob</div>
                <div class="value">{rf_p:.1f}%</div>
                <div class="sub">random forest</div>
            </div>""", unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown("#### 🔎 Top Contributing Features")
        st.markdown("These features had the **highest influence** on the fraud prediction for this transaction.")

        explanations = explain_transaction(tx_X, feature_cols, rf_model)
        max_contrib = max(c for _, c, _ in explanations) + 1e-9

        for feat, contrib, val in explanations:
            pct = int(contrib / max_contrib * 100)
            direction = "▲ above avg" if val > 0 else "▼ below avg"
            st.markdown(f"""
            <div class="feat-bar-wrap">
                <div class="feat-label">
                    <span>{feat} <span style="color:#475569;font-size:0.65rem">({direction})</span></span>
                    <span style="color:#a5b4fc">{pct}% influence</span>
                </div>
                <div class="feat-bar-bg">
                    <div class="feat-bar-fill" style="width:{pct}%"></div>
                </div>
            </div>""", unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown("#### 📋 Raw Transaction Values")
        raw_vals = {f: f"{X_all[tx_idx][i]:.4f}" for i, f in enumerate(feature_cols)}
        raw_vals["Amount ($)"] = f"{float(tx_row.get('Amount', 0)):.2f}"
        st.dataframe(
            pd.DataFrame(raw_vals, index=["Value"]).T,
            use_container_width=True, height=250
        )


# ════════════════════════════════════════════
# TAB 4 — MODEL PERFORMANCE
# ════════════════════════════════════════════
with tab4:
    st.markdown('<div class="section-title">Model Evaluation</div>', unsafe_allow_html=True)

    if y_test is not None and len(set(y_test)) > 1:
        rf_test_preds = rf_model.predict(X_test)
        report = classification_report(y_test, rf_test_preds, output_dict=True)

        c1, c2, c3, c4 = st.columns(4)
        metrics = [
            (c1, "Precision\n(Fraud)",  report.get("1", {}).get("precision", 0)),
            (c2, "Recall\n(Fraud)",     report.get("1", {}).get("recall", 0)),
            (c3, "F1-Score\n(Fraud)",   report.get("1", {}).get("f1-score", 0)),
            (c4, "ROC-AUC",             auc if auc else 0),
        ]
        for col, lbl, val in metrics:
            with col:
                st.markdown(f"""
                <div class="metric-card">
                    <div class="label" style="white-space:pre">{lbl}</div>
                    <div class="value">{val:.3f}</div>
                </div>""", unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)

        fig2, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4), facecolor=BG)
        for ax in (ax1, ax2):
            ax.set_facecolor(BG2)
            for spine in ax.spines.values(): spine.set_edgecolor("#1e2330")
            ax.tick_params(colors=TXT, labelsize=9)

        # Confusion matrix
        cm = confusion_matrix(y_test, rf_test_preds)
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax1,
                    linewidths=1, linecolor="#0d0f14",
                    annot_kws={"size": 14, "color": "white", "fontweight": "bold"})
        ax1.set_title("Confusion Matrix (Random Forest)", color="#e2e8f0", fontsize=11, fontweight="bold")
        ax1.set_xlabel("Predicted", color=TXT, fontsize=9)
        ax1.set_ylabel("Actual", color=TXT, fontsize=9)
        ax1.set_xticklabels(["Normal", "Fraud"], color=TXT)
        ax1.set_yticklabels(["Normal", "Fraud"], color=TXT, rotation=0)

        # ROC curve
        from sklearn.metrics import roc_curve
        fpr, tpr, _ = roc_curve(y_test, compute_fraud_scores(X_test, rf_model, iso_model))
        ax2.plot(fpr, tpr, color=ACC, linewidth=2, label=f"Ensemble (AUC={auc:.3f})")
        ax2.plot([0,1],[0,1], color="#334155", linewidth=1, linestyle="--")
        ax2.fill_between(fpr, tpr, alpha=0.1, color=ACC)
        ax2.set_title("ROC Curve", color="#e2e8f0", fontsize=11, fontweight="bold")
        ax2.set_xlabel("False Positive Rate", color=TXT, fontsize=9)
        ax2.set_ylabel("True Positive Rate",  color=TXT, fontsize=9)
        ax2.legend(fontsize=9, facecolor=BG2, edgecolor="#1e2330", labelcolor=TXT)
        ax2.set_facecolor(BG2)

        plt.tight_layout(pad=1.5)
        st.pyplot(fig2, use_container_width=True)
        plt.close()

        st.markdown('<div class="section-title">Full Classification Report</div>', unsafe_allow_html=True)
        rpt_df = pd.DataFrame(report).T
        st.dataframe(rpt_df.style.format("{:.3f}").background_gradient(cmap="Blues", vmin=0, vmax=1),
                     use_container_width=True)
    else:
        st.info("Upload a labeled dataset (with 'Class' column) to see model performance metrics.")

# Footer
st.markdown("""
<div style="text-align:center;padding:2rem 0 1rem;
    font-family:'IBM Plex Mono',monospace;font-size:0.65rem;
    color:#334155;border-top:1px solid #1e2330;margin-top:2rem">
    FRAUDGUARD · Random Forest + Isolation Forest Ensemble · SMOTE Balancing
</div>
""", unsafe_allow_html=True)
