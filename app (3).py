import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier, IsolationForest
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, roc_curve
from imblearn.over_sampling import SMOTE
import warnings

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

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;600&family=IBM+Plex+Sans:wght@300;400;600;700&display=swap');
html, body, [class*="css"] { font-family: 'IBM Plex Sans', sans-serif; background-color: #0d0f14; color: #e2e8f0; }
section[data-testid="stSidebar"] { background: #111318 !important; border-right: 1px solid #1e2330; }
section[data-testid="stSidebar"] * { color: #cbd5e1 !important; }
.fraud-header { background: linear-gradient(135deg,#0f172a,#1e1b4b 50%,#0f172a); border: 1px solid #312e81; border-radius: 12px; padding: 2rem 2.5rem; margin-bottom: 1.5rem; position: relative; overflow: hidden; }
.fraud-header::before { content:''; position:absolute; top:0; left:0; right:0; bottom:0; background: repeating-linear-gradient(45deg,transparent,transparent 10px,rgba(99,102,241,0.03) 10px,rgba(99,102,241,0.03) 20px); }
.fraud-header h1 { font-family:'IBM Plex Mono',monospace; font-size:2rem; font-weight:600; color:#a5b4fc; margin:0; }
.fraud-header p { color:#64748b; margin:0.5rem 0 0; font-size:0.9rem; font-family:'IBM Plex Mono',monospace; }
.metric-card { background:#111318; border:1px solid #1e2330; border-radius:10px; padding:1.2rem 1.5rem; text-align:center; }
.metric-card .label { font-size:0.7rem; text-transform:uppercase; letter-spacing:2px; color:#475569; font-family:'IBM Plex Mono',monospace; }
.metric-card .value { font-size:2rem; font-weight:700; font-family:'IBM Plex Mono',monospace; color:#a5b4fc; line-height:1.1; }
.metric-card .sub { font-size:0.75rem; color:#475569; margin-top:4px; }
.risk-critical { color:#f87171; font-weight:700; font-family:'IBM Plex Mono',monospace; }
.risk-high     { color:#fb923c; font-weight:700; font-family:'IBM Plex Mono',monospace; }
.risk-medium   { color:#fbbf24; font-weight:600; font-family:'IBM Plex Mono',monospace; }
.risk-low      { color:#34d399; font-weight:600; font-family:'IBM Plex Mono',monospace; }
.section-title { font-family:'IBM Plex Mono',monospace; font-size:0.75rem; font-weight:600; text-transform:uppercase; letter-spacing:3px; color:#6366f1; border-bottom:1px solid #1e2330; padding-bottom:0.6rem; margin:1.5rem 0 1rem; }
.alert-box { background:rgba(239,68,68,0.08); border:1px solid rgba(239,68,68,0.3); border-left:4px solid #ef4444; border-radius:8px; padding:1rem 1.2rem; margin:0.5rem 0; }
.alert-box .tx-id { font-family:'IBM Plex Mono',monospace; font-size:0.85rem; color:#f87171; font-weight:600; }
.alert-box .tx-detail { font-size:0.8rem; color:#94a3b8; margin-top:4px; }
.feat-bar-wrap { margin:6px 0; }
.feat-label { font-family:'IBM Plex Mono',monospace; font-size:0.75rem; color:#94a3b8; display:flex; justify-content:space-between; margin-bottom:3px; }
.feat-bar-bg { background:#1e2330; border-radius:3px; height:6px; overflow:hidden; }
.feat-bar-fill { height:6px; border-radius:3px; background:linear-gradient(90deg,#6366f1,#a5b4fc); }
.stDataFrame { border-radius:10px; overflow:hidden; }
.stButton>button { background:linear-gradient(135deg,#4f46e5,#6366f1)!important; color:white!important; border:none!important; border-radius:8px!important; font-family:'IBM Plex Mono',monospace!important; font-size:0.8rem!important; }
.stTabs [data-baseweb="tab"] { font-family:'IBM Plex Mono',monospace!important; font-size:0.75rem!important; text-transform:uppercase!important; letter-spacing:2px!important; color:#475569!important; }
.stTabs [aria-selected="true"] { color:#a5b4fc!important; }
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────
def safe_col(df, col, default=None):
    """Return df[col] if it exists, else a Series of `default`."""
    return df[col] if col in df.columns else pd.Series([default] * len(df), index=df.index)


def risk_label(score):
    if score >= 75: return "CRITICAL", "risk-critical"
    if score >= 50: return "HIGH",     "risk-high"
    if score >= 25: return "MEDIUM",   "risk-medium"
    return "LOW", "risk-low"


def explain_transaction(row_values, feature_cols, rf):
    importances = rf.feature_importances_
    n = min(len(row_values), len(importances), len(feature_cols))
    contributions = np.abs(row_values[:n]) * importances[:n]
    top_idx = np.argsort(contributions)[::-1][:5]
    return [(feature_cols[i], contributions[i], row_values[i]) for i in top_idx]


# ─────────────────────────────────────────────
# SYNTHETIC DATA
# ─────────────────────────────────────────────
@st.cache_data
def generate_synthetic_data(n_samples=5000, fraud_ratio=0.02, seed=42):
    rng = np.random.default_rng(seed)
    n_fraud  = max(1, int(n_samples * fraud_ratio))
    n_normal = n_samples - n_fraud

    normal = pd.DataFrame(rng.standard_normal((n_normal, 28)), columns=[f"V{i}" for i in range(1, 29)])
    normal["Amount"] = rng.exponential(80, n_normal)
    normal["Time"]   = np.sort(rng.uniform(0, 172800, n_normal))
    normal["Class"]  = 0

    fraud = pd.DataFrame(
        rng.standard_normal((n_fraud, 28)) * 1.5 + rng.choice([-3, 3], (n_fraud, 28)),
        columns=[f"V{i}" for i in range(1, 29)],
    )
    fraud["Amount"] = rng.exponential(300, n_fraud)
    fraud["Time"]   = rng.uniform(0, 172800, n_fraud)
    fraud["Class"]  = 1

    return pd.concat([normal, fraud], ignore_index=True).sample(frac=1, random_state=seed)


# ─────────────────────────────────────────────
# PREPROCESSING
# ─────────────────────────────────────────────
@st.cache_data
def preprocess(df: pd.DataFrame):
    df = df.copy()
    # Drop non-numeric columns silently
    df = df.select_dtypes(include=[np.number])
    df.fillna(df.median(numeric_only=True), inplace=True)

    feature_cols = [c for c in df.columns if c != "Class"]
    if not feature_cols:
        st.error("No numeric feature columns found in the uploaded CSV.")
        st.stop()

    scaler = StandardScaler()
    df[feature_cols] = scaler.fit_transform(df[feature_cols])

    X = df[feature_cols].values
    y = df["Class"].values if "Class" in df.columns else None
    return df, X, y, feature_cols, scaler


# ─────────────────────────────────────────────
# MODEL TRAINING
# ─────────────────────────────────────────────
@st.cache_resource
def train_models(_X_train, _y_train, _feature_cols_tuple):
    if _y_train is not None and len(set(_y_train)) > 1:
        try:
            smote = SMOTE(random_state=42)
            X_res, y_res = smote.fit_resample(_X_train, _y_train)
        except Exception:
            X_res, y_res = _X_train, _y_train
        rf = RandomForestClassifier(n_estimators=150, max_depth=12, class_weight="balanced", random_state=42, n_jobs=-1)
        rf.fit(X_res, y_res)
    else:
        # No labels — train unsupervised RF on synthetic binary labels from ISO
        iso_pre = IsolationForest(n_estimators=50, contamination=0.05, random_state=42)
        pseudo_y = (iso_pre.fit_predict(_X_train) == -1).astype(int)
        rf = RandomForestClassifier(n_estimators=100, max_depth=8, random_state=42, n_jobs=-1)
        rf.fit(_X_train, pseudo_y)

    iso = IsolationForest(n_estimators=100, contamination=0.02, random_state=42, n_jobs=-1)
    iso.fit(_X_train)
    return rf, iso


# ─────────────────────────────────────────────
# SCORING
# ─────────────────────────────────────────────
def compute_fraud_scores(X, rf, iso):
    rf_prob  = rf.predict_proba(X)[:, 1]
    iso_raw  = iso.score_samples(X)
    iso_norm = 1 - (iso_raw - iso_raw.min()) / (iso_raw.max() - iso_raw.min() + 1e-9)
    combined = 0.70 * rf_prob + 0.30 * iso_norm
    return np.clip(combined * 100, 0, 100)


# ─────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────
with st.sidebar:
    st.markdown("### 🛡️ FraudGuard")
    st.markdown("---")
    uploaded  = st.file_uploader("Upload CSV dataset", type=["csv"],
                                  help="Any numeric CSV. Optional 'Class' column (0=normal,1=fraud).")
    st.markdown("**Or use synthetic demo data**")
    n_samples = st.slider("Sample size", 500, 10000, 3000, step=500)
    fraud_pct = st.slider("Fraud %", 1, 10, 2)
    st.markdown("---")
    st.markdown("**Risk threshold**")
    threshold = st.slider("Flag if score ≥", 10, 90, 50)
    st.markdown("---")
    st.caption("Models: Random Forest + Isolation Forest")
    st.caption("Imbalance: SMOTE oversampling")


# ─────────────────────────────────────────────
# LOAD DATA
# ─────────────────────────────────────────────
if uploaded:
    try:
        raw_df = pd.read_csv(uploaded)
        st.sidebar.success(f"✓ Loaded {len(raw_df):,} rows")
    except Exception as e:
        st.sidebar.error(f"Could not read CSV: {e}")
        raw_df = generate_synthetic_data(n_samples, fraud_pct / 100)
else:
    raw_df = generate_synthetic_data(n_samples, fraud_pct / 100)

HAS_AMOUNT = "Amount" in raw_df.columns
HAS_TIME   = "Time"   in raw_df.columns
HAS_CLASS  = "Class"  in raw_df.columns


# ─────────────────────────────────────────────
# PREPROCESS + TRAIN
# ─────────────────────────────────────────────
with st.spinner("🔄 Preprocessing & training models…"):
    df_proc, X_all, y_all, feature_cols, scaler = preprocess(raw_df)

    if y_all is not None and len(set(y_all)) > 1:
        X_train, X_test, y_train, y_test = train_test_split(
            X_all, y_all, test_size=0.2, stratify=y_all, random_state=42)
    else:
        X_train, X_test = X_all, X_all
        y_train = y_all
        y_test  = y_all

    rf_model, iso_model = train_models(X_train, y_train, tuple(feature_cols))
    scores   = compute_fraud_scores(X_all, rf_model, iso_model)


# ─────────────────────────────────────────────
# BUILD RESULT DATAFRAME
# ─────────────────────────────────────────────
result_df = raw_df.copy()
result_df["_RiskScore"] = np.round(scores, 1)
result_df["_RiskLevel"] = [risk_label(s)[0] for s in scores]
result_df["_Flagged"]   = scores >= threshold
result_df["_TxID"]      = [f"TX-{i:05d}" for i in range(len(result_df))]

flagged_df   = result_df[result_df["_Flagged"]].copy()
n_flagged    = len(flagged_df)
n_total      = len(result_df)
n_fraud_true = int(result_df["Class"].sum()) if HAS_CLASS else "N/A"

auc = None
if y_test is not None and len(set(y_test)) > 1:
    try:
        test_scores = compute_fraud_scores(X_test, rf_model, iso_model)
        auc = roc_auc_score(y_test, test_scores / 100)
    except Exception:
        auc = None


# ─────────────────────────────────────────────
# HEADER + KPIs
# ─────────────────────────────────────────────
st.markdown("""
<div class="fraud-header">
  <h1>🛡️ FRAUDGUARD</h1>
  <p>Real-time transaction risk scoring · Random Forest + Isolation Forest ensemble</p>
</div>
""", unsafe_allow_html=True)

k1, k2, k3, k4, k5 = st.columns(5)
for col, label, val, sub in [
    (k1, "TRANSACTIONS", f"{n_total:,}",                    "total loaded"),
    (k2, "FLAGGED",      f"{n_flagged:,}",                  f"≥ {threshold} risk score"),
    (k3, "FLAG RATE",    f"{n_flagged/n_total*100:.1f}%",   "of all transactions"),
    (k4, "TRUE FRAUD",   str(n_fraud_true),                  "labeled in dataset"),
    (k5, "ROC-AUC",      f"{auc:.3f}" if auc else "—",      "test set"),
]:
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
    if risk_filter:
        view_df = view_df[view_df["_RiskLevel"].isin(risk_filter)]
    if search:
        view_df = view_df[view_df["_TxID"].str.contains(search.upper(), na=False)]

    # Build display table from only existing columns
    possible = ["_TxID", "Amount", "Time", "_RiskScore", "_RiskLevel", "_Flagged", "Class"]
    display_cols = [c for c in possible if c in view_df.columns]
    rename_map   = {"_TxID":"TX ID","Amount":"Amount ($)","Time":"Time (s)",
                    "_RiskScore":"Risk Score","_RiskLevel":"Risk Level",
                    "_Flagged":"🚨 Flagged","Class":"True Label"}
    show = view_df[display_cols].rename(columns=rename_map).head(200)

    def color_risk(val):
        return {"CRITICAL":"color:#f87171;font-weight:700",
                "HIGH":    "color:#fb923c;font-weight:700",
                "MEDIUM":  "color:#fbbf24",
                "LOW":     "color:#34d399"}.get(val, "")

    style = show.style
    if "Risk Level" in show.columns:
        style = style.applymap(color_risk, subset=["Risk Level"])
    if "Risk Score" in show.columns:
        style = style.background_gradient(subset=["Risk Score"], cmap="RdYlGn_r", vmin=0, vmax=100)
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
            lbl, css = risk_label(row["_RiskScore"])
            amt_str  = f"${float(row['Amount']):.2f}" if HAS_AMOUNT else "N/A"
            time_str = f"{float(row['Time']):.0f}s"   if HAS_TIME   else "N/A"
            true_str = f" · True label: {'FRAUD' if row['Class']==1 else 'NORMAL'}" if HAS_CLASS else ""
            st.markdown(f"""
            <div class="alert-box">
                <div class="tx-id">⚠ {row['_TxID']}
                    <span class="{css}" style="margin-left:12px">{lbl}</span>
                    <span style="float:right;font-size:1.1rem;color:#f87171">{row['_RiskScore']:.1f} / 100</span>
                </div>
                <div class="tx-detail">Amount: {amt_str} · Time: {time_str}{true_str}</div>
            </div>""", unsafe_allow_html=True)


# ════════════════════════════════════════════
# TAB 2 — ANALYTICS
# ════════════════════════════════════════════
with tab2:
    st.markdown('<div class="section-title">Fraud Analytics Dashboard</div>', unsafe_allow_html=True)

    BG  = "#0d0f14"; BG2 = "#111318"; ACC = "#6366f1"
    RED = "#f87171"; GRN = "#34d399"; TXT = "#94a3b8"

    plt.style.use("dark_background")
    fig, axes = plt.subplots(2, 3, figsize=(16, 8), facecolor=BG)
    fig.patch.set_facecolor(BG)
    for ax in axes.flat:
        ax.set_facecolor(BG2)
        for sp in ax.spines.values(): sp.set_edgecolor("#1e2330")
        ax.tick_params(colors=TXT, labelsize=8)
        ax.xaxis.label.set_color(TXT)
        ax.yaxis.label.set_color(TXT)
        ax.title.set_color("#e2e8f0")

    # 1. Fraud vs Normal (donut if Class exists, else risk level pie)
    ax = axes[0, 0]
    if HAS_CLASS:
        counts = result_df["Class"].value_counts()
        ax.pie([counts.get(0, 0), counts.get(1, 0)], colors=[GRN, RED], startangle=90,
               wedgeprops=dict(width=0.55, edgecolor=BG, linewidth=2))
        pct = counts.get(1, 0) / max(len(result_df), 1) * 100
        ax.text(0, 0, f"{pct:.1f}%\nFRAUD", ha="center", va="center",
                fontsize=11, fontweight="bold", color=RED, fontfamily="monospace")
        ax.set_title("Fraud vs Normal", fontsize=10, fontweight="bold")
        ax.legend(["Normal", "Fraud"], loc="lower center", ncol=2,
                  fontsize=8, facecolor=BG2, edgecolor="#1e2330", labelcolor=TXT)
    else:
        lvl_counts = result_df["_RiskLevel"].value_counts()
        lvl_colors = {"CRITICAL": RED, "HIGH": "#fb923c", "MEDIUM": "#fbbf24", "LOW": GRN}
        colors = [lvl_colors.get(l, ACC) for l in lvl_counts.index]
        ax.pie(lvl_counts.values, labels=lvl_counts.index, colors=colors, startangle=90,
               wedgeprops=dict(edgecolor=BG, linewidth=1.5),
               textprops={"color": TXT, "fontsize": 8})
        ax.set_title("Risk Level Distribution", fontsize=10, fontweight="bold")

    # 2. Risk score distribution
    ax = axes[0, 1]
    if HAS_CLASS:
        class_col = result_df["Class"].values
        ax.hist(scores[class_col == 0], bins=40, color=GRN, alpha=0.6, label="Normal", density=True)
        ax.hist(scores[class_col == 1], bins=40, color=RED, alpha=0.8, label="Fraud",  density=True)
    else:
        ax.hist(scores[result_df["_Flagged"]], bins=30, color=RED, alpha=0.7, label="Flagged", density=True)
        ax.hist(scores[~result_df["_Flagged"]], bins=30, color=GRN, alpha=0.5, label="Normal", density=True)
    ax.axvline(threshold, color="#fbbf24", linestyle="--", linewidth=1.5, label=f"Threshold ({threshold})")
    ax.set_title("Risk Score Distribution", fontsize=10, fontweight="bold")
    ax.set_xlabel("Risk Score (0-100)", fontsize=8)
    ax.legend(fontsize=7, facecolor=BG2, edgecolor="#1e2330", labelcolor=TXT)

    # 3. Risk level breakdown
    ax = axes[0, 2]
    lvl_counts = result_df["_RiskLevel"].value_counts().reindex(
        ["CRITICAL","HIGH","MEDIUM","LOW"], fill_value=0)
    bar_colors = [RED, "#fb923c", "#fbbf24", GRN]
    bars = ax.bar(lvl_counts.index, lvl_counts.values, color=bar_colors, edgecolor=BG, linewidth=1.5)
    for bar, val in zip(bars, lvl_counts.values):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 2,
                str(val), ha="center", fontsize=8, color=TXT)
    ax.set_title("Risk Level Breakdown", fontsize=10, fontweight="bold")
    ax.set_ylabel("Count", fontsize=8)

    # 4. Amount vs Risk Score (or index vs score if no Amount)
    ax = axes[1, 0]
    samp = result_df.sample(min(800, len(result_df)), random_state=1)
    if HAS_AMOUNT:
        x_vals = samp["Amount"].values
        ax.set_xlabel("Transaction Amount ($)", fontsize=8)
        ax.set_title("Amount vs Risk Score", fontsize=10, fontweight="bold")
    else:
        x_vals = np.arange(len(samp))
        ax.set_xlabel("Transaction Index", fontsize=8)
        ax.set_title("Index vs Risk Score", fontsize=10, fontweight="bold")
    sc = ax.scatter(x_vals, samp["_RiskScore"].values,
                    c=samp["_RiskScore"].values, cmap="RdYlGn_r", vmin=0, vmax=100,
                    alpha=0.6, s=12, edgecolors="none")
    ax.axhline(threshold, color="#fbbf24", linestyle="--", linewidth=1, alpha=0.8)
    ax.set_ylabel("Risk Score", fontsize=8)
    fig.colorbar(sc, ax=ax, pad=0.02).ax.tick_params(colors=TXT, labelsize=7)

    # 5. Top feature importances
    ax = axes[1, 1]
    imp = pd.Series(rf_model.feature_importances_, index=feature_cols).nlargest(10)
    ax.barh(imp.index[::-1], imp.values[::-1], color=ACC, edgecolor=BG, linewidth=1)
    ax.set_title("Top 10 Feature Importances (RF)", fontsize=10, fontweight="bold")
    ax.set_xlabel("Importance", fontsize=8)

    # 6. Risk score over Time (or over index if no Time)
    ax = axes[1, 2]
    if HAS_TIME:
        time_df = result_df[["Time", "_RiskScore"]].copy().sort_values("Time")
        x_time  = time_df["Time"].values
        y_time  = time_df["_RiskScore"].values
        ax.set_xlabel("Time (s)", fontsize=8)
        ax.set_title("Risk Score Over Time", fontsize=10, fontweight="bold")
    else:
        time_df = result_df[["_RiskScore"]].copy().reset_index(drop=True)
        x_time  = np.arange(len(time_df))
        y_time  = time_df["_RiskScore"].values
        ax.set_xlabel("Transaction Index", fontsize=8)
        ax.set_title("Risk Score Over Transactions", fontsize=10, fontweight="bold")
    roll_n = max(1, len(x_time) // 100)
    y_smooth = pd.Series(y_time).rolling(roll_n).mean().fillna(method="bfill").values
    ax.plot(x_time, y_smooth, color=ACC, linewidth=1.2, alpha=0.9)
    ax.fill_between(x_time, y_smooth, alpha=0.15, color=ACC)
    ax.axhline(threshold, color="#fbbf24", linestyle="--", linewidth=1, alpha=0.8)
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
        st.info("No transactions flagged at the current threshold. Try lowering the slider in the sidebar.")
    else:
        tx_options = flagged_df["_TxID"].tolist()
        selected_tx = st.selectbox("Select flagged transaction", tx_options)

        match = result_df[result_df["_TxID"] == selected_tx]
        if match.empty:
            st.warning("Transaction not found.")
        else:
            tx_row = match.iloc[0]
            tx_idx = match.index[0]

            # Guard: tx_idx must be valid for X_all
            if tx_idx >= len(X_all):
                st.error("Index out of range for feature array. Please reload.")
            else:
                tx_X  = X_all[tx_idx]
                score = tx_row["_RiskScore"]
                lbl, css = risk_label(score)

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
                        <div class="value {css}" style="font-size:1.4rem">{lbl}</div>
                        <div class="sub">classification</div>
                    </div>""", unsafe_allow_html=True)
                with c3:
                    try:
                        rf_p = rf_model.predict_proba(tx_X.reshape(1, -1))[0][1] * 100
                    except Exception:
                        rf_p = 0.0
                    st.markdown(f"""
                    <div class="metric-card">
                        <div class="label">RF Fraud Prob</div>
                        <div class="value">{rf_p:.1f}%</div>
                        <div class="sub">random forest</div>
                    </div>""", unsafe_allow_html=True)

                # Extra info row
                extra = []
                if HAS_AMOUNT and "Amount" in tx_row.index:
                    extra.append(f"Amount: ${float(tx_row['Amount']):.2f}")
                if HAS_TIME and "Time" in tx_row.index:
                    extra.append(f"Time: {float(tx_row['Time']):.0f}s")
                if HAS_CLASS and "Class" in tx_row.index:
                    extra.append(f"True Label: {'FRAUD' if tx_row['Class']==1 else 'NORMAL'}")
                if extra:
                    st.markdown(f"<p style='font-family:IBM Plex Mono,monospace;font-size:.75rem;color:#64748b;margin-top:8px'>"
                                + " · ".join(extra) + "</p>", unsafe_allow_html=True)

                st.markdown("<br>", unsafe_allow_html=True)
                st.markdown("#### 🔎 Top Contributing Features")
                st.markdown("These features had the **highest influence** on the fraud prediction for this transaction.")

                try:
                    explanations = explain_transaction(tx_X, feature_cols, rf_model)
                    max_contrib  = max(c for _, c, _ in explanations) + 1e-9
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
                except Exception as e:
                    st.warning(f"Could not compute feature contributions: {e}")

                st.markdown("<br>", unsafe_allow_html=True)
                st.markdown("#### 📋 Raw Feature Values (scaled)")
                try:
                    raw_vals = {fc: f"{tx_X[i]:.4f}" for i, fc in enumerate(feature_cols)}
                    st.dataframe(pd.DataFrame(raw_vals, index=["Value"]).T,
                                 use_container_width=True, height=250)
                except Exception as e:
                    st.warning(f"Could not display raw values: {e}")


# ════════════════════════════════════════════
# TAB 4 — MODEL PERFORMANCE
# ════════════════════════════════════════════
with tab4:
    st.markdown('<div class="section-title">Model Evaluation</div>', unsafe_allow_html=True)

    BG  = "#0d0f14"; BG2 = "#111318"; ACC = "#6366f1"; TXT = "#94a3b8"

    if not HAS_CLASS:
        st.info("No 'Class' column found in your dataset. Upload a labeled CSV (0=normal, 1=fraud) to see full model metrics.")
        st.markdown("**Isolation Forest anomaly scores** (unsupervised) are still used for risk scoring.")
        # Show feature importance only
        st.markdown('<div class="section-title">Feature Importances</div>', unsafe_allow_html=True)
        imp_df = pd.Series(rf_model.feature_importances_, index=feature_cols).nlargest(15).reset_index()
        imp_df.columns = ["Feature", "Importance"]
        st.dataframe(imp_df.style.background_gradient(subset=["Importance"], cmap="Blues"),
                     use_container_width=True)
    else:
        # y_test guaranteed to exist and have both classes here
        has_both_classes = y_test is not None and len(set(y_test)) > 1
        if not has_both_classes:
            st.warning("Test set contains only one class — metrics not available. Try increasing the Fraud % slider.")
        else:
            rf_test_preds = rf_model.predict(X_test)
            report = classification_report(y_test, rf_test_preds, output_dict=True)
            test_scores = compute_fraud_scores(X_test, rf_model, iso_model)

            try:
                auc_val = roc_auc_score(y_test, test_scores / 100)
            except Exception:
                auc_val = 0.0

            c1, c2, c3, c4 = st.columns(4)
            for col, lbl, val in [
                (c1, "Precision\n(Fraud)", report.get("1", report.get(1, {})).get("precision", 0)),
                (c2, "Recall\n(Fraud)",    report.get("1", report.get(1, {})).get("recall", 0)),
                (c3, "F1-Score\n(Fraud)",  report.get("1", report.get(1, {})).get("f1-score", 0)),
                (c4, "ROC-AUC",            auc_val),
            ]:
                with col:
                    st.markdown(f"""
                    <div class="metric-card">
                        <div class="label" style="white-space:pre">{lbl}</div>
                        <div class="value">{val:.3f}</div>
                    </div>""", unsafe_allow_html=True)

            st.markdown("<br>", unsafe_allow_html=True)

            plt.style.use("dark_background")
            fig2, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4), facecolor=BG)
            for ax in (ax1, ax2):
                ax.set_facecolor(BG2)
                for sp in ax.spines.values(): sp.set_edgecolor("#1e2330")
                ax.tick_params(colors=TXT, labelsize=9)

            # Confusion matrix
            cm = confusion_matrix(y_test, rf_test_preds)
            sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax1,
                        linewidths=1, linecolor="#0d0f14",
                        annot_kws={"size": 14, "color": "white", "fontweight": "bold"})
            ax1.set_title("Confusion Matrix (Random Forest)", color="#e2e8f0", fontsize=11, fontweight="bold")
            ax1.set_xlabel("Predicted", color=TXT, fontsize=9)
            ax1.set_ylabel("Actual",    color=TXT, fontsize=9)
            n_classes = cm.shape[0]
            xticklabels = ["Normal", "Fraud"] if n_classes == 2 else [str(i) for i in range(n_classes)]
            ax1.set_xticklabels(xticklabels, color=TXT)
            ax1.set_yticklabels(xticklabels, color=TXT, rotation=0)

            # ROC curve
            try:
                fpr, tpr, _ = roc_curve(y_test, test_scores / 100)
                ax2.plot(fpr, tpr, color=ACC, linewidth=2, label=f"Ensemble (AUC={auc_val:.3f})")
                ax2.fill_between(fpr, tpr, alpha=0.1, color=ACC)
            except Exception:
                pass
            ax2.plot([0, 1], [0, 1], color="#334155", linewidth=1, linestyle="--")
            ax2.set_title("ROC Curve", color="#e2e8f0", fontsize=11, fontweight="bold")
            ax2.set_xlabel("False Positive Rate", color=TXT, fontsize=9)
            ax2.set_ylabel("True Positive Rate",  color=TXT, fontsize=9)
            ax2.legend(fontsize=9, facecolor=BG2, edgecolor="#1e2330", labelcolor=TXT)
            ax2.set_facecolor(BG2)

            plt.tight_layout(pad=1.5)
            st.pyplot(fig2, use_container_width=True)
            plt.close()

            st.markdown('<div class="section-title">Full Classification Report</div>', unsafe_allow_html=True)
            try:
                rpt_df = pd.DataFrame(report).T
                # Only format numeric columns
                num_cols = rpt_df.select_dtypes(include=[np.number]).columns
                st.dataframe(
                    rpt_df.style.format("{:.3f}", subset=num_cols)
                                .background_gradient(cmap="Blues", subset=num_cols, vmin=0, vmax=1),
                    use_container_width=True
                )
            except Exception as e:
                st.warning(f"Could not render classification report: {e}")
                st.write(report)

# Footer
st.markdown("""
<div style="text-align:center;padding:2rem 0 1rem;font-family:'IBM Plex Mono',monospace;font-size:0.65rem;color:#334155;border-top:1px solid #1e2330;margin-top:2rem">
    FRAUDGUARD · Random Forest + Isolation Forest Ensemble · SMOTE Balancing
</div>
""", unsafe_allow_html=True)
