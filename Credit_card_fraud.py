import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, roc_curve
from sklearn.metrics import precision_recall_curve, accuracy_score, f1_score
import warnings
warnings.filterwarnings('ignore')

# Page configuration
st.set_page_config(
    page_title="Credit Card Fraud Detection",
    page_icon="💳",
    layout="wide",
    initial_sidebar_state="collapsed"   # collapsed by default on mobile
)

# ── Responsive CSS ──────────────────────────────────────────────────────────
st.markdown("""
<style>
/* ── Google Font ── */
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&display=swap');

html, body, [class*="css"] {
    font-family: 'DM Sans', sans-serif;
}

/* ── Global layout ── */
.main .block-container {
    padding: clamp(0.75rem, 3vw, 2.5rem) clamp(0.5rem, 4vw, 3rem);
    max-width: 1400px;
}

/* ── Hero title ── */
h1 {
    font-size: clamp(1.4rem, 4vw, 2.4rem) !important;
    color: #1a56db !important;
    font-weight: 700 !important;
    letter-spacing: -0.02em;
    line-height: 1.2;
}

h2 { color: #1e7e34 !important; font-size: clamp(1.1rem, 3vw, 1.6rem) !important; }
h3 { font-size: clamp(1rem, 2.5vw, 1.25rem) !important; }

/* ── Metric cards ── */
[data-testid="stMetric"] {
    background: #ffffff;
    border: 1px solid #e5e9f0;
    border-radius: 14px;
    padding: clamp(0.6rem, 2vw, 1.1rem) clamp(0.75rem, 2.5vw, 1.4rem);
    box-shadow: 0 2px 8px rgba(0,0,0,0.05);
    min-width: 0;           /* allow flex shrink */
}

[data-testid="stMetricLabel"]  { font-size: clamp(0.65rem, 1.8vw, 0.82rem) !important; }
[data-testid="stMetricValue"]  { font-size: clamp(1rem,   3vw,  1.6rem)  !important; font-weight: 700; }
[data-testid="stMetricDelta"]  { font-size: clamp(0.6rem, 1.5vw, 0.75rem) !important; }

/* ── Sidebar ── */
[data-testid="stSidebar"] {
    background: #0f172a !important;
    min-width: 220px;
    max-width: 320px;
}
[data-testid="stSidebar"] * { color: #e2e8f0 !important; }
[data-testid="stSidebar"] .stSlider > label,
[data-testid="stSidebar"] .stButton > button { color: #e2e8f0 !important; }

/* ── Primary buttons ── */
.stButton > button[kind="primary"] {
    background: linear-gradient(135deg, #1a56db 0%, #1e7e34 100%);
    color: #fff !important;
    border: none;
    border-radius: 10px;
    font-weight: 600;
    font-size: clamp(0.8rem, 2vw, 0.95rem);
    padding: 0.55em 1.4em;
    transition: opacity 0.2s, transform 0.15s;
    width: 100%;
}
.stButton > button[kind="primary"]:hover { opacity: 0.88; transform: translateY(-1px); }

/* ── Tabs ── */
[data-testid="stTabs"] [role="tablist"] {
    gap: clamp(2px, 1vw, 10px);
    flex-wrap: wrap;        /* tabs wrap on small screens */
}
[data-testid="stTabs"] [role="tab"] {
    font-size: clamp(0.72rem, 1.8vw, 0.9rem);
    padding: 0.4em clamp(0.5em, 2vw, 1em);
    white-space: nowrap;
}

/* ── Plotly charts: let them shrink ── */
[data-testid="stPlotlyChart"] > div { width: 100% !important; }
.js-plotly-plot .plotly { width: 100% !important; }

/* ── Number inputs & selects ── */
.stNumberInput input,
.stSelectbox select,
.stSlider { font-size: clamp(0.8rem, 2vw, 0.95rem) !important; }

/* ── Responsive column stacking helper ──
   Streamlit columns don't stack natively on mobile; we compensate with
   tighter padding so each panel stays usable at ~360 px. */
@media (max-width: 640px) {
    .main .block-container { padding: 0.6rem 0.4rem; }

    /* Stack the 4-metric row */
    [data-testid="column"] { min-width: 45% !important; }

    /* Shrink sidebar toggle button area */
    [data-testid="collapsedControl"] { top: 0.5rem !important; }

    h1 { font-size: 1.25rem !important; }
}

@media (max-width: 400px) {
    [data-testid="column"] { min-width: 100% !important; }
}

/* ── Alert / info boxes ── */
.stAlert {
    border-radius: 10px;
    font-size: clamp(0.78rem, 2vw, 0.9rem);
}

/* ── Divider ── */
hr { border-color: #e5e9f0; margin: 0.8rem 0; }

/* ── Spinner text ── */
.stSpinner > div { font-size: clamp(0.8rem, 2vw, 0.95rem); }
</style>
""", unsafe_allow_html=True)

# ── Title ─────────────────────────────────────────────────────────────────
st.title("💳 Credit Card Fraud Detection System")
st.markdown("---")

# ── Sidebar ───────────────────────────────────────────────────────────────
st.sidebar.header("⚙️ Configuration")
n_samples   = st.sidebar.slider("Number of Transactions", 1000, 50000, 10000, 1000)
test_size   = st.sidebar.slider("Test Size (%)", 10, 40, 30, 5) / 100
fraud_rate  = st.sidebar.slider("Fraud Rate (%)",  5, 30, 15, 5) / 100

# ── Data generation ────────────────────────────────────────────────────────
@st.cache_data
def generate_fraud_data(n_fraud):
    data = np.zeros((n_fraud, 20))
    data[:, 0]  = np.random.exponential(800, n_fraud) + 200
    data[:, 1]  = np.random.choice([0,1,2,3,4,5,22,23], n_fraud)
    data[:, 2]  = np.random.randint(0, 7, n_fraud)
    data[:, 3]  = np.random.exponential(500, n_fraud) + 100
    data[:, 4]  = np.random.exponential(300, n_fraud) + 50
    data[:, 5]  = np.random.uniform(2.5, 10, n_fraud)
    data[:, 6]  = np.random.binomial(1, 0.2, n_fraud)
    data[:, 7]  = np.random.binomial(1, 0.3, n_fraud)
    data[:, 8]  = np.random.binomial(1, 0.2, n_fraud)
    data[:, 9]  = np.random.binomial(1, 0.6, n_fraud)
    data[:, 10] = np.random.poisson(3, n_fraud) + 1
    data[:, 11] = np.random.poisson(8, n_fraud) + 3
    data[:, 12] = data[:, 0] * np.random.uniform(0.8, 1.2, n_fraud)
    data[:, 13] = data[:, 12] * np.random.uniform(0.5, 1.5, n_fraud)
    data[:, 14] = np.random.randint(0, 10, n_fraud)
    data[:, 15] = np.random.binomial(1, 0.3, n_fraud)
    data[:, 16] = np.random.binomial(1, 0.5, n_fraud)
    data[:, 17] = np.random.binomial(1, 0.4, n_fraud)
    data[:, 18] = np.random.binomial(1, 0.7, n_fraud)
    data[:, 19] = (data[:, 2] >= 5).astype(int)
    return data

@st.cache_data
def generate_normal_data(n_normal):
    data = np.zeros((n_normal, 20))
    data[:, 0]  = np.random.exponential(100, n_normal) + 10
    data[:, 1]  = np.random.choice(range(7, 23), n_normal)
    data[:, 2]  = np.random.randint(0, 7, n_normal)
    data[:, 3]  = np.random.exponential(20, n_normal)
    data[:, 4]  = np.random.exponential(15, n_normal)
    data[:, 5]  = np.random.uniform(0.5, 2, n_normal)
    data[:, 6]  = np.random.binomial(1, 0.7, n_normal)
    data[:, 7]  = np.random.binomial(1, 0.8, n_normal)
    data[:, 8]  = np.random.binomial(1, 0.7, n_normal)
    data[:, 9]  = np.random.binomial(1, 0.3, n_normal)
    data[:, 10] = np.random.poisson(1, n_normal)
    data[:, 11] = np.random.poisson(3, n_normal)
    data[:, 12] = data[:, 0] * np.random.uniform(0.9, 1.1, n_normal)
    data[:, 13] = data[:, 12] * np.random.uniform(0.2, 0.5, n_normal)
    data[:, 14] = np.random.randint(0, 10, n_normal)
    data[:, 15] = np.random.binomial(1, 0.8, n_normal)
    data[:, 16] = np.random.binomial(1, 0.1, n_normal)
    data[:, 17] = np.random.binomial(1, 0.05, n_normal)
    data[:, 18] = np.random.binomial(1, 0.2, n_normal)
    data[:, 19] = (data[:, 2] >= 5).astype(int)
    return data

@st.cache_data
def generate_dataset(n_samples, fraud_rate):
    np.random.seed(42)
    n_fraud  = int(n_samples * fraud_rate)
    n_normal = n_samples - n_fraud

    fraud_data  = generate_fraud_data(n_fraud)
    normal_data = generate_normal_data(n_normal)

    data   = np.vstack([fraud_data, normal_data])
    labels = np.hstack([np.ones(n_fraud), np.zeros(n_normal)])

    indices = np.random.permutation(n_samples)
    data    = data[indices]
    labels  = labels[indices]

    feature_names = [
        'amount', 'time_hour', 'day_of_week', 'distance_from_home',
        'distance_from_last', 'ratio_to_median', 'repeat_retailer',
        'used_chip', 'used_pin', 'online_order', 'velocity_1h',
        'velocity_24h', 'avg_last_10', 'std_last_10', 'merchant_category',
        'card_present', 'international', 'high_risk_country',
        'unusual_time', 'weekend'
    ]
    df = pd.DataFrame(data, columns=feature_names)
    df['is_fraud'] = labels
    return df

# ── Responsive chart helper ────────────────────────────────────────────────
CHART_H_MOBILE = 320
CHART_H_DESKTOP = 420

def chart_height():
    """Return a height that is reasonable on both mobile and desktop."""
    return CHART_H_MOBILE   # Streamlit can't detect viewport; keep compact

# ── Generate Data button ───────────────────────────────────────────────────
if st.sidebar.button("🔄 Generate Data", type="primary"):
    st.session_state['data_generated'] = True
    st.session_state['df'] = generate_dataset(n_samples, fraud_rate)

# ── Main app ───────────────────────────────────────────────────────────────
if 'data_generated' in st.session_state and st.session_state['data_generated']:
    df = st.session_state['df']

    # ── KPI metrics ──
    # Use 2 columns on narrow screens via gap trick; Streamlit handles 4 cols fine on wide
    col1, col2, col3, col4 = st.columns(4, gap="small")
    with col1:
        st.metric("📊 Total Transactions", f"{len(df):,}")
    with col2:
        fraud_count = int(df['is_fraud'].sum())
        st.metric("🚨 Fraud Cases", f"{fraud_count:,}",
                  f"{fraud_count/len(df)*100:.1f}%")
    with col3:
        normal_count = len(df) - fraud_count
        st.metric("✅ Normal Cases", f"{normal_count:,}",
                  f"{normal_count/len(df)*100:.1f}%")
    with col4:
        avg_amount = df['amount'].mean()
        st.metric("💰 Avg Amount", f"${avg_amount:.2f}")

    st.markdown("---")

    # ── Tabs ──────────────────────────────────────────────────────────────
    tab1, tab2, tab3, tab4 = st.tabs([
        "📊 Data Exploration",
        "🤖 Model Training",
        "📈 Results",
        "🔍 Test Prediction"
    ])

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # TAB 1 — Data Exploration
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    with tab1:
        st.header("Data Exploration")

        col1, col2 = st.columns([1, 1], gap="medium")

        with col1:
            fraud_dist = df['is_fraud'].value_counts()
            fig = px.pie(
                values=fraud_dist.values,
                names=['Normal', 'Fraud'],
                title='Transaction Distribution',
                color_discrete_sequence=['#2ecc71', '#e74c3c'],
                hole=0.35
            )
            fig.update_layout(
                height=chart_height(),
                margin=dict(l=10, r=10, t=40, b=10),
                legend=dict(orientation="h", yanchor="bottom", y=-0.15),
                font=dict(size=12)
            )
            st.plotly_chart(fig, use_container_width=True)

            fig = px.box(
                df, x='is_fraud', y='amount',
                color='is_fraud',
                labels={'is_fraud': 'Transaction Type', 'amount': 'Amount ($)'},
                title='Amount Distribution by Type',
                color_discrete_map={0: '#2ecc71', 1: '#e74c3c'}
            )
            fig.update_xaxes(ticktext=['Normal', 'Fraud'], tickvals=[0, 1])
            fig.update_layout(
                height=chart_height(),
                margin=dict(l=10, r=10, t=40, b=10),
                showlegend=False,
                font=dict(size=12)
            )
            st.plotly_chart(fig, use_container_width=True)

        with col2:
            fig = go.Figure()
            fig.add_trace(go.Histogram(
                x=df[df['is_fraud'] == 0]['time_hour'],
                name='Normal', marker_color='#2ecc71', opacity=0.7, nbinsx=24
            ))
            fig.add_trace(go.Histogram(
                x=df[df['is_fraud'] == 1]['time_hour'],
                name='Fraud', marker_color='#e74c3c', opacity=0.7, nbinsx=24
            ))
            fig.update_layout(
                title='Transaction Time Distribution',
                xaxis_title='Hour of Day',
                yaxis_title='Count',
                barmode='overlay',
                height=chart_height(),
                margin=dict(l=10, r=10, t=40, b=10),
                legend=dict(orientation="h", yanchor="bottom", y=-0.2),
                font=dict(size=12)
            )
            st.plotly_chart(fig, use_container_width=True)

            # Scatter — cap points for perf on mobile
            sample_df = df.sample(min(2000, len(df)), random_state=1)
            fig = px.scatter(
                sample_df,
                x='distance_from_home', y='amount',
                color='is_fraud',
                labels={'distance_from_home': 'Distance from Home (km)', 'amount': 'Amount ($)'},
                title='Distance vs Amount',
                color_discrete_map={0: '#2ecc71', 1: '#e74c3c'},
                opacity=0.6
            )
            fig.update_layout(
                height=chart_height(),
                margin=dict(l=10, r=10, t=40, b=10),
                legend=dict(orientation="h", yanchor="bottom", y=-0.2),
                font=dict(size=12)
            )
            st.plotly_chart(fig, use_container_width=True)

        # Correlation heatmap — full width
        st.subheader("Feature Correlation Matrix")
        top_features = ['amount', 'distance_from_home', 'ratio_to_median',
                        'velocity_1h', 'velocity_24h', 'unusual_time', 'is_fraud']
        corr = df[top_features].corr()
        fig = px.imshow(
            corr, text_auto='.2f', aspect='auto',
            color_continuous_scale='RdBu_r',
            title='Correlation Heatmap'
        )
        fig.update_layout(
            height=380,
            margin=dict(l=10, r=10, t=40, b=10),
            font=dict(size=11)
        )
        st.plotly_chart(fig, use_container_width=True)

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # TAB 2 — Model Training
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    with tab2:
        st.header("Model Training")

        if st.button("🚀 Train Models", type="primary"):
            with st.spinner("Training models… Please wait"):
                X = df.drop('is_fraud', axis=1)
                y = df['is_fraud']
                X_train, X_test, y_train, y_test = train_test_split(
                    X, y, test_size=test_size, random_state=42, stratify=y
                )
                scaler = StandardScaler()
                X_train_scaled = scaler.fit_transform(X_train)
                X_test_scaled  = scaler.transform(X_test)

                models = {
                    'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42),
                    'Random Forest':       RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42),
                    'Gradient Boosting':   GradientBoostingClassifier(n_estimators=100, max_depth=5, random_state=42),
                    'SVM':                 SVC(kernel='rbf', probability=True, random_state=42)
                }

                results      = {}
                progress_bar = st.progress(0)

                for idx, (name, model) in enumerate(models.items()):
                    model.fit(X_train_scaled, y_train)
                    y_pred       = model.predict(X_test_scaled)
                    y_pred_proba = model.predict_proba(X_test_scaled)[:, 1]

                    results[name] = {
                        'model':          model,
                        'predictions':    y_pred,
                        'probabilities':  y_pred_proba,
                        'accuracy':       accuracy_score(y_test, y_pred),
                        'f1':             f1_score(y_test, y_pred),
                        'roc_auc':        roc_auc_score(y_test, y_pred_proba),
                        'confusion_matrix': confusion_matrix(y_test, y_pred)
                    }
                    progress_bar.progress((idx + 1) / len(models))

                st.session_state['results']  = results
                st.session_state['X_test']   = X_test_scaled
                st.session_state['y_test']   = y_test
                st.session_state['scaler']   = scaler
                st.success("✅ Models trained successfully!")

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # TAB 3 — Results
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    with tab3:
        if 'results' in st.session_state:
            st.header("Model Performance")

            results  = st.session_state['results']
            y_test   = st.session_state['y_test']

            model_names = list(results.keys())
            accuracies  = [results[m]['accuracy'] for m in model_names]
            f1_scores   = [results[m]['f1']       for m in model_names]
            roc_aucs    = [results[m]['roc_auc']  for m in model_names]

            col1, col2 = st.columns([1, 1], gap="medium")

            with col1:
                fig = go.Figure()
                fig.add_trace(go.Bar(name='Accuracy', x=model_names, y=accuracies, marker_color='#3498db'))
                fig.add_trace(go.Bar(name='F1-Score', x=model_names, y=f1_scores,  marker_color='#2ecc71'))
                fig.add_trace(go.Bar(name='ROC-AUC',  x=model_names, y=roc_aucs,   marker_color='#e74c3c'))
                fig.update_layout(
                    title='Model Performance Comparison',
                    barmode='group',
                    yaxis_title='Score',
                    yaxis=dict(range=[0, 1.1]),
                    height=chart_height(),
                    margin=dict(l=10, r=10, t=40, b=60),
                    legend=dict(orientation="h", yanchor="bottom", y=-0.3),
                    font=dict(size=11)
                )
                st.plotly_chart(fig, use_container_width=True)

            with col2:
                fig = go.Figure()
                for name in model_names:
                    fpr, tpr, _ = roc_curve(y_test, results[name]['probabilities'])
                    fig.add_trace(go.Scatter(
                        x=fpr, y=tpr, mode='lines',
                        name=f"{name} ({results[name]['roc_auc']:.3f})"
                    ))
                fig.add_trace(go.Scatter(
                    x=[0, 1], y=[0, 1], mode='lines',
                    name='Random', line=dict(dash='dash', color='gray')
                ))
                fig.update_layout(
                    title='ROC Curves',
                    xaxis_title='False Positive Rate',
                    yaxis_title='True Positive Rate',
                    height=chart_height(),
                    margin=dict(l=10, r=10, t=40, b=60),
                    legend=dict(orientation="h", yanchor="bottom", y=-0.4, font=dict(size=10)),
                    font=dict(size=11)
                )
                st.plotly_chart(fig, use_container_width=True)

            # Confusion matrices — 2 per row on small screens, 4 on wide
            st.subheader("Confusion Matrices")

            # Row 1: first two models
            cm_col1, cm_col2 = st.columns(2, gap="small")
            for idx, (col, name) in enumerate(zip([cm_col1, cm_col2], model_names[:2])):
                with col:
                    cm = results[name]['confusion_matrix']
                    fig = px.imshow(
                        cm, text_auto=True,
                        labels=dict(x="Predicted", y="Actual"),
                        x=['Normal', 'Fraud'], y=['Normal', 'Fraud'],
                        color_continuous_scale='Blues', title=name
                    )
                    fig.update_layout(height=260, margin=dict(l=5, r=5, t=35, b=5), font=dict(size=11))
                    st.plotly_chart(fig, use_container_width=True)

            # Row 2: next two models
            cm_col3, cm_col4 = st.columns(2, gap="small")
            for idx, (col, name) in enumerate(zip([cm_col3, cm_col4], model_names[2:])):
                with col:
                    cm = results[name]['confusion_matrix']
                    fig = px.imshow(
                        cm, text_auto=True,
                        labels=dict(x="Predicted", y="Actual"),
                        x=['Normal', 'Fraud'], y=['Normal', 'Fraud'],
                        color_continuous_scale='Blues', title=name
                    )
                    fig.update_layout(height=260, margin=dict(l=5, r=5, t=35, b=5), font=dict(size=11))
                    st.plotly_chart(fig, use_container_width=True)

            # Feature importance
            st.subheader("Feature Importance (Random Forest)")
            rf_model     = results['Random Forest']['model']
            feature_names = [
                'amount', 'time_hour', 'day_of_week', 'dist_home',
                'dist_last', 'ratio', 'repeat', 'chip', 'pin', 'online',
                'vel_1h', 'vel_24h', 'avg_10', 'std_10', 'category',
                'card_present', 'intl', 'high_risk', 'unusual', 'weekend'
            ]
            importances = rf_model.feature_importances_
            indices     = np.argsort(importances)[::-1][:10]

            fig = go.Figure(go.Bar(
                x=importances[indices],
                y=[feature_names[i] for i in indices],
                orientation='h',
                marker_color='#3498db'
            ))
            fig.update_layout(
                title='Top 10 Important Features',
                xaxis_title='Importance',
                yaxis_title='Feature',
                height=350,
                margin=dict(l=10, r=10, t=40, b=10),
                font=dict(size=12)
            )
            st.plotly_chart(fig, use_container_width=True)

        else:
            st.info("👈 Please train models first in the 'Model Training' tab")

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # TAB 4 — Test Prediction
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    with tab4:
        if 'results' in st.session_state:
            st.header("Test New Transaction")

            # Responsive: 2 cols on medium+, 1 col on very small
            col1, col2 = st.columns(2, gap="medium")

            with col1:
                st.markdown("**Transaction Details**")
                amount        = st.number_input("Amount ($)",                  10.0,  10000.0, 100.0)
                time_hour     = st.slider("Hour of Day",                        0, 23, 12)
                distance_home = st.number_input("Distance from Home (km)",      0.0,  1000.0,  10.0)
                velocity_1h   = st.number_input("Transactions in Last Hour",    0,    10,       1)
                velocity_24h  = st.number_input("Transactions in 24h",          0,    20,       3)

            with col2:
                st.markdown("**Transaction Flags**")
                online        = st.selectbox("Online Order",       [0, 1], format_func=lambda x: "Yes" if x else "No")
                chip          = st.selectbox("Used Chip",          [0, 1], format_func=lambda x: "Yes" if x else "No")
                pin           = st.selectbox("Used PIN",           [0, 1], format_func=lambda x: "Yes" if x else "No")
                international = st.selectbox("International",      [0, 1], format_func=lambda x: "Yes" if x else "No")
                unusual_time  = st.selectbox("Unusual Time",       [0, 1], format_func=lambda x: "Yes" if x else "No")
                day_of_week   = st.slider("Day of Week (0=Mon)",   0, 6, 3)
                ratio         = st.number_input("Ratio to Median", 0.1, 10.0, 1.0)

            st.markdown("")
            if st.button("🔍 Predict Fraud", type="primary"):
                transaction = np.array([[
                    amount, time_hour, day_of_week, distance_home,
                    distance_home * 0.5, ratio, 0, chip, pin, online,
                    velocity_1h, velocity_24h, amount, amount * 0.3, 5,
                    1, international, 0, unusual_time,
                    1 if day_of_week >= 5 else 0
                ]])

                scaler             = st.session_state['scaler']
                transaction_scaled = scaler.transform(transaction)
                results            = st.session_state['results']

                st.markdown("### 🧠 Predictions")
                st.markdown("---")

                for name, result in results.items():
                    model = result['model']
                    pred  = model.predict(transaction_scaled)[0]
                    proba = model.predict_proba(transaction_scaled)[0]

                    m_col1, m_col2, m_col3 = st.columns([3, 2, 2], gap="small")
                    with m_col1:
                        st.write(f"**{name}**")
                    with m_col2:
                        if pred == 1:
                            st.error("🚨 FRAUD")
                        else:
                            st.success("✅ NORMAL")
                    with m_col3:
                        st.write(f"Confidence: **{proba[int(pred)]*100:.1f}%**")
                    st.markdown("---")

        else:
            st.info("👈 Please train models first in the 'Model Training' tab")

else:
    st.info("👈 Click **'Generate Data'** in the sidebar to start!")
    st.markdown("""
    ### How to use this app
    1. Open the **sidebar** (arrow on the top-left) and adjust the sliders
    2. Click **Generate Data** to create a synthetic fraud dataset
    3. Explore the data in the **Data Exploration** tab
    4. Train ML models in the **Model Training** tab
    5. Compare results in the **Results** tab
    6. Test custom transactions in **Test Prediction**
    """)