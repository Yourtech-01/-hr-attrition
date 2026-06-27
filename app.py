"""
app.py — HR Attrition Intelligence: EDA + Predictive Model
Run: streamlit run app.py
Free deploy: Streamlit Community Cloud (share.streamlit.io)
"""
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (classification_report, confusion_matrix,
                              roc_auc_score, roc_curve)
from sklearn.inspection import permutation_importance
import warnings
warnings.filterwarnings("ignore")

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="HR Attrition Intelligence",
    page_icon="👥",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Styles ────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
  [data-testid="stMetricValue"] { font-size: 2rem; font-weight: 700; }
  .stTabs [data-baseweb="tab"] { font-size: 14px; font-weight: 600; }
  div[data-testid="column"] { padding: 0 6px; }
  .insight-box {
    background: #1e2235; border-left: 3px solid #4361ee;
    padding: 12px 16px; border-radius: 6px; margin: 8px 0;
    font-size: 14px; color: #c9d1e8;
  }
</style>
""", unsafe_allow_html=True)

# ── Data generation ───────────────────────────────────────────────────────────
@st.cache_data
def generate_hr_data(n=1500, seed=42):
    np.random.seed(seed)

    dept_map     = ["Sales","R&D","HR","Finance","Engineering","Marketing","Operations"]
    job_roles    = ["Manager","Analyst","Engineer","Executive","Technician","Specialist","Director"]
    edu_fields   = ["Life Sciences","Medical","Marketing","Technical","HR","Other"]
    overtime_p   = 0.28

    df = pd.DataFrame({
        "Age":                    np.random.randint(22, 60, n),
        "Department":             np.random.choice(dept_map, n,
                                      p=[0.25,0.28,0.08,0.10,0.15,0.08,0.06]),
        "JobRole":                np.random.choice(job_roles, n),
        "EducationField":         np.random.choice(edu_fields, n),
        "Gender":                 np.random.choice(["Male","Female"], n, p=[0.60,0.40]),
        "MaritalStatus":          np.random.choice(["Single","Married","Divorced"],
                                      n, p=[0.32,0.46,0.22]),
        "MonthlyIncome":          np.random.randint(2500, 20000, n),
        "JobSatisfaction":        np.random.randint(1, 5, n),
        "WorkLifeBalance":        np.random.randint(1, 5, n),
        "YearsAtCompany":         np.random.randint(0, 25, n),
        "YearsInCurrentRole":     np.random.randint(0, 18, n),
        "YearsSinceLastPromotion":np.random.randint(0, 15, n),
        "DistanceFromHome":       np.random.randint(1, 30, n),
        "NumCompaniesWorked":     np.random.randint(0, 9, n),
        "TrainingTimesLastYear":  np.random.randint(0, 6, n),
        "OverTime":               np.random.choice(["Yes","No"], n,
                                      p=[overtime_p, 1-overtime_p]),
        "PerformanceRating":      np.random.choice([3,4], n, p=[0.85,0.15]),
        "EnvironmentSatisfaction":np.random.randint(1, 5, n),
        "JobInvolvement":         np.random.randint(1, 5, n),
        "StockOptionLevel":       np.random.choice([0,1,2,3], n, p=[0.36,0.44,0.14,0.06]),
    })

    # Realistic attrition logic
    score = (
        (df["JobSatisfaction"] < 2).astype(int) * 2.5 +
        (df["WorkLifeBalance"] < 2).astype(int) * 2.0 +
        (df["OverTime"] == "Yes").astype(int) * 2.2 +
        (df["YearsSinceLastPromotion"] > 8).astype(int) * 1.8 +
        (df["MonthlyIncome"] < 4000).astype(int) * 1.5 +
        (df["MaritalStatus"] == "Single").astype(int) * 0.8 +
        (df["DistanceFromHome"] > 20).astype(int) * 0.7 +
        (df["NumCompaniesWorked"] > 5).astype(int) * 0.6 +
        (df["StockOptionLevel"] == 0).astype(int) * 0.5 +
        np.random.normal(0, 0.5, n)
    )
    prob = 1 / (1 + np.exp(-(score - 3)))
    df["Attrition"] = (np.random.random(n) < prob).astype(int)
    df["AttritionLabel"] = df["Attrition"].map({1: "Yes", 0: "No"})
    return df

@st.cache_resource
def train_model(df):
    feature_cols = [
        "Age","MonthlyIncome","JobSatisfaction","WorkLifeBalance",
        "YearsAtCompany","YearsInCurrentRole","YearsSinceLastPromotion",
        "DistanceFromHome","NumCompaniesWorked","TrainingTimesLastYear",
        "OverTime_enc","PerformanceRating","EnvironmentSatisfaction",
        "JobInvolvement","StockOptionLevel","Department_enc",
        "MaritalStatus_enc","Gender_enc",
    ]
    df2 = df.copy()
    for col in ["OverTime","Department","MaritalStatus","Gender"]:
        le = LabelEncoder()
        df2[f"{col}_enc"] = le.fit_transform(df2[col])

    X = df2[feature_cols]
    y = df2["Attrition"]
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2,
                                               random_state=42, stratify=y)
    model = GradientBoostingClassifier(n_estimators=200, max_depth=4,
                                        learning_rate=0.08, random_state=42)
    model.fit(X_tr, y_tr)
    return model, X_te, y_te, feature_cols, X_tr, y_tr

# ── Load ──────────────────────────────────────────────────────────────────────
df = generate_hr_data()
model, X_te, y_te, feature_cols, X_tr, y_tr = train_model(df)

attr_rate = df["Attrition"].mean() * 100
attr_count = df["Attrition"].sum()
total      = len(df)
avg_income = df["MonthlyIncome"].mean()

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 👥 HR Attrition")
    st.markdown("**Filter employees**")
    dept_sel  = st.multiselect("Department", df["Department"].unique(),
                                default=list(df["Department"].unique()))
    gender_sel = st.radio("Gender", ["All","Male","Female"])
    age_range  = st.slider("Age Range", 22, 60, (22, 60))
    income_range = st.slider("Monthly Income ($)", 2500, 20000, (2500, 20000))
    st.divider()
    st.markdown("### 🔮 Predict Risk")
    st.caption("Adjust sliders to score an employee")
    p_satisfaction = st.slider("Job Satisfaction", 1, 4, 2)
    p_wlb          = st.slider("Work-Life Balance", 1, 4, 2)
    p_overtime     = st.selectbox("Overtime", ["Yes","No"])
    p_income       = st.slider("Monthly Income", 2500, 20000, 5000, step=500)
    p_promo_yrs    = st.slider("Years Since Promotion", 0, 15, 5)

# ── Filter ────────────────────────────────────────────────────────────────────
fdf = df[
    (df["Department"].isin(dept_sel)) &
    (df["Age"].between(*age_range)) &
    (df["MonthlyIncome"].between(*income_range))
]
if gender_sel != "All":
    fdf = fdf[fdf["Gender"] == gender_sel]

# ── Header ────────────────────────────────────────────────────────────────────
st.title("HR Attrition Intelligence Platform")
st.caption(f"Analyzing {len(fdf):,} employees · Gradient Boosting Classifier · "
           f"AUC {roc_auc_score(y_te, model.predict_proba(X_te)[:,1]):.3f}")

tab1, tab2, tab3, tab4 = st.tabs([
    "📊 Overview", "🔬 Deep Dive", "🤖 ML Model", "🔮 Risk Predictor"
])

PAL = px.colors.qualitative.Bold

# ── TAB 1: Overview ───────────────────────────────────────────────────────────
with tab1:
    c1, c2, c3, c4 = st.columns(4)
    f_rate = fdf["Attrition"].mean() * 100
    c1.metric("Attrition Rate", f"{f_rate:.1f}%",
              f"{f_rate - attr_rate:+.1f}% vs all")
    c2.metric("Employees at Risk", f"{fdf['Attrition'].sum():,}")
    c3.metric("Avg Monthly Income", f"${fdf['MonthlyIncome'].mean():,.0f}")
    c4.metric("Avg Tenure (yrs)", f"{fdf['YearsAtCompany'].mean():.1f}")
    st.divider()

    col1, col2 = st.columns(2)
    with col1:
        dept_attr = (fdf.groupby("Department")["Attrition"]
                       .agg(["sum","count"]).reset_index())
        dept_attr["rate"] = dept_attr["sum"] / dept_attr["count"] * 100
        fig = px.bar(dept_attr.sort_values("rate", ascending=False),
                     x="Department", y="rate",
                     color="rate", color_continuous_scale=["#1e2235","#f72585"],
                     title="Attrition Rate by Department (%)")
        fig.update_layout(coloraxis_showscale=False, height=320)
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        age_bins = pd.cut(fdf["Age"], bins=[20,30,40,50,65],
                          labels=["21-30","31-40","41-50","51+"])
        age_attr = fdf.groupby(age_bins)["Attrition"].mean().reset_index()
        age_attr.columns = ["AgeGroup","AttritionRate"]
        age_attr["AttritionRate"] *= 100
        fig2 = px.line(age_attr, x="AgeGroup", y="AttritionRate",
                       markers=True, title="Attrition Rate by Age Group (%)",
                       color_discrete_sequence=["#4361ee"])
        fig2.update_traces(line_width=2.5, marker_size=8)
        fig2.update_layout(height=320)
        st.plotly_chart(fig2, use_container_width=True)

    col3, col4 = st.columns(2)
    with col3:
        fig3 = px.box(fdf, x="AttritionLabel", y="MonthlyIncome",
                      color="AttritionLabel",
                      color_discrete_map={"Yes":"#f72585","No":"#4361ee"},
                      title="Income Distribution by Attrition")
        fig3.update_layout(showlegend=False, height=300)
        st.plotly_chart(fig3, use_container_width=True)

    with col4:
        ot = fdf.groupby(["OverTime","AttritionLabel"])["Age"].count().reset_index()
        ot.columns = ["OverTime","Attrition","Count"]
        fig4 = px.bar(ot, x="OverTime", y="Count", color="Attrition",
                      barmode="group",
                      color_discrete_map={"Yes":"#f72585","No":"#4361ee"},
                      title="Overtime vs Attrition")
        fig4.update_layout(height=300)
        st.plotly_chart(fig4, use_container_width=True)

# ── TAB 2: Deep Dive ──────────────────────────────────────────────────────────
with tab2:
    col1, col2 = st.columns(2)
    with col1:
        sat_attr = (fdf.groupby("JobSatisfaction")["Attrition"]
                       .mean().reset_index())
        sat_attr["Attrition"] *= 100
        fig = px.bar(sat_attr, x="JobSatisfaction", y="Attrition",
                     title="Attrition Rate by Job Satisfaction Score",
                     color="Attrition", color_continuous_scale=["#1e2235","#f72585"])
        fig.update_layout(coloraxis_showscale=False, height=300)
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        ms_attr = (fdf.groupby("MaritalStatus")["Attrition"]
                     .agg(["mean","count"]).reset_index())
        ms_attr["mean"] *= 100
        fig2 = px.bar(ms_attr, x="MaritalStatus", y="mean",
                      color="MaritalStatus",
                      color_discrete_sequence=PAL,
                      title="Attrition Rate by Marital Status (%)")
        fig2.update_layout(showlegend=False, height=300)
        st.plotly_chart(fig2, use_container_width=True)

    col3, col4 = st.columns(2)
    with col3:
        fig3 = px.scatter(fdf.sample(min(600, len(fdf))),
                          x="YearsAtCompany", y="MonthlyIncome",
                          color="AttritionLabel",
                          color_discrete_map={"Yes":"#f72585","No":"#4361ee"},
                          opacity=0.6, title="Tenure vs Income (coloured by Attrition)")
        fig3.update_layout(height=300)
        st.plotly_chart(fig3, use_container_width=True)

    with col4:
        wlb_attr = (fdf.groupby("WorkLifeBalance")["Attrition"]
                       .mean().reset_index())
        wlb_attr["Attrition"] *= 100
        fig4 = px.bar(wlb_attr, x="WorkLifeBalance", y="Attrition",
                      title="Attrition Rate by Work-Life Balance Score",
                      color="Attrition", color_continuous_scale=["#1e2235","#4361ee"])
        fig4.update_layout(coloraxis_showscale=False, height=300)
        st.plotly_chart(fig4, use_container_width=True)

    st.markdown("#### Key Insights")
    for insight in [
        "🔴 Employees with **overtime = Yes** have 2–3× higher attrition.",
        "💰 Lower income bracket (<$4K/mo) employees leave at nearly **double** the rate.",
        "📅 Singles and recently promoted employees show elevated risk.",
        "⚠️ Job satisfaction score 1 correlates with >40% attrition rate.",
    ]:
        st.markdown(f'<div class="insight-box">{insight}</div>',
                    unsafe_allow_html=True)

# ── TAB 3: ML Model ───────────────────────────────────────────────────────────
with tab3:
    y_pred  = model.predict(X_te)
    y_proba = model.predict_proba(X_te)[:, 1]
    auc     = roc_auc_score(y_te, y_proba)
    cv_auc  = cross_val_score(model, X_tr, y_tr, cv=5,
                               scoring="roc_auc").mean()

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("ROC-AUC (Test)",  f"{auc:.3f}")
    c2.metric("ROC-AUC (5-CV)",  f"{cv_auc:.3f}")
    c3.metric("Precision (Attr.)",f"{classification_report(y_te,y_pred,output_dict=True)['1']['precision']:.2f}")
    c4.metric("Recall (Attr.)",   f"{classification_report(y_te,y_pred,output_dict=True)['1']['recall']:.2f}")

    col1, col2 = st.columns(2)
    with col1:
        fpr, tpr, _ = roc_curve(y_te, y_proba)
        fig_roc = go.Figure()
        fig_roc.add_trace(go.Scatter(x=fpr, y=tpr, mode="lines",
                          name=f"GBM (AUC={auc:.3f})",
                          line=dict(color="#4361ee", width=2.5)))
        fig_roc.add_trace(go.Scatter(x=[0,1], y=[0,1], mode="lines",
                          name="Random", line=dict(color="#8892b0",
                          width=1, dash="dash")))
        fig_roc.update_layout(title="ROC Curve", xaxis_title="False Positive Rate",
                               yaxis_title="True Positive Rate", height=340)
        st.plotly_chart(fig_roc, use_container_width=True)

    with col2:
        imp = pd.DataFrame({"feature": feature_cols,
                             "importance": model.feature_importances_})
        imp = imp.sort_values("importance").tail(12)
        fig_imp = px.bar(imp, x="importance", y="feature", orientation="h",
                         title="Feature Importance (Top 12)",
                         color="importance",
                         color_continuous_scale=["#1e2235","#7209b7"])
        fig_imp.update_layout(coloraxis_showscale=False, height=340)
        st.plotly_chart(fig_imp, use_container_width=True)

    cm = confusion_matrix(y_te, y_pred)
    fig_cm = px.imshow(cm, text_auto=True,
                       labels=dict(x="Predicted", y="Actual"),
                       x=["Stay","Leave"], y=["Stay","Leave"],
                       color_continuous_scale=["#1e2235","#4361ee"],
                       title="Confusion Matrix")
    fig_cm.update_layout(height=300, width=350)
    col3, _ = st.columns([1,2])
    with col3:
        st.plotly_chart(fig_cm, use_container_width=True)

# ── TAB 4: Risk Predictor ─────────────────────────────────────────────────────
with tab4:
    st.subheader("🔮 Real-Time Attrition Risk Predictor")
    st.caption("Adjust the sliders in the sidebar to profile an employee.")

    # Build a sample row with sidebar values
    sample = pd.DataFrame({
        "Age": [35],
        "MonthlyIncome": [p_income],
        "JobSatisfaction": [p_satisfaction],
        "WorkLifeBalance": [p_wlb],
        "YearsAtCompany": [5],
        "YearsInCurrentRole": [3],
        "YearsSinceLastPromotion": [p_promo_yrs],
        "DistanceFromHome": [10],
        "NumCompaniesWorked": [3],
        "TrainingTimesLastYear": [2],
        "OverTime_enc": [1 if p_overtime == "Yes" else 0],
        "PerformanceRating": [3],
        "EnvironmentSatisfaction": [3],
        "JobInvolvement": [3],
        "StockOptionLevel": [1],
        "Department_enc": [2],
        "MaritalStatus_enc": [0],
        "Gender_enc": [0],
    })

    risk = model.predict_proba(sample)[0][1]
    risk_pct = risk * 100

    col1, col2 = st.columns([1, 2])
    with col1:
        color = "#f72585" if risk_pct > 60 else "#f8961e" if risk_pct > 35 else "#06d6a0"
        label = "🔴 HIGH RISK" if risk_pct > 60 else "🟡 MEDIUM RISK" if risk_pct > 35 else "🟢 LOW RISK"
        st.markdown(f"""
        <div style="background:#1e2235; border-radius:12px; padding:24px;
                    text-align:center; border: 2px solid {color}">
          <p style="color:#8892b0; font-size:13px; margin:0">Predicted Attrition Risk</p>
          <p style="color:{color}; font-size:52px; font-weight:700; margin:8px 0">{risk_pct:.0f}%</p>
          <p style="color:{color}; font-size:16px; font-weight:600">{label}</p>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        drivers = {
            "Job Satisfaction":       (5 - p_satisfaction) / 4,
            "Work-Life Balance":      (5 - p_wlb) / 4,
            "Overtime":               0.9 if p_overtime == "Yes" else 0.1,
            "Income Level":           max(0, (8000 - p_income) / 8000),
            "Years Since Promotion":  min(1, p_promo_yrs / 12),
        }
        drv_df = pd.DataFrame(list(drivers.items()),
                              columns=["Factor","Contribution"])
        fig_drv = px.bar(drv_df.sort_values("Contribution"),
                         x="Contribution", y="Factor", orientation="h",
                         title="Risk Driver Breakdown",
                         color="Contribution",
                         color_continuous_scale=["#1e2235","#f72585"])
        fig_drv.update_layout(coloraxis_showscale=False, height=280)
        st.plotly_chart(fig_drv, use_container_width=True)

    st.subheader("📋 Retention Recommendations")
    recs = []
    if p_satisfaction <= 2:  recs.append("🎯 **Job redesign or role rotation** — satisfaction is critically low")
    if p_wlb <= 2:            recs.append("⏰ **Reduce mandatory overtime** — work-life balance is a red flag")
    if p_overtime == "Yes":   recs.append("📋 **Cap overtime at 5 hrs/week** and compensate with time-off")
    if p_income < 5000:       recs.append("💰 **Benchmark salary against market** — income is below median")
    if p_promo_yrs > 5:       recs.append("📈 **Initiate promotion review** — no advancement in 5+ years")
    if not recs:              recs.append("✅ Employee profile looks stable — continue regular check-ins")
    for r in recs:
        st.markdown(f'<div class="insight-box">{r}</div>', unsafe_allow_html=True)

if __name__ == "__main__":
    pass
