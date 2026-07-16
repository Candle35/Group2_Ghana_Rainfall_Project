import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import seaborn as sns
from src.predict import load_model, predict_single, predict_batch
from src.config import MODEL_PATH, TARGET_CLASSES

# -----------------------------------------------------------------------
# PAGE SETUP
# -----------------------------------------------------------------------
st.set_page_config(
    page_title="Ghana Rainfall Predictor",
    page_icon="🌧️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# A little CSS to make things bigger, friendlier, and easier to read
# on the small/mid-range phone screens common in rural areas.
st.markdown("""
    <style>
    html, body, [class*="css"] { font-size: 17px; }
    div.stButton > button {
        width: 100%; padding: 0.9em 1em;
        font-size: 1.1em; font-weight: 600; border-radius: 10px;
    }
    .result-card {
        padding: 1.5em; border-radius: 14px;
        text-align: center; margin-top: 0.5em; margin-bottom: 1em;
    }
    .result-label { font-size: 2em; font-weight: 800; margin: 0; }
    .section-title { font-size: 1.3em; font-weight: 700; margin-top: 0.2em; }
    .action-card {
        padding: 1.2em; border-radius: 12px;
        border-left: 5px solid #0B5563;
        background: #E8F2F2; margin-top: 0.5em;
    }
    .history-card {
        padding: 0.8em; border-radius: 10px;
        background: #F7F9F9; margin-bottom: 0.5em;
        border: 1px solid #DDE5E5;
    }
    .confidence-high { color: #2E7D32; font-weight: 700; }
    .confidence-mid  { color: #F9A825; font-weight: 700; }
    .confidence-low  { color: #C62828; font-weight: 700; }
    </style>
""", unsafe_allow_html=True)

# -----------------------------------------------------------------------
# CONSTANTS — slide palette colors
# -----------------------------------------------------------------------
RESULT_COLORS = {
    'NORAIN':     '#0B5563',
    'SMALLRAIN':  '#4D8A8F',
    'MEDIUMRAIN': '#8FB8BB',
    'HEAVYRAIN':  '#C9763B',
}
RESULT_ICONS = {
    'NORAIN':     '☀️',
    'SMALLRAIN':  '🌦️',
    'MEDIUMRAIN': '🌧️',
    'HEAVYRAIN':  '⛈️',
}

# Plain English descriptions
LABEL_DESCRIPTIONS = {
    'NORAIN':     'No rainfall expected. Your fields will remain dry today.',
    'SMALLRAIN':  'Light drizzle expected. Not enough to meet crop water needs.',
    'MEDIUMRAIN': 'Moderate rainfall expected. Generally good for most crops.',
    'HEAVYRAIN':  'Heavy downpour expected. Risk of flooding or waterlogging.',
}

# Farmer action cards
ACTION_CARDS = {
    'NORAIN':     '🌱 Irrigate today. Avoid applying fertilizer without water. Monitor soil moisture closely.',
    'SMALLRAIN':  '💧 Supplement with irrigation. The rain alone will not be enough for most crops.',
    'MEDIUMRAIN': '✅ Good conditions for transplanting seedlings. Hold off on irrigation for now.',
    'HEAVYRAIN':  '⚠️ Delay planting today. Secure loose equipment. Watch for waterlogging in low-lying fields.',
}

# Twi translations
TWI_LABELS = {
    'NORAIN':     'Nsuo ntɔ (No Rain)',
    'SMALLRAIN':  'Nsuo bɛ piti Kakra (Small Rain)',
    'MEDIUMRAIN': 'Nsuo no bɛtɔ kakra (Medium Rain)',
    'HEAVYRAIN':  'Nsuo bɛtɔ kɛseɛ paa (Heavy Rain)',
}
TWI_ACTIONS = {
    'NORAIN':     '🌱 Fa nsuo gu nnobaeɛ no so. Mfa fertilizer ngu nnobaeɛ no so abrɛ wo mfa nsuo ngu so. Na ma wani nkɔ asaase no mu nsuo no ho.',
    'SMALLRAIN':  '💧 Fa nsuo kakra gu nnobaeɛ no so bio. Nsuo a ɛbɛtɔ no nkoara ntumi mboa nnobaeaɛ no.',
    'MEDIUMRAIN': '✅ Ɛyɛ sɛ wɔ tim nnobaeɛ nketua. Mfa nsuo ngu nnobaeɛ no sesei.',
    'HEAVYRAIN':  '⚠️ Twɛn, ntim nnobaeɛ nketoa no asa. Hwɛ sɛ nsuo nta nta asaase no so pii.',
}

TWI_DESCRIPTIONS = {
    'NORAIN':     'Nsuo rentɔ. Wʼafuo bɛtena dede nnɛ.',
    'SMALLRAIN':  'Nsuo kakra bɛtɔ. Ɛnyɛ nso a ɛho hia afuo no.',
    'MEDIUMRAIN': 'Nsuo mmerɛw bɛtɔ. Ɛyɛ afuo ahorow pii.',
    'HEAVYRAIN':  'Nsuo kɛseɛ bɛtɔ. Nsuo tɔ a ɛboro so na afuo rentumi nsi.',
}

TWI_ADVISORIES = {
    'NORAIN':     'Nsuo rentɔ. Sua nsuo agu afuo so. Mfa tumi agu so kwa.',
    'SMALLRAIN':  'Nsuo kakra bɛtɔ. Sua nsuo agu afuo so bio. Nsuo a ɛbɛtɔ no nso ara.',
    'MEDIUMRAIN': 'Nsuo mmerɛw bɛtɔ. Wobɛtumi a sua nsuo, nanso ɛho nhia.',
    'HEAVYRAIN':  'Nsuo kɛseɛ bɛtɔ. Nsuo tumi agu afuo so. Mfa nnua nkumaa ntwa nnɛ.',
}

TWI_ADVISORIES = {
    'NORAIN':     'Nsuo rentɔ. Sua nsuo agu afuo so. Ɛseseɛ wode nsuo gu nnobaeɛ no so paa.',
    'SMALLRAIN':  'Nsuo kakra bɛtɔ. Fa nsuo kakra gu nnobaeɛ no so. Nsuo kakra ɛbɛtɔ no nkoara ntɔmi mboa nnoɔbaeɛ no.',
    'MEDIUMRAIN': 'Nsuo bɛtɔ kakra. Wobɛtumi dɛ nsuo agu nnoɔbaeɛ no so, nanso ɛho nhia pii saa.',
    'HEAVYRAIN':  'Nsuo kɛseɛ bɛtɔ.  Mfa nsuo ngu nnobaeɛ no so biom',
}

# -----------------------------------------------------------------------
# HELPER FUNCTIONS
# -----------------------------------------------------------------------
def color_for_label(label):
    return RESULT_COLORS.get(label, '#0B5563')

def icon_for_label(label):
    return RESULT_ICONS.get(label, '🌦️')

def irrigation_advisory(predicted_category: str) -> dict:
    category = predicted_category.strip().lower().replace(" ", "")
    if category == "heavyrain":
        return {"level": "no_irrigation_needed",
                "message": "Heavy rainfall predicted. Irrigation not necessary this period."}
    elif category == "mediumrain":
        return {"level": "irrigation_optional",
                "message": "Medium rainfall predicted. Irrigation may help but is not critical."}
    elif category in ("smallrain", "lightrain"):
        return {"level": "irrigation_recommended",
                "message": "Light/small rainfall predicted. Irrigation recommended to supplement crop water needs."}
    elif category == "norain":
        return {"level": "irrigation_strongly_recommended",
                "message": "No rainfall predicted. Irrigation strongly recommended."}
    else:
        return {"level": "unknown",
                "message": f"Unrecognized category ('{predicted_category}'). Unable to provide irrigation advice."}

def confidence_traffic_light(top_prob):
    if top_prob >= 0.70:
        return "high", f"🟢 High confidence ({top_prob*100:.0f}%)"
    elif top_prob >= 0.50:
        return "mid",  f"🟡 Moderate confidence ({top_prob*100:.0f}%) — treat with some caution"
    else:
        return "low",  f"🔴 Low confidence ({top_prob*100:.0f}%) — consider seeking a second opinion"

def plot_donut(probabilities, label):
    """Draw a donut chart of class probabilities."""
    labels = list(probabilities.keys())
    values = list(probabilities.values())
    colors = [RESULT_COLORS.get(l, '#8FB8BB') for l in labels]

    fig, ax = plt.subplots(figsize=(4, 4))
    wedges, _ = ax.pie(
        values, colors=colors, startangle=90,
        wedgeprops=dict(width=0.5, edgecolor='white', linewidth=2)
    )
    # Centre text
    ax.text(0, 0.08, icon_for_label(label), ha='center', va='center', fontsize=28)
    ax.text(0, -0.25, label, ha='center', va='center',
            fontsize=11, fontweight='bold', color=RESULT_COLORS.get(label, '#0B5563'))

    legend_patches = [
        mpatches.Patch(color=colors[i], label=f"{labels[i]}  {values[i]*100:.1f}%")
        for i in range(len(labels))
    ]
    ax.legend(handles=legend_patches, loc='lower center',
              bbox_to_anchor=(0.5, -0.22), ncol=2, fontsize=9, frameon=False)
    ax.set_title("Model confidence per class", fontsize=11, pad=12)
    plt.tight_layout()
    return fig

# -----------------------------------------------------------------------
# SESSION STATE — prediction history
# -----------------------------------------------------------------------
if 'history' not in st.session_state:
    st.session_state.history = []

# -----------------------------------------------------------------------
# HEADER
# -----------------------------------------------------------------------
st.markdown("""
    <style>
    @keyframes gradientShift {
        0%   { background-position: 0% 50%; }
        50%  { background-position: 100% 50%; }
        100% { background-position: 0% 50%; }
    }
    @keyframes fall {
        0%   { top: -10%; opacity: 1; }
        100% { top: 110%; opacity: 0.3; }
    }
    .hero-header {
    background: linear-gradient(270deg, #0B5563, #4D8A8F, #8FB8BB, #0B3D42);
    background-size: 400% 400%;
    animation: gradientShift 8s ease infinite;
    border-radius: 14px;
    padding: 1.5em 2em;
    margin-bottom: 1em;
    position: relative;
    overflow: hidden;
    min-height: 110px;
    }
    .raindrop {
        position: absolute;
        width: 2px;
        border-radius: 2px;
        background: rgba(255, 255, 255, 0.5);
        animation: fall linear infinite;
    }
    .hero-text { position: relative; z-index: 2; }
    </style>
    <div class="hero-header">
        <div class="raindrop" style="left:5%;  height:12px; animation-duration:1.2s; animation-delay:0.0s;"></div>
        <div class="raindrop" style="left:10%; height:18px; animation-duration:0.9s; animation-delay:0.3s;"></div>
        <div class="raindrop" style="left:18%; height:10px; animation-duration:1.5s; animation-delay:0.1s;"></div>
        <div class="raindrop" style="left:25%; height:15px; animation-duration:1.1s; animation-delay:0.7s;"></div>
        <div class="raindrop" style="left:33%; height:20px; animation-duration:0.8s; animation-delay:0.4s;"></div>
        <div class="raindrop" style="left:40%; height:12px; animation-duration:1.3s; animation-delay:0.2s;"></div>
        <div class="raindrop" style="left:48%; height:16px; animation-duration:1.0s; animation-delay:0.6s;"></div>
        <div class="raindrop" style="left:55%; height:14px; animation-duration:1.4s; animation-delay:0.1s;"></div>
        <div class="raindrop" style="left:62%; height:11px; animation-duration:0.9s; animation-delay:0.5s;"></div>
        <div class="raindrop" style="left:70%; height:19px; animation-duration:1.2s; animation-delay:0.3s;"></div>
        <div class="raindrop" style="left:77%; height:13px; animation-duration:1.1s; animation-delay:0.8s;"></div>
        <div class="raindrop" style="left:84%; height:17px; animation-duration:0.7s; animation-delay:0.2s;"></div>
        <div class="raindrop" style="left:91%; height:10px; animation-duration:1.6s; animation-delay:0.4s;"></div>
        <div class="raindrop" style="left:96%; height:14px; animation-duration:1.0s; animation-delay:0.9s;"></div>
        <div class="hero-text">
            <h1 style="color:#FFFFFF; margin:0; font-size:2.5em; font-weight:800;">
                🌦️ Ghana Rainfall Predictor
            </h1>
            <p style="color:#E0F0F0; margin:0.5em 0 0 0; font-size:1.05em;">
                <strong>A simple tool for farmers and extension officers in the Pra River Basin</strong> · Indigenous Knowledge
            </p>
        </div>
    </div>
""", unsafe_allow_html=True)

# -----------------------------------------------------------------------
# SIDEBAR
# -----------------------------------------------------------------------
with st.sidebar:
    st.header("ℹ️ How to use this app")
    st.markdown("""
        **Step 1.** Choose *Single Prediction* or *Batch Prediction*.

        **Step 2.** Fill in the details as accurately as you can.

        **Step 3.** Press **Predict** to see the result.

        **Need help?**
        Contact your local agricultural extension officer.
    """)
    st.divider()

    # Language toggle
    lang = st.radio("🌐 Language / Kasa", ["English", "Twi"], horizontal=True)
    st.divider()

    # District map
    st.markdown("**📍 Covered Districts**")
    st.markdown("""
    - 🟢 **Atiwa West** — Eastern Region
    - 🟡 **Assin Fosu** — Central Region
    - 🔵 **Obuasi East** — Ashanti Region
    """)
    st.divider()
    st.caption("Built to support smallholder farmers with rainfall planning.")

# -----------------------------------------------------------------------
# LOAD MODEL
# -----------------------------------------------------------------------
@st.cache_resource
def get_model():
    if not os.path.exists(MODEL_PATH):
        st.error(
            "⚠️ Model not found.\n\n"
            "Run `python -m src.train` first to train and save the model."
        )
        st.stop()
    return load_model()

model = get_model()

# -----------------------------------------------------------------------
# TABS
# -----------------------------------------------------------------------
tab1, tab2, tab3 = st.tabs([
    "👤 Single Prediction",
    "📄 Batch Prediction (CSV)",
    "🕐 Prediction History"
])

# -----------------------------------------------------------------------
# TAB 1 — SINGLE PREDICTION
# -----------------------------------------------------------------------
with tab1:
    st.markdown('<p class="section-title">Farmer & Community Details</p>', unsafe_allow_html=True)
    col1, col2 = st.columns(2)

    with col1:
        user_id = st.number_input(
            "Farmer ID", min_value=1, max_value=10000, value=11,
            help="The unique number assigned to this farmer's records."
        )
        community = st.text_input(
            "Community / Village", "Akwaduuso",
            help="The name of the town or village where the observation was made."
        )
        district = st.selectbox(
            "District",
            ["assin_fosu", "atiwa_west", "obuasi_east"],
            format_func=lambda x: x.replace("_", " ").title(),
            help="The administrative district the community belongs to."
        )

    with col2:
        prediction_month = st.selectbox(
            "Month of Observation",
            list(range(1, 13)),
            format_func=lambda m: pd.Timestamp(2000, m, 1).strftime("%B"),
        )
        prediction_hour = st.slider(
            "Time of Day (Hour)", 0, 23, 8,
            help="0 = midnight, 12 = noon"
        )
        forecast_length = st.selectbox(
            "How far ahead is this forecast?",
            [12, 24],
            format_func=lambda h: f"{h} hours",
        )

    st.markdown('<p class="section-title">Farmer\'s Own Forecast</p>', unsafe_allow_html=True)
    col3, col4 = st.columns(2)
    with col3:
        confidence = st.select_slider(
            "How confident is the farmer?",
            options=[0.3, 0.6, 1.0],
            value=0.6,
            format_func=lambda x: {0.3: "Low", 0.6: "Medium", 1.0: "High"}[x],
        )
    with col4:
        predicted_intensity = st.slider(
            "Expected Rainfall Intensity (0 = none, 10 = very heavy)",
            0.0, 10.0, 2.0, step=0.5,
        )

    st.write("")
    predict_clicked = st.button("🔮 Predict Rainfall Type", type="primary", key="single_predict")

    if predict_clicked:
        with st.spinner("Analyzing indicators and forecast data..."):
            input_data = {
                "user_id":             user_id,
                "confidence":          confidence,
                "predicted_intensity": predicted_intensity,
                "community":           community,
                "district":            district,
                "forecast_length":     forecast_length,
                "prediction_hour":     prediction_hour,
                "prediction_month":    prediction_month,
                "prediction_time":     pd.Timestamp.now(),
            }
            result = predict_single(model, input_data)

        label  = result["label"]
        color  = color_for_label(label)
        icon   = icon_for_label(label)
        top_prob = max(result["probabilities"].values())
        conf_level, conf_msg = confidence_traffic_light(top_prob)

        # Display label in chosen language
        display_label  = TWI_LABELS.get(label, label) if lang == "Twi" else label
        display_desc   = TWI_DESCRIPTIONS.get(label, "") if lang == "Twi" else LABEL_DESCRIPTIONS.get(label, "")
        display_action = TWI_ACTIONS.get(label, ACTION_CARDS.get(label, "")) \
                    if lang == "Twi" else ACTION_CARDS.get(label, "")

        st.divider()

        # Result card
        st.markdown(
            f"""
            <div class="result-card" style="background:{color}22; border:2px solid {color};">
                <div style="font-size:3em;">{icon}</div>
                <p class="result-label" style="color:{color};">{display_label}</p>
                <p style="color:#555; margin-top:0.3em;">{display_desc}</p>
                <p style="color:#888; font-size:0.85em;">
                    Predicted for {community}, {district.replace('_',' ').title()}
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )

        # Confidence traffic light
        css_class = f"confidence-{conf_level}"
        st.markdown(f'<p class="{css_class}">{conf_msg}</p>', unsafe_allow_html=True)

        st.divider()

        # Donut chart + action card side by side
        res_col1, res_col2 = st.columns([1, 1])

        with res_col1:
            st.markdown("**Model confidence per rainfall class:**")
            fig = plot_donut(result["probabilities"], label)
            st.pyplot(fig)
            plt.close()

        with res_col2:
            st.markdown("**💡 What should you do today?**")
            st.markdown(
                f'<div class="action-card">{display_action}</div>',
                unsafe_allow_html=True
            )
            st.write("")

            # Irrigation advisory
            advisory = irrigation_advisory(label)
            advisory_msg = TWI_ADVISORIES.get(label, advisory["message"]) \
                            if lang == "Twi" else advisory["message"]
            st.markdown("**💧 Irrigation Advisory**" if lang == "English" else "**💧 Nsuo Gu Afuo So**")
            if advisory["level"] == "no_irrigation_needed":
                st.success(advisory_msg)
            elif advisory["level"] == "irrigation_optional":
                st.info(advisory_msg)
            elif advisory["level"] == "irrigation_recommended":
                st.warning(advisory_msg)
            else:
                st.error(advisory_msg)

        st.caption(
            "⚠️ This is a decision-support estimate, not a certified weather forecast. "
            "Combine it with guidance from your local extension officer."
        )

        # Save to history
        st.session_state.history.insert(0, {
            "Community":   community,
            "District":    district.replace("_", " ").title(),
            "Prediction":  label,
            "Confidence":  f"{top_prob*100:.1f}%",
            "Month":       pd.Timestamp(2000, prediction_month, 1).strftime("%B"),
            "Hour":        f"{prediction_hour}:00",
        })
        # Keep only last 5
        st.session_state.history = st.session_state.history[:5]

# -----------------------------------------------------------------------
# TAB 2 — BATCH PREDICTION
# -----------------------------------------------------------------------
with tab2:
    st.markdown('<p class="section-title">Upload Multiple Farmer Reports</p>', unsafe_allow_html=True)
    st.markdown(
        "Upload a CSV file with several farmer observations at once "
        "(e.g. all submissions from a community meeting)."
    )

    uploaded = st.file_uploader("📎 Choose a CSV file", type=["csv"])

    if uploaded:
        try:
            df_batch = pd.read_csv(uploaded)
        except Exception as e:
            st.error(f"Could not read that file.\n\nDetails: {e}")
            df_batch = None

        if df_batch is not None:
            st.success(f"✅ Loaded {len(df_batch):,} rows. Preview:")
            st.dataframe(df_batch.head(), use_container_width=True)

            run_clicked = st.button("🔮 Run Predictions on All Rows", type="primary", key="batch_predict")

            if run_clicked:
                with st.spinner(f"Predicting for {len(df_batch):,} records..."):
                    results = predict_batch(model, df_batch)
                    results["Irrigation Advisory"] = results["Label"].apply(
                        lambda lbl: TWI_ADVISORIES.get(lbl, irrigation_advisory(lbl)["message"])
                        if lang == "Twi" else irrigation_advisory(lbl)["message"]
                    )

                st.divider()
                st.markdown('<p class="section-title">Results</p>', unsafe_allow_html=True)

                most_common = results["Label"].mode()[0]
                needs_irrigation = (results["Label"] != "HEAVYRAIN").sum()

                # Metrics row
                m1, m2, m3 = st.columns(3)
                m1.metric("Total Records", f"{len(results):,}")
                m2.metric("Most Common Prediction", most_common)
                m3.metric("Records Needing Irrigation", f"{needs_irrigation:,}/{len(results):,}")

                # Results table
                st.dataframe(
                    results[["community", "district", "Label", "Irrigation Advisory"]
                            if "community" in results.columns
                            else ["Label", "Irrigation Advisory"]].head(20),
                    use_container_width=True,
                    hide_index=True,
                )

                # Distribution chart
                st.markdown("**Distribution of Predicted Rainfall Types**")
                fig2, ax2 = plt.subplots(figsize=(7, 4))
                counts = results["Label"].value_counts()
                bar_colors = [color_for_label(lbl) for lbl in counts.index]
                counts.plot(kind="bar", ax=ax2, color=bar_colors, edgecolor="white", linewidth=1.5)
                ax2.set_title("Rainfall Type Distribution", fontsize=13, fontweight='bold')
                ax2.set_xlabel("")
                ax2.set_ylabel("Number of Records")
                ax2.tick_params(axis="x", rotation=0)
                for i, v in enumerate(counts.values):
                    ax2.text(i, v + 0.5, str(v), ha='center', fontsize=11, fontweight='bold')
                plt.tight_layout()
                st.pyplot(fig2)
                plt.close()

                # Batch irrigation advisory (based on most common)
                st.markdown("**💧 Overall Irrigation Advisory**")
                st.caption(f"Based on the most common prediction across all records: **{most_common}**")
                batch_advisory = irrigation_advisory(most_common)
                if batch_advisory["level"] == "no_irrigation_needed":
                    st.success(batch_advisory["message"])
                elif batch_advisory["level"] == "irrigation_optional":
                    st.info(batch_advisory["message"])
                elif batch_advisory["level"] == "irrigation_recommended":
                    st.warning(batch_advisory["message"])
                else:
                    st.error(batch_advisory["message"])

                st.download_button(
                    "⬇️ Download Full Results (CSV)",
                    results.to_csv(index=False),
                    "rainfall_predictions.csv",
                    type="primary",
                )

                st.caption(
                    "⚠️ Decision-support estimate only. "
                    "Combine with guidance from your local extension officer."
                )
    else:
        st.info(
            "No file uploaded yet. Your CSV should include one row per observation "
            "with the same fields used in the Single Prediction tab."
        )

# -----------------------------------------------------------------------
# TAB 3 — PREDICTION HISTORY
# -----------------------------------------------------------------------
with tab3:
    st.markdown('<p class="section-title">🕐 Recent Predictions (This Session)</p>', unsafe_allow_html=True)

    if not st.session_state.history:
        st.info("No predictions made yet in this session. Go to the Single Prediction tab to get started.")
    else:
        for i, h in enumerate(st.session_state.history):
            color = color_for_label(h["Prediction"])
            icon  = icon_for_label(h["Prediction"])
            st.markdown(
                f"""
                <div class="history-card">
                    <span style="font-size:1.4em;">{icon}</span>
                    <strong style="color:{color}; margin-left:0.5em;">{h['Prediction']}</strong>
                    &nbsp;·&nbsp; {h['Community']}, {h['District']}
                    &nbsp;·&nbsp; {h['Month']} at {h['Hour']}
                    &nbsp;·&nbsp; Model confidence: {h['Confidence']}
                </div>
                """,
                unsafe_allow_html=True,
            )

        if st.button("🗑️ Clear History"):
            st.session_state.history = []
            st.rerun()

    st.caption("History is cleared when you close or refresh the app.")

