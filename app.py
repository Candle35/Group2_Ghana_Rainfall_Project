import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
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
st.markdown(
    """
    <style>
    /* Bigger base font for readability outdoors / on small screens */
    html, body, [class*="css"] { font-size: 17px; }

    /* Big, easy-to-tap buttons */
    div.stButton > button {
        width: 100%;
        padding: 0.9em 1em;
        font-size: 1.1em;
        font-weight: 600;
        border-radius: 10px;
    }

    /* Prediction result card */
    .result-card {
        padding: 1.5em;
        border-radius: 14px;
        text-align: center;
        margin-top: 0.5em;
        margin-bottom: 1em;
    }
    .result-label {
        font-size: 2em;
        font-weight: 800;
        margin: 0;
    }

    /* Section headers */
    .section-title {
        font-size: 1.3em;
        font-weight: 700;
        margin-top: 0.2em;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# -----------------------------------------------------------------------
# HEADER
# -----------------------------------------------------------------------
st.title("🌦️ Ghana Rainfall Predictor")
st.markdown(
    "**A simple tool for farmers and extension officers in the Pra River Basin** "
    "to predict likely rainfall type using local weather signs (indigenous "
    "ecological indicators) alongside forecast data."
)

with st.sidebar:
    st.header("ℹ️ How to use this app")
    st.markdown(
        """
        **Step 1.** Choose *Single Prediction* to check one farmer's report,
        or *Batch Prediction* to upload many reports at once (CSV file).

        **Step 2.** Fill in the details as accurately as you can. If you are
        unsure about a value, use your best estimate — the tool still works.

        **Step 3.** Press the big **Predict** button to see the result.

        ---
        **Need help?**
        Contact your local agricultural extension officer, or reach out to
        the project team supporting this tool.
        """
    )
    st.divider()
    st.caption("Built to support smallholder farmers with rainfall planning.")

# -----------------------------------------------------------------------
# LOAD MODEL
# -----------------------------------------------------------------------
@st.cache_resource
def get_model():
    if not os.path.exists(MODEL_PATH):
        st.error(
            "⚠️ The prediction model could not be found.\n\n"
            "If you are the app administrator, please run `python src/train.py` "
            "first to train and save the model."
        )
        st.stop()
    return load_model()

model = get_model()

# Simple color mapping so results feel intuitive at a glance.
# Falls back gracefully if TARGET_CLASSES has more/fewer categories.
RESULT_COLORS = ["#1976D2", "#2E7D32", "#F9A825", "#C62828", "#6A1B9A"]
RESULT_ICONS = ["🌦️", "🌧️", "⛅", "☀️", "⛈️"]

def irrigation_advisory(predicted_category: str) -> dict:
    """
    Maps a predicted rainfall category to an irrigation advisory.
    Handles labels like "MEDIUMRAIN", "Medium Rain", "medium rain", etc.
    """
    category = predicted_category.strip().lower().replace(" ", "")

    if category == "heavyrain":
        return {
            "level": "no_irrigation_needed",
            "message": "Heavy rainfall predicted. Irrigation not necessary this period.",
        }
    elif category == "mediumrain":
        return {
            "level": "irrigation_optional",
            "message": "Medium rainfall predicted. Irrigation may help but is not critical.",
        }
    elif category == "lightrain":
        return {
            "level": "irrigation_recommended",
            "message": "Light rainfall predicted. Irrigation recommended to supplement crop water needs.",
        }
    elif category == "norain":
        return {
            "level": "irrigation_strongly_recommended",
            "message": "No rainfall predicted. Irrigation strongly recommended.",
        }
    elif category == "smallrain":
        return {
            "level": "irrigation_recommended",
            "message": "Small rainfall predicted. Irrigation recommended to ensure adequate water supply.",
        }
    else:
        return {
            "level": "unknown",
            "message": f"Unrecognized rainfall category ('{predicted_category}'). Unable to provide irrigation advice.",
        }

def color_for_label(label: str) -> str:
    labels = list(TARGET_CLASSES) if hasattr(TARGET_CLASSES, "__iter__") else []
    if label in labels:
        return RESULT_COLORS[labels.index(label) % len(RESULT_COLORS)]
    return "#1976D2"


def icon_for_label(label: str) -> str:
    labels = list(TARGET_CLASSES) if hasattr(TARGET_CLASSES, "__iter__") else []
    if label in labels:
        return RESULT_ICONS[labels.index(label) % len(RESULT_ICONS)]
    return "🌦️"


# -----------------------------------------------------------------------
# TABS
# -----------------------------------------------------------------------
tab1, tab2 = st.tabs(["👤 Single Prediction", "📄 Batch Prediction (CSV)"])

# -----------------------------------------------------------------------
# TAB 1 — SINGLE PREDICTION
# -----------------------------------------------------------------------
with tab1:
    st.markdown('<p class="section-title">Farmer & Community Details</p>', unsafe_allow_html=True)
    col1, col2 = st.columns(2)

    with col1:
        user_id = st.number_input(
            "Farmer ID", min_value=1, max_value=1000, value=11,
            help="The unique number assigned to this farmer's records."
        )
        community = st.text_input(
            "Community / Village", "Akwaduuso",
            help="The name of the town or village where the observation was made."
        )
        district = st.text_input(
            "District", "Assin South",
            help="The administrative district the community belongs to."
        )

    with col2:
        prediction_month = st.selectbox(
            "Month of Observation",
            list(range(1, 13)),
            format_func=lambda m: pd.Timestamp(2000, m, 1).strftime("%B"),
            help="Which month was this observation made in?"
        )
        prediction_hour = st.slider(
            "Time of Day (Hour)", 0, 23, 8,
            help="What hour of the day was the observation made? (0 = midnight, 12 = noon)"
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
            "How confident is the farmer in this forecast?",
            options=[0.3, 0.6, 1.0],
            value=0.6,
            format_func=lambda x: {0.3: "Low", 0.6: "Medium", 1.0: "High"}[x],
            help="Based on the farmer's own judgement of the local signs observed."
        )
    with col4:
        predicted_intensity = st.slider(
            "Expected Rainfall Intensity (0 = none, 10 = very heavy)",
            0.0, 10.0, 2.0, step=0.5,
        )

    st.write("")  # spacing
    predict_clicked = st.button("🔮 Predict Rainfall Type", type="primary")

    if predict_clicked:
        with st.spinner("Analyzing local indicators and forecast data..."):
            input_data = {
                "user_id": user_id,
                "confidence": confidence,
                "predicted_intensity": predicted_intensity,
                "community": community,
                "district": district,
                "forecast_length": forecast_length,
                "prediction_hour": prediction_hour,
                "prediction_month": prediction_month,
                "prediction_time": pd.Timestamp.now(),
            }
            result = predict_single(model, input_data)

        st.divider()

        label = result["label"]
        color = color_for_label(label)
        icon = icon_for_label(label)

        st.markdown(
            f"""
            <div class="result-card" style="background-color:{color}22; border: 2px solid {color};">
                <div style="font-size:2.5em;">{icon}</div>
                <p class="result-label" style="color:{color};">{label}</p>
                <p style="color:#555;">Predicted rainfall type for {community}, {district}</p>
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.markdown("**How confident is the model in each possible outcome?**")
        prob_df = pd.DataFrame(
            list(result["probabilities"].items()), columns=["Rainfall Type", "Likelihood"]
        ).sort_values("Likelihood", ascending=False)
        prob_df["Likelihood"] = (prob_df["Likelihood"] * 100).round(1)

        st.dataframe(
            prob_df,
            column_config={
                "Likelihood": st.column_config.ProgressColumn(
                    "Likelihood (%)", min_value=0, max_value=100, format="%.1f%%"
                )
            },
            hide_index=True,
            use_container_width=True,
        )

        # ---- Irrigation advisory ----
        advisory = irrigation_advisory(label)
        st.markdown('<p class="section-title">💧 Irrigation Advisory</p>', unsafe_allow_html=True)
        if advisory["level"] == "no_irrigation_needed":
            st.success(advisory["message"])
        elif advisory["level"] == "irrigation_optional":
            st.info(advisory["message"])
        elif advisory["level"] == "irrigation_recommended":
            st.warning(advisory["message"])
        else:
            st.error(advisory["message"])

        st.caption(
            "⚠️ This is a decision-support estimate, not a certified weather forecast. "
            "Please combine it with guidance from your local extension officer."
        )

# -----------------------------------------------------------------------
# TAB 2 — BATCH PREDICTION
# -----------------------------------------------------------------------
with tab2:
    st.markdown('<p class="section-title">Upload Multiple Farmer Reports</p>', unsafe_allow_html=True)
    st.markdown(
        "Upload a CSV file containing several farmer observations at once "
        "(for example, all submissions collected during a community meeting)."
    )

    uploaded = st.file_uploader("📎 Choose a CSV file", type=["csv"])

    if uploaded:
        try:
            df = pd.read_csv(uploaded)
        except Exception as e:
            st.error(f"We couldn't read that file. Please check it is a valid CSV.\n\nDetails: {e}")
            df = None

        if df is not None:
            st.success(f"✅ Loaded {len(df)} rows. Here's a preview:")
            st.dataframe(df.head(), use_container_width=True)

            run_clicked = st.button("🔮 Run Predictions on All Rows", type="primary")

            if run_clicked:
                with st.spinner(f"Predicting rainfall type for {len(df)} records..."):
                    results = predict_batch(model, df)

                st.divider()
                st.markdown('<p class="section-title">Results</p>', unsafe_allow_html=True)

                col_a, col_b = st.columns([2, 1])
                with col_a:
                    st.dataframe(
                        results[["Prediction", "Label"]].head(20),
                        use_container_width=True,
                        hide_index=True,
                    )
                with col_b:
                    st.metric("Total Records Processed", len(results))
                    most_common = results["Label"].mode()[0]
                    st.metric("Most Common Prediction", most_common)

                st.markdown("**Distribution of Predicted Rainfall Types**")
                fig, ax = plt.subplots(figsize=(6, 4))
                counts = results["Label"].value_counts()
                bar_colors = [color_for_label(lbl) for lbl in counts.index]
                counts.plot(kind="bar", ax=ax, color=bar_colors, edgecolor="black")
                ax.set_title("Rainfall Type Distribution")
                ax.set_xlabel("")
                ax.set_ylabel("Number of Records")
                ax.tick_params(axis="x", rotation=0)
                plt.tight_layout()
                st.pyplot(fig)
                plt.close()

                st.download_button(
                    "⬇️ Download Full Results (CSV)",
                    results.to_csv(index=False),
                    "rainfall_predictions.csv",
                    type="primary",
                )
    else:
        st.info(
            "No file uploaded yet. Your CSV should include one row per observation, "
            "with the same fields used in the Single Prediction tab "
            "(e.g. community, district, prediction_month, prediction_hour, etc.)."
        )
