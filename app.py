import pandas as pd
import streamlit as st
import pickle
import numpy as np
import requests
import plotly.express as px
from io import StringIO
from dashboard_graphs import MaternalHealthDashboard
import warnings
from streamlit_option_menu import option_menu

# Load Models
maternal_model = pickle.load(
    open('final_maternal_model.pkl', 'rb')
)

fetal_model = pickle.load(
    open('final fetal health.pkl', 'rb')
)

with st.sidebar:
    st.title("🏥 Medicare Health services")

    st.markdown("---")

    st.success("AI Powered Healthcare Prediction")

    selected = option_menu(
        menu_title="Navigation",
        options=[
            "About Us",
            "Pregnancy Risk Prediction",
            "Fetal Health Prediction",
            "Dashboard"
        ],
        icons=[
            "house",
            "heart-pulse",
            "activity",
            "bar-chart"
        ],
        menu_icon="hospital",
        default_index=0
    )
    st.info(
        """
        ### Features
        ✅ Maternal Risk Prediction

        ✅ Fetal Health Prediction

        ✅ Interactive Dashboard

        ✅ ML-Based Decision Support
        """
    )

if selected == "About Us":
    st.title("🏥 Medicare health")
    st.subheader("AI-Powered Maternal & Fetal Health Prediction System")

    st.image(
        "https://images.unsplash.com/photo-1576091160399-112ba8d25d1d",
        use_container_width=True)

    st.markdown("---")

    st.write("""
     Medicare health is an intelligent healthcare platform developed
     using Machine Learning algorithms to assist in maternal
     and fetal health assessment during pregnancy.

     The system helps identify potential health risks at an
     early stage and supports healthcare decision-making through
     predictive analytics and data-driven insights.
     """)

    st.markdown("---")

    st.header("🎯 Core Features")

    col1, col2 = st.columns(2)

    with col1:
        st.image(
            "https://cdn-icons-png.flaticon.com/512/3209/3209265.png",
            width=100
        )

        st.subheader("Maternal Risk Prediction")

        st.write("""
         Predicts pregnancy risk levels using:

         • Age

         • Blood Pressure

         • Blood Sugar

         • Body Temperature

         • Heart Rate

         • Haemoglobin

         • Gestational Age

         • Previous Complications
         """)

    with col2:
        st.image(
            "https://cdn-icons-png.flaticon.com/512/2785/2785819.png",
            width=100
        )

        st.subheader("Fetal Health Prediction")

        st.write("""
         Evaluates fetal condition using:

         • Baseline Fetal Heart Rate

         • Accelerations

         • Fetal Movements

         • Uterine Contractions

         • Variability Measurements

         • Decelerations

         • Histogram Features
         """)

    st.markdown("---")

    st.header("🤖 Machine Learning Models")

    st.write("""
     The system compares multiple machine learning models:

     ✔ Logistic Regression

     ✔ Decision Tree

     ✔ Random Forest

     ✔ K-Nearest Neighbors

     ✔ Support Vector Machine

     ✔ Gradient Boosting

     ✔ XGBoost

     ✔ Artificial Neural Network
     """)

    st.success(
        "🏆 Best Performing Model: XGBoost (Highest Accuracy)"
    )

    st.markdown("---")

    st.header("📊 Dashboard Features")

    st.write("""
     • Interactive Data Visualizations

     • Risk Distribution Analysis

     • Feature Importance Analysis

     • Health Trends Monitoring

     • Model Performance Comparison

     • Clinical Decision Support
     """)

    st.markdown("---")

    st.header("👨‍💻 Developer")

    st.info("""
     Harsh Singh

     B.Tech CSE (Machine Learning)

     Gautam Buddha University

     Skills:
     Python | Machine Learning | Deep Learning |
     Data Science | NLP | Predictive Analytics
     """)

    st.markdown("---")

    st.warning(
        "⚠️ This application is intended for educational and "
        "research purposes only and should not replace "
        "professional medical consultation."
    )

if selected == 'Pregnancy Risk Prediction':

    st.title('🤰 Pregnancy Risk Prediction')
    content = "Predicting risk in pregnancy involves analyzing maternal health parameters, tracking vital baselines, and identifying underlying complications. By evaluating these parameters, we can assess potential health flags and provide automated predictive risk insights."
    st.markdown(f"<div style='white-space: pre-wrap;'><b>{content}</b></div></br>", unsafe_allow_html=True)

    # ─── MEDICAL TERMS EXPLANATION GUIDE ───
    with st.expander("📘 Understand the Medical Terms (Plain English Guide)"):
        st.markdown("""
        * **Systolic BP (Top Number):** Measures the pressure in your blood vessels when your heart beats.
        * **Diastolic BP (Bottom Number):** Measures the pressure in your blood vessels when your heart rests between beats.
        * **Blood Glucose:** Your blood sugar level. Higher trends can indicate gestational diabetes risk.
        * **Parity:** The total number of times you have given birth before.
        * **Pulse Pressure & MAP:** Calculated metrics that show your overall cardiovascular and heart health.
        """)

    st.markdown("---")

    # Load your preprocessing scaler asset safely
    maternal_model = pickle.load(open("final_maternal_model.pkl", "rb"))


    # ─── THREE-COLUMN USER INPUTS FOR ALL 11 MANUALLY ENTERED ATTRIBUTES ───
    col1, col2, col3 = st.columns(3)

    with col1:
        age = st.number_input('Age of the Person', min_value=15, max_value=50, value=25)
        bodyTemp = st.number_input('Body Temperature (°F)', min_value=95.0, max_value=105.0, value=98.6)
        parity = st.number_input('Parity (Previous Births)', min_value=0, max_value=15, value=0)

    with col2:
        systolic = st.number_input('Systolic BP (Top Number in mmHg)', min_value=80, max_value=200, value=120)
        heartRate = st.number_input('Heart Rate (Beats per Minute)', min_value=40, max_value=180, value=80)
        prev_comp = st.selectbox('Previous Complications', ["None", "Yes - Minor", "Yes - Major"])

    with col3:
        diastolic = st.number_input('Diastolic BP (Bottom Number in mmHg)', min_value=50, max_value=150, value=80)
        BS = st.number_input('Blood Glucose (mmol/L)', min_value=4.0, max_value=20.0, value=6.0)
        gest_age = st.number_input('Gestational Age (Weeks)', min_value=1, max_value=42, value=20)

    # Placing remaining layout attributes lower down for visual clarity
    sub_col1, sub_col2 = st.columns(2)
    with sub_col1:
        distance = st.number_input('Distance to Nearest Hospital (km)', min_value=0.0, max_value=100.0, value=5.0)
    with sub_col2:
        weight_gain = st.number_input('Total Weight Gain So Far (kg)', min_value=0.0, max_value=40.0, value=5.0)

    # ─── SCHEMATIC ENCODING & ENGINEERED FEATURES ───
    comp_mapped = 0 if prev_comp == "None" else (1 if prev_comp == "Yes - Minor" else 2)

    # Deriving the 3 missing engineered features your model requires:
    pulse_pressure = systolic - diastolic
    map_value = (systolic + 2 * diastolic) / 3

    risk_score = 0
    if systolic > 140: risk_score += 2
    if BS > 8: risk_score += 2
    if bodyTemp > 100: risk_score += 1
    if age > 35: risk_score += 1

    st.markdown("###")

    # ─── BUTTON INTERACTIONS SECTION ───
    btn_col1, btn_col2 = st.columns([1, 4])

    with btn_col1:
        predict_clicked = st.button('Predict Risk')
    with btn_col2:
        clear_clicked = st.button("Clear Inputs")

    if clear_clicked:
        st.rerun()

    if predict_clicked:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")

            # ─── CONSTRUCT ARRAY IN THE EXACT EXPECTED TRAINING FEATURE ORDER ───
            # Order: Age, SystolicBP, DiastolicBP, BS, BodyTemp, HeartRate, GestationalAge, Parity,
            #        PreviousComplications, DistanceToHospital, WeightGain, PulsePressure, MAP, RiskScore

            input_df = pd.DataFrame({
                'Age': [age],
                'SystolicBP': [systolic],
                'DiastolicBP': [diastolic],
                'BS': [BS],
                'BodyTemp': [bodyTemp],
                'HeartRate': [heartRate],
                'GestationalAge': [gest_age],
                'Parity': [parity],
                'PreviousComplications': [comp_mapped],
                'DistanceToHospital': [distance],
                'WeightGain': [weight_gain],
                'PulsePressure': [pulse_pressure],
                'MAP': [map_value],
                'RiskScore': [risk_score]
            })

            # Scale and run through prediction model matrix
            predicted_risk = maternal_model.predict(input_df)
            st.write("Prediction Value:", int(predicted_risk[0]))
            st.write(input_df)


            # Convert numeric prediction to original label
            predicted_risk = maternal_model.predict(input_df)

            pred = int(predicted_risk[0])

            st.write("Prediction Value:", pred)

            probs = maternal_model.predict_proba(input_df)[0]

            st.write("Low Risk Probability :", round(probs[2] * 100, 2), "%")
            st.write("Mid Risk Probability :", round(probs[1] * 100, 2), "%")
            st.write("High Risk Probability:", round(probs[0] * 100, 2), "%")


        # ─── CONDITIONAL DYNAMIC PATIENT OUTPUT DISPLAY ───

            st.markdown("---")
            st.subheader("Assessment Result:")

            pred = int(predicted_risk[0])

            if pred == 2:

                st.success("🟢 Low Risk")

                st.write("""
                • Continue regular prenatal checkups.
                • Maintain a balanced diet.
                • Stay physically active.
                • Monitor weight gain regularly.
                """)

            elif pred == 1:

                st.warning("🟡 Mid Risk")

                st.write("""
                • Additional monitoring recommended.
                • Check blood pressure regularly.
                • Monitor blood sugar levels.
                • Follow doctor's advice closely.
                """)

            else:

                st.error("🔴 High Risk")

                st.write("""
                • Immediate medical consultation recommended.
                • Frequent monitoring required.
                • Follow specialist advice strictly.
                • Do not ignore warning symptoms.
                """)

if selected == "Fetal Health Prediction":

    st.title("👶 Fetal Health Prediction")

    content = """
    Cardiotocography (CTG) is a non-invasive monitoring technique used to assess fetal well-being.
    This predictive system analyzes fetal heart rate patterns and uterine activity to classify
    fetal health as Normal, Suspect, or Pathological.
    """

    st.markdown(
        f"<div style='white-space: pre-wrap;'><b>{content}</b></div>",
        unsafe_allow_html=True
    )

    st.markdown("---")

    with st.expander("📘 Understand the CTG Parameters"):
        st.markdown("""
        • **Baseline Value:** Average fetal heart rate.

        • **Accelerations:** Temporary increases in fetal heart rate.

        • **Fetal Movement:** Number of fetal movements detected.

        • **Uterine Contractions:** Frequency of contractions.

        • **Decelerations:** Temporary decreases in fetal heart rate.

        • **Histogram Features:** Statistical measures extracted from CTG recordings.
        """)

    st.markdown("### Enter Fetal Monitoring Values")

    col1, col2, col3 = st.columns(3)

    with col1:
        BaselineValue = st.number_input("Baseline Value", value=120.0)
        uterine_contractions = st.number_input("Uterine Contractions", value=0.0)
        abnormal_short_term_variability = st.number_input(
            "Abnormal Short-Term Variability (%)", value=50.0
        )
        histogram_min = st.number_input("Histogram Min", value=50.0)
        histogram_number_of_zeroes = st.number_input(
            "Histogram Number Of Zeroes", value=0.0
        )
        histogram_median = st.number_input("Histogram Median", value=120.0)

    with col2:
        Accelerations = st.number_input("Accelerations", value=0.0)
        light_decelerations = st.number_input("Light Decelerations", value=0.0)
        mean_value_of_short_term_variability = st.number_input(
            "Mean Short-Term Variability", value=1.0
        )
        histogram_max = st.number_input("Histogram Max", value=180.0)
        histogram_mode = st.number_input("Histogram Mode", value=120.0)
        histogram_variance = st.number_input("Histogram Variance", value=10.0)

    with col3:
        fetal_movement = st.number_input("Fetal Movement", value=0.0)
        severe_decelerations = st.number_input("Severe Decelerations", value=0.0)
        prolongued_decelerations = st.number_input(
            "Prolongued Decelerations", value=0.0
        )
        percentage_of_time_with_abnormal_long_term_variability = st.number_input(
            "% Time With ALTV", value=10.0
        )
        mean_value_of_long_term_variability = st.number_input(
            "Mean Long-Term Variability", value=8.0
        )
        histogram_width = st.number_input("Histogram Width", value=70.0)

    st.markdown("---")

    colA, colB = st.columns([1, 4])

    with colA:
        predict_btn = st.button("🔍 Predict")

    with colB:
        clear_btn = st.button("🗑 Clear")

    if clear_btn:
        st.rerun()

    if predict_btn:

        try:

            input_df = pd.DataFrame({
                'baseline value':[BaselineValue],
                'accelerations':[Accelerations],
                'fetal_movement':[fetal_movement],
                'uterine_contractions':[uterine_contractions],
                'light_decelerations':[light_decelerations],
                'severe_decelerations':[severe_decelerations],
                'prolongued_decelerations':[prolongued_decelerations],
                'abnormal_short_term_variability':[abnormal_short_term_variability],
                'mean_value_of_short_term_variability':[mean_value_of_short_term_variability],
                'percentage_of_time_with_abnormal_long_term_variability':
                    [percentage_of_time_with_abnormal_long_term_variability],
                'mean_value_of_long_term_variability':
                    [mean_value_of_long_term_variability],
                'histogram_width':[histogram_width],
                'histogram_min':[histogram_min],
                'histogram_max':[histogram_max],
                'histogram_number_of_peaks':[0],
                'histogram_number_of_zeroes':[histogram_number_of_zeroes],
                'histogram_mode':[histogram_mode],
                'histogram_mean':[histogram_mode],
                'histogram_median':[histogram_median],
                'histogram_variance':[histogram_variance],
                'histogram_tendency':[0]
            })

            prediction = fetal_model.predict(input_df)

            pred = int(prediction[0])

            st.markdown("---")
            st.subheader("Assessment Result")

            if pred == 0:

                st.success("🟢 Normal Fetal Health")

                st.write("""
                • Fetal condition appears normal.

                • Continue routine prenatal monitoring.

                • Maintain scheduled clinical visits.
                """)

            elif pred == 1:

                st.warning("🟡 Suspect Fetal Health")

                st.write("""
                • Additional observation recommended.

                • Repeat CTG monitoring.

                • Consult healthcare provider.
                """)

            else:

                st.error("🔴 Pathological Fetal Health")

                st.write("""
                • Immediate medical evaluation advised.

                • Specialist consultation recommended.

                • Continuous monitoring may be required.
                """)

        except Exception as e:

            st.error(f"Prediction Error: {e}")

if selected == "Dashboard":
    api_key = "579b464db66ec23bdd0000018c8a5d87e40f422346dc83ec15756816"
    api_endpoint = f"https://api.data.gov.in/resource/6d6a373a-4529-43e0-9cff-f39aa8aa5957?api-key={api_key}&format=csv"

    st.header("Dashboard")

    # 1. Initialize the dashboard object safely
    dashboard = MaternalHealthDashboard(api_endpoint)

 # 2. Add a structural data check before building UI components
    if dashboard.maternal_health_data is not None:
        content = "Our interactive dashboard offers a comprehensive visual representation of maternal health achievements across diverse regions..."
        st.markdown(f"<div style='white-space: pre-wrap;'><b>{content}</b></div></br>", unsafe_allow_html=True)

        # Render charts safely
        dashboard.create_bubble_chart()
        with st.expander("Show More"):
            bubble_content = dashboard.get_bubble_chart_data()
            st.markdown(f"<div style='white-space: pre-wrap;'><b>{bubble_content}</b></div>", unsafe_allow_html=True)

        dashboard.create_pie_chart()
        with st.expander("Show More"):
            pie_content = dashboard.get_pie_graph_data()
            st.markdown(f"<div style='white-space: pre-wrap;'><b>{pie_content}</b></div>", unsafe_allow_html=True)
    else:
        st.error(
            "⚠️ Unable to load dashboard components because the external government API data could not be retrieved. Please check your internet connection or API key limits.")
