# it is a magic word which write the app.py file (which counts from here)

import streamlit as st
import pandas as pd
import plotly.express as px # for graphical visuals
import plotly.graph_objects as go # for graphical features
import numpy as np
import tensorflow as tf
#from tensorflow.keras.models import load_model
import numpy as np
import cv2
from PIL import Image
import pickle
import joblib
import os
import gdown

MODEL_FILES = {
    "individual_model_tree_model.pkl": "10pyik0KkI0e9PVtW19FkmnPSgJkwBJfi",
    "individual_model_preprocessor.pkl": "1n1dxtSM0ZcVmb565QWjGSaOMzM_89-5c",
    "Vehicle_model_tree_model.pkl": "13Aywt-fM8K6n8AdKGlOWCbW89W4KvJw6",
    "Vehicle_model_preprocessor.pkl": "1_fZ__caRCFRakam0fR0-flzxSYwzJMaD",
    "industry_model_tree_model.pkl": "1dpfwxdndv2pfxMNCOr34qIcEgVfBwo5i",
    "industry_model_preprocessor.pkl": "1v-8__qYcEZocPAZyepPBTsYJiEZdNru7",
    "scope3_model_tree_model.pkl": "1EevUJVqzccOLOrysFKcraYRs3z5i7w3n",
    "scope3_model_preprocessor.pkl": "1SO_G0uJllqELzjstuCHA9pnU_XAWZwKu",
    "carbon_emission_model.keras": "1pkwr5NZL1uJBHiEEs9v7LH4Fz_Raqs4o",
    "class_names.pkl": "1hV5Z_FKoJfi9sMcu1DU_pkU6dqRH_xZk"
}

@st.cache_resource
def download_all_models():
    for filename, file_id in MODEL_FILES.items():
        if not os.path.exists(filename):
            url = f"https://drive.google.com/uc?id={file_id}"
            st.write(f"Downloading {filename}...")
            gdown.download(url, filename, quiet=False)
    return True

download_all_models()

@st.cache_data
def load_data():
    url = "https://raw.githubusercontent.com/owid/co2-data/master/owid-co2-data.csv"
    df = pd.read_csv(url)

    last_year = df['year'].max()

    if last_year < 2026:
        future_years = []
        for y in range(last_year + 1, 2027):
            temp = df[df['year'] == last_year].copy()
            temp['year'] = y
            temp['co2'] = temp['co2'] * (1 + np.random.uniform(-0.005, 0.01))
            future_years.append(temp)

        df = pd.concat([df] + future_years, ignore_index=True)

    return df

df = load_data()
if "zoomed_chart" not in st.session_state:
    st.session_state.zoomed_chart = None

@st.cache_resource
def load_individual_model():
    model = joblib.load("individual_model_tree_model.pkl")
    preprocessor = joblib.load("individual_model_preprocessor.pkl")
    return model, preprocessor


@st.cache_resource
def load_transport_model():
    model = joblib.load("Vehicle_model_tree_model.pkl")
    preprocessor = joblib.load("Vehicle_model_preprocessor.pkl")
    return model, preprocessor


@st.cache_resource
def load_industry_model():
    model = joblib.load("industry_model_tree_model.pkl")
    preprocessor = joblib.load("industry_model_preprocessor.pkl")
    return model, preprocessor


@st.cache_resource
def load_scope3_model():
    model = joblib.load("scope3_model_tree_model.pkl")
    preprocessor = joblib.load("scope3_model_preprocessor.pkl")
    return model, preprocessor

@st.cache_resource
def load_cnn_model():
    cnn_model = tf.keras.models.load_model("carbon_emission_model.keras")

    with open("class_names.pkl", "rb") as f:
        class_names = pickle.load(f)

    return cnn_model, class_names


cnn_model, original_class_names = load_cnn_model()


# Setting page configuration

st.set_page_config(page_title="EcoTerraMatrix-X", layout="wide")

# Customizing CSS for the Pop Animation with brighter text colors

st.markdown("""
    <style>
    @keyframes pop {
        0% { transform: scale(1); }
        30% { transform: scale(1.4); }
        60% { transform: scale(1); }
        100% { transform: scale(1); }
    }
    .pop-container {
        display: flex;
        justify-content: center;
        align-items: center;
        margin-top: -140px;
        position: relative;
        z-index: 10;
        pointer-events: none;
    }
    .pop-text {
        display: inline-block;
        animation: pop 0.6s ease-in-out 3;
        font-weight: 900;
        font-size: 3.8rem;
        font-family: 'Arial Black', sans-serif;
    }
    /* Super Bright Neon Colors for High Visibility */
    .high-emission { color: #FF0000; text-shadow: 0px 0px 10px rgba(255,0,0,0.2); }
    .medium-emission { color: #FF8C00; text-shadow: 0px 0px 10px rgba(255,140,0,0.2); }
    .low-emission { color: #00E676; text-shadow: 0px 0px 10px rgba(0,230,118,0.2); }
    </style>
    """, unsafe_allow_html=True)


continent_map = {
    "Africa": ["Algeria", "Angola", "Benin", "Botswana", "Burkina Faso", "Burundi", "Cameroon", "Cape Verde", "Central African Republic", "Chad", "Comoros", "Congo", "Democratic Republic of Congo", "Djibouti", "Egypt", "Equatorial Guinea", "Eritrea", "Eswatini", "Ethiopia", "Gabon", "Gambia", "Ghana", "Guinea", "Guinea-Bissau", "Ivory Coast", "Kenya", "Lesotho", "Liberia", "Libya", "Madagascar", "Malawi", "Mali", "Mauritania", "Mauritius", "Morocco", "Mozambique", "Namibia", "Niger", "Nigeria", "Rwanda", "Sao Tome and Principe", "Senegal", "Seychelles", "Sierra Leone", "Somalia", "South Africa", "South Sudan", "Sudan", "Tanzania", "Togo", "Tunisia", "Uganda", "Zambia", "Zimbabwe"],
    "Asia": ["Afghanistan", "Armenia", "Azerbaijan", "Bahrain", "Bangladesh", "Bhutan", "Brunei", "Cambodia", "China", "Georgia", "India", "Indonesia", "Iran", "Iraq", "Israel", "Japan", "Jordan", "Kazakhstan", "Kuwait", "Kyrgyzstan", "Laos", "Lebanon", "Malaysia", "Maldives", "Mongolia", "Myanmar", "Nepal", "North Korea", "Oman", "Pakistan", "Palestine", "Philippines", "Qatar", "Saudi Arabia", "Singapore", "South Korea", "Sri Lanka", "Syria", "Taiwan", "Tajikistan", "Thailand", "Timor", "Turkey", "Turkmenistan", "United Arab Emirates", "Uzbekistan", "Vietnam", "Yemen"],
    "Europe": ["Albania", "Andorra", "Austria", "Belarus", "Belgium", "Bosnia and Herzegovina", "Bulgaria", "Croatia", "Cyprus", "Czechia", "Denmark", "Estonia", "Finland", "France", "Germany", "Greece", "Hungary", "Iceland", "Ireland", "Italy", "Kosovo", "Latvia", "Liechtenstein", "Lithuania", "Luxembourg", "Malta", "Moldova", "Monaco", "Montenegro", "Netherlands", "North Macedonia", "Norway", "Poland", "Portugal", "Romania", "Russia", "San Marino", "Serbia", "Slovakia", "Slovenia", "Spain", "Sweden", "Switzerland", "Ukraine", "United Kingdom", "Vatican"],
    "North America": ["Antigua and Barbuda", "Bahamas", "Barbados", "Belize", "Canada", "Costa Rica", "Cuba", "Dominica", "Dominican Republic", "El Salvador", "Grenada", "Guatemala", "Haiti", "Honduras", "Jamaica", "Mexico", "Nicaragua", "Panama", "Saint Kitts and Nevis", "Saint Lucia", "Saint Vincent and the Grenadines", "Trinidad and Tobago", "United States"],
    "South America": ["Argentina", "Bolivia", "Brazil", "Chile", "Colombia", "Ecuador", "Guyana", "Paraguay", "Peru", "Suriname", "Uruguay", "Venezuela"],
    "Oceania": ["Australia", "Fiji", "Kiribati", "Marshall Islands", "Micronesia", "Nauru", "New Zealand", "Palau", "Papua New Guinea", "Samoa", "Solomon Islands", "Tonga", "Tuvalu", "Vanuatu"]
}

#  SIDEBAR

st.sidebar.title(" Dashboard Controls")
selected_year = st.sidebar.slider("Select Year", 1950, 2026, 2026)
selected_continent = st.sidebar.selectbox("Select Continent", list(continent_map.keys()))
selected_country = st.sidebar.selectbox("Select Country", sorted(continent_map[selected_continent]))

if st.sidebar.button("Reset All Charts"):
    st.session_state.zoomed_chart = None
    st.rerun()

# CALCULATIONS

year_df = df[df["year"] == selected_year]
world_total = year_df[year_df["country"] == "World"]["co2"].iloc[0] if not year_df[year_df["country"] == "World"].empty else 1
cont_val = year_df[year_df["country"] == selected_continent]["co2"].iloc[0] if not year_df[year_df["country"] == selected_continent].empty else 0
global_share_pct = (cont_val / world_total) * 100

status_class = "high-emission" if global_share_pct > 20 else ("medium-emission" if global_share_pct > 5 else "low-emission")

# HEADER

st.markdown(
    "<h1 style='text-align: left;'> EcoTerraMatrix-X </h1>",
    unsafe_allow_html=True
)

st.markdown(
    "<h5 style='text-align: center;'> The EcoTerraMatrix-X dashboard is a comprehensive carbon tracking platform that visualizes global CO2 analytics through continental emission gauges and interactive distribution maps. It features multi-sector calculators for individual , transport , industry , and Scope 3  footprints. Additionally, an AI-powered computer vision tool classifies emission levels from uploaded city images , all manageable via centralized geographic and temporal controls. </h5>",
    unsafe_allow_html=True
)

st.markdown(
    f"<h3 style='text-align: center;'>🌍 {selected_continent} Emission Share</h3>",
    unsafe_allow_html=True
)

# BRIGHTER ADAPTIVE GAUGE

fig_gauge = go.Figure(go.Indicator(
    mode = "gauge",
    value = global_share_pct,
    gauge = {
        'axis': {'range': [0, 100], 'visible': False},
        'bar': {'color': "#2C3E50", 'thickness': 0.18},
        'bgcolor': "rgba(0,0,0,0)",
        'steps': [
            {'range': [0, 33], 'color': "rgba(0, 255, 127, 0.95)"},  # Brighter Spring Green

            {'range': [33, 66], 'color': "rgba(255, 215, 0, 0.95)"}, # Brighter Gold

            {'range': [66, 100], 'color': "rgba(255, 30, 30, 0.95)"} # Brighter Pure Red
        ],
    }
))

fig_gauge.update_layout(
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    height=350,
    margin=dict(l=50, r=50, t=20, b=100)
)
st.plotly_chart(fig_gauge, use_container_width=True)

# POPPING PERCENTAGE

st.markdown(f"""
    <div class="pop-container">
        <div class="pop-text {status_class}">
            {global_share_pct:.1f}%
        </div>
    </div>
    """, unsafe_allow_html=True)

# LOWER INTERACTIVE GRAPHS

st.markdown("<br><br>", unsafe_allow_html=True)

def render_chart(chart_type):
    if chart_type == "continent":
        data = year_df[year_df["country"].isin(continent_map.keys())]
        fig = px.bar(data, x="country", y="co2", color="country")
    elif chart_type == "top_countries":
        data = year_df[year_df["country"].isin(continent_map[selected_continent])].sort_values("co2")
        fig = px.bar(data, x="co2", y="country", orientation='h')
    else:
        data = df[df["country"] == selected_country]
        fig = px.area(data, x="year", y="co2")
        fig.add_vline(x=selected_year, line_dash="dash", line_color="red")

    height = 550 if st.session_state.zoomed_chart == chart_type else 300

    fig.update_layout(
        height=height,
        template="streamlit",
        transition_duration=800,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)"
    )
    return fig

# ------------------ WORLD MAP  -------------------------------

st.markdown("<br>", unsafe_allow_html=True)

# Filter for real countries and calculate percentage

map_df = year_df[year_df["iso_code"].notna()].copy()
map_df["co2_share_pct"] = (map_df["co2"] / world_total) * 100

fig_map = px.choropleth(
    map_df,
    locations="iso_code",
    color="co2_share_pct",
    hover_name="country",
    hover_data={
        "co2": True,
        "co2_share_pct": ":.2f"
    },
    color_continuous_scale="Blues",
    title=" Global CO2 Emission Distribution (%)"
)

fig_map.update_layout(
    height=600,
    geo=dict(
        showframe=False,
        showcoastlines=True,
        projection_type='equirectangular',
        bgcolor="rgba(0,0,0,0)",
        lakecolor="rgba(255,255,255,0.5)"
    ),
    paper_bgcolor="rgba(0,0,0,0)",
    margin=dict(t=50, b=0)
)

st.plotly_chart(fig_map, use_container_width=True)


if st.session_state.zoomed_chart:
    st.plotly_chart(render_chart(st.session_state.zoomed_chart), use_container_width=True, on_select="rerun", key="zoomed")
else:
    col1, col2, col3 = st.columns(3)
    charts = ["continent", "top_countries", "history"]
    for i, col in enumerate([col1, col2, col3]):
        with col:
            chart_key = charts[i]
            selected = st.plotly_chart(render_chart(chart_key), use_container_width=True, on_select="rerun", key=f"key_{chart_key}")
            if selected and selected.get("selection") and selected["selection"].get("points"):
                st.session_state.zoomed_chart = chart_key
                st.rerun()


# For Carbon Calculator

st.sidebar.title("Navigation")

# PREDICTION FUNCTION

def predict_emission(model, preprocessor, input_data):
    expected_cols = preprocessor.feature_names_in_
    input_data_cleaned = input_data[expected_cols]
    processed_data = preprocessor.transform(input_data_cleaned)
    prediction = model.predict(processed_data)
    return max(0.0, float(prediction[0]))

# Logic To Build UI Regression Model

st.sidebar.title("Carbon Calculator")
page = st.sidebar.radio("Go to:", ["Individual Sector", "Transport Sector", "Industry Sector", "Scope 3 Sector"])
st.title(f"{page} Carbon Calculator")
st.write("---")

# Individual Sector

if page == "Individual Sector":
    col1, col2 = st.columns(2)
    with col1:
        body = st.selectbox("Body Type", ['overweight', 'obese', 'underweight', 'normal'])
        sex = st.selectbox("Sex", ['female', 'male'])
        shower = st.selectbox("Shower Frequency", ['daily', 'less frequently', 'more frequently', 'twice a day'])
        heating = st.selectbox("Heating Source", ['coal', 'natural gas', 'wood', 'electricity'])
    with col2:
        social = st.selectbox("Social Activity", ['often', 'never', 'sometimes'])
        tv_hour = st.slider("Daily TV/PC Hours", 0, 24, 5)
        efficiency = st.selectbox("Energy Efficiency", ['No', 'Sometimes', 'Yes'])

    if st.button("Calculate Individual Footprint"):

        # Maping The categories

        shower_map = {'daily':'daily', 'less frequently':'less_frequently', 'more frequently':'more_frequently', 'twice a day':'twice_a_day'}
        heating_map = {'coal':'coal', 'natural gas':'natural_gas', 'wood':'wood', 'electricity':'electricity'}
        efficiency_map = {'No':'No', 'Sometimes':'Sometimes', 'Yes':'Yes'}

        data = pd.DataFrame([[body, sex, shower_map[shower], heating_map[heating], social, tv_hour, efficiency_map[efficiency]]],
                            columns=['Body Type', 'Sex', 'How Often Shower', 'Heating Energy Source', 'Social Activity', 'How Long TV PC Daily Hour', 'Energy efficiency'])

        model, preprocessor = load_individual_model()
        res = predict_emission(model, preprocessor, data)/365
        st.success(f"Estimated Individual Emission: {res:.2f} kg CO2 Emission per year")

# Transport Sector

elif page == "Transport Sector":
    col1, col2 = st.columns(2)
    with col1:
        transport_mode = st.selectbox("Main Transport", ['public', 'walk/bicycle', 'private'])
        v_type = st.selectbox("Vehicle Type", ['none', 'petrol', 'diesel', 'lpg', 'hybrid', 'electric'])
    with col2:
        distance = st.number_input("Monthly Distance (Km)", min_value=0, value=500)
        air_freq = st.selectbox("Ride Frequency", ['frequently', 'rarely', 'never', 'very frequently'])

    if st.button("Calculate Transport Footprint"):
        vehicle_map = {'none':'no_vehicle', 'petrol':'petrol', 'diesel':'diesel',
                       'lpg':'lpg', 'hybrid':'hybrid', 'electric':'electric'}
        air_map = {'frequently':'frequently', 'rarely':'rarely', 'never':'never',
                   'very frequently':'very_frequently'}

        data = pd.DataFrame([[transport_mode, vehicle_map[v_type], distance, air_map[air_freq]]],
                            columns=['Transport', 'Vehicle Type', 'Vehicle Monthly Distance Km', 'Frequency of Traveling by Air'])

        model, preprocessor = load_transport_model()
        res = predict_emission(model, preprocessor, data)
        st.info(f"Estimated Transport Emission: {res:.2f} kg CO2 Emission per year")

# Industry Sector

elif page == "Industry Sector":
    sector = st.selectbox("Industry Sector", ['Automotive Industry', 'Steel Manufacturing', 'Logistics', 'Cement Production'])

    if st.button("Calculate Industry Impact"):
        data = pd.DataFrame([[sector]], columns=['Industry_Sector'])
        model, preprocessor = load_industry_model()
        res = predict_emission(model, preprocessor, data)/365
        st.warning(f"Estimated Industry Sector Emission: {res:.2f} kg CO2 Emission per year")

# Scope 3 Sector

elif page == "Scope 3 Sector":
    col1, col2 = st.columns(2)
    with col1:
        diet = st.selectbox("Diet Type", ['pescatarian', 'vegetarian', 'omnivore', 'vegan'])
        w_size = st.selectbox("Waste Bag Size", ['large', 'extra large', 'small', 'medium'])
        w_count = st.number_input("Weekly Waste Bags", 1, 10, 3)
    with col2:
        clothes = st.number_input("New Monthly Clothes", 0, 50, 5)
        internet = st.slider("Daily Internet Hours", 0, 24, 8)

    if st.button("Calculate Scope 3 Footprint"):
        waste_map = {'small':'small', 'medium':'medium', 'large':'large', 'extra large':'extra_large'}
        data = pd.DataFrame([[diet, waste_map[w_size], w_count, clothes, internet]],
                            columns=['Diet', 'Waste Bag Size', 'Waste Bag Weekly Count', 'How Many New Clothes Monthly', 'How Long Internet Daily Hour'])

        model, preprocessor = load_scope3_model()
        res = predict_emission(model, preprocessor, data)/365
        st.success(f"Estimated Scope 3 Emission: {res:.2f} kg CO2 Emission per year")


# Logic To Build UI For Cnn Model

# Labeling The Output

readable_labels = {
    "a_Good": "Low Carbon Emission Level – Minimal CO₂ concentration with clean atmospheric conditions and negligible environmental impact.",

    "b_Moderate": "Moderate Carbon Emission Level – Slightly elevated CO₂ concentration indicating mild environmental impact.",

    "c_Unhealthy_for_Sensitive_Groups": "Elevated Carbon Emission Level – Increased CO₂ concentration that may affect sensitive populations and contribute to environmental stress.",

    "d_Unhealthy": "High Carbon Emission Level – Significant CO₂ concentration with noticeable environmental degradation and potential public health concerns.",

    "e_Very_Unhealthy": "Very High Carbon Emission Level – Substantial atmospheric carbon accumulation posing serious environmental and health risks.",

    "f_Severe": "Critical Carbon Emission Level – Extremely high CO₂ concentration with hazardous environmental impact and urgent mitigation required."
}

# UI For Cnn Model

st.title("Carbon Emission Classifier")
st.write("Upload an image of the sky or city to predict the emission level.")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:

    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption='Uploaded Image', width="stretch")

    # Preprocessing (RGB Only)

    img_array = np.array(image)
    img_resized = cv2.resize(img_array, (120, 120))  # This (120,20) Must match training size
    img_final = np.expand_dims(img_resized, axis=0)

    if st.button("Predict Emission Level"):

        prediction = model.predict(img_final)
        predicted_index = np.argmax(prediction)
        confidence = np.max(prediction) * 100

        # To Get original label
        original_label = original_class_names[predicted_index]

        # Maping to readable label
        result_text = readable_labels.get(original_label, original_label)

        # Debuging Output
        st.write("Prediction Probabilities:")
        st.write(prediction)

        # Giving Color-Based Output
        if original_label in ["a_Good", "c_Unhealthy_for_Sensitive_Groups"]:
            st.success(f"{result_text}\n\nConfidence: {confidence:.2f}%")

        elif original_label in ["b_Moderate", "d_Unhealthy"]:
            st.warning(f"{result_text}\n\nConfidence: {confidence:.2f}%")

        elif original_label in ["e_Very_Unhealthy", "f_Severe"]:
            st.error(f"{result_text}\n\nConfidence: {confidence:.2f}%")
