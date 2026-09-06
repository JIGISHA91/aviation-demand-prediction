import streamlit as st
import joblib
import pandas as pd

lr_model = joblib.load("models/aviation_model.pkl")
nn_model = joblib.load("models/aviation_nn_model.pkl")
scaler = joblib.load("models/scaler.pkl")
feature_columns = joblib.load("models/feature_columns.pkl")

airline_options = [c.replace("Airline_", "") for c in feature_columns if c.startswith("Airline_")]
month_map = {
    1: "January", 2: "February", 3: "March", 4: "April",
    5: "May", 6: "June", 7: "July", 8: "August",
    9: "September", 10: "October", 11: "November", 12: "December"
}

st.set_page_config(page_title="Aviation Demand Predictor", layout="wide")

st.markdown("""
    <style>
    .stApp { background-color: #FFFFFF; }
    section[data-testid="stSidebar"] { background-color: #0B1F3A; }
    section[data-testid="stSidebar"] * { color: #FFFFFF !important; }
    .metric-card {
        background-color: #F8F9FB;
        border-radius: 10px;
        padding: 18px 20px;
        border: 1px solid #E5E8EC;
    }
    .metric-label { font-size: 13px; color: #5A6472; margin-bottom: 6px; }
    .metric-value { font-size: 28px; font-weight: 700; color: #0B1F3A; }
    .metric-delta-up { color: #1D9E75; font-size: 13px; font-weight: 600; }
    .metric-delta-down { color: #D85A30; font-size: 13px; font-weight: 600; }
    </style>
""", unsafe_allow_html=True)

st.title("Indian Aviation Passenger Demand Predictor")
st.write("Multiple Linear Regression vs. Neural Network (Deep Learning) - trained on real DGCA monthly carrier-wise data.")
st.divider()

st.sidebar.header("Model Information")
st.sidebar.write("Data source: DGCA, Govt. of India")
st.sidebar.write("Observations: 3,355 (cleaned)")
st.sidebar.subheader("Linear Regression")
st.sidebar.metric("Test R2", "0.8346")
st.sidebar.subheader("Neural Network (MLP)")
st.sidebar.metric("Test R2", "0.9683")

c1, c2, c3, c4 = st.columns(4)
cards = [
    (c1, "Linear Regression R2", "0.8346", "up"),
    (c2, "Neural Network R2", "0.9683", "up"),
    (c3, "LR Mean Error (MAE)", "247,213", "down"),
    (c4, "NN Mean Error (MAE)", "111,475", "down"),
]
for col, label, value, direction in cards:
    arrow = "up" if direction == "up" else "down"
    css_class = "metric-delta-up" if direction == "up" else "metric-delta-down"
    col.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">{label}</div>
            <div class="metric-value">{value}</div>
            <div class="{css_class}">{arrow}</div>
        </div>
    """, unsafe_allow_html=True)

st.write("")
st.subheader("Choose model and enter flight details")

model_choice = st.radio("Prediction model:", ["Linear Regression", "Neural Network (Deep Learning)"], horizontal=True)

col1, col2 = st.columns(2)
with col1:
    airline = st.selectbox("Airline", airline_options)
    month_name = st.selectbox("Month", list(month_map.values()))
    year = st.number_input("Year", min_value=2015, max_value=2026, value=2023, step=1)
with col2:
    aircraft_km = st.number_input("Aircraft Kilometres", min_value=0.0, value=20000.0)
    seat_km = st.number_input("Seat Kilometers", min_value=0.0, value=3000000.0)
    load_factor = st.slider("Passenger Load Factor (%)", 0.0, 100.0, 75.0)

if st.button("Predict Passenger Volume", type="primary"):
    month_number = None
    for k in month_map:
        if month_map[k] == month_name:
            month_number = k

    input_dict = {}
    for col in feature_columns:
        input_dict[col] = 0

    input_dict["Aircraft Kilometres"] = aircraft_km
    input_dict["Seat Kilometers"] = seat_km
    input_dict["Passenger Load Factor"] = load_factor
    input_dict["Year"] = year

    airline_col = "Airline_" + airline
    if airline_col in input_dict:
        input_dict[airline_col] = 1

    month_col = "Month_" + str(month_number)
    if month_col in input_dict:
        input_dict[month_col] = 1

    input_df = pd.DataFrame([input_dict])
    input_df = input_df[feature_columns]

    if model_choice == "Linear Regression":
        prediction = lr_model.predict(input_df)[0]
    else:
        scaled_input = scaler.transform(input_df)
        prediction = nn_model.predict(scaled_input)[0]

    if prediction < 0:
        prediction = 0

    st.success("Predicted Passenger Number: " + f"{int(prediction):,}")
    st.caption("Model used: " + model_choice + " | " + airline + ", " + month_name + " " + str(year))

st.divider()

with st.expander("View Model Comparison"):
    comparison_df = pd.DataFrame({
        "Metric": ["Training R2", "Test R2", "MAE", "RMSE"],
        "Linear Regression": ["0.8269", "0.8346", "247,212.96", "463,014.92"],
        "Neural Network (MLP)": ["0.9696", "0.9683", "111,474.76", "202,807.84"]
    })
    st.dataframe(comparison_df, use_container_width=True)

with st.expander("View Sample Training Data"):
    df_preview = pd.read_csv("data/carrier.csv")
    df_preview = df_preview[~df_preview["Airline"].str.contains("Total", case=False, na=False)]
    df_preview = df_preview.dropna(subset=["Passenger Number", "Aircraft Kilometres", "Seat Kilometers", "Passenger Load Factor", "Airline"])
    df_preview = df_preview.head(10)
    st.dataframe(df_preview)

st.divider()
st.subheader("Explore the Data")

df_charts = pd.read_csv("data/carrier.csv")
df_charts = df_charts[~df_charts["Airline"].str.contains("Total", case=False, na=False)]
df_charts = df_charts.dropna(subset=["Passenger Number", "Aircraft Kilometres", "Seat Kilometers", "Passenger Load Factor", "Airline"])

tab1, tab2, tab3 = st.tabs(["Passengers by Airline", "Load Factor vs Passengers", "Passenger Distribution"])

with tab1:
    selected_airlines = st.multiselect(
        "Select airlines to compare",
        options=sorted(df_charts["Airline"].unique()),
        default=["IndiGo", "Air India", "SpiceJet", "Vistara"]
    )
    if selected_airlines:
        filtered = df_charts[df_charts["Airline"].isin(selected_airlines)]
        chart_data = filtered.groupby("Airline")["Passenger Number"].mean().sort_values(ascending=False)
        st.bar_chart(chart_data)
    else:
        st.info("Select at least one airline to see the chart.")

with tab2:
    st.scatter_chart(df_charts, x="Passenger Load Factor", y="Passenger Number", color="Airline")

with tab3:
    hist_values, bin_edges = pd.cut(df_charts["Passenger Number"], bins=20, retbins=True)
    hist_counts = hist_values.value_counts().sort_index()
    labels = []
    for i in range(len(bin_edges) - 1):
        labels.append(f"{int(bin_edges[i]/1000):05d}K")
    chart_df = pd.DataFrame({"Passenger Number Range": labels, "Count": hist_counts.values})
    st.bar_chart(chart_df.set_index("Passenger Number Range"))