import streamlit as st
import pandas as pd
import numpy as np
import pickle
import requests
import altair as alt

# 1) Page config must be first Streamlit call
st.set_page_config(
    page_title="Carolina Auto Auction Wholesale Evaluator",
    layout="wide",
)

# 2) Load data & detect date column
@st.cache_data
def load_data():
    df = pd.read_excel("226842_196b62e8465_127512.xlsx")
    # ensure consistent dtypes
    for c in ['Make','Model','Series','Engine Code','Auction Region','Color','Roof','Interior']:
        if c in df.columns:
            df[c] = df[c].astype(str)
    # find any column with 'date' in its name
    date_col = next((col for col in df.columns if 'date' in col.lower()), None)
    if date_col:
        df[date_col] = pd.to_datetime(df[date_col])
    return df, date_col

df_data, date_col = load_data()

# 3) Load your trained pipeline if still needed (unused below)
@st.cache_resource
def load_pipeline():
    with open("model_pipeline.pkl", "rb") as f:
        return pickle.load(f)

pipeline = load_pipeline()

# 4) VIN decoder via NHTSA API
@st.cache_data
def decode_vin(vin):
    url = f"https://vpic.nhtsa.dot.gov/api/vehicles/DecodeVinValues/{vin}?format=json"
    r = requests.get(url).json().get("Results", [{}])[0]
    return {
        "Year": int(r.get("ModelYear") or 0),
        "Make": r.get("Make", "").upper(),
        "Model": r.get("Model", "").upper(),
        "Series": r.get("Trim", "").upper()
    }

# 5) Build sidebar dropdown mappings
@st.cache_data
def build_mappings(df):
    makes = sorted(df["Make"].unique())
    models_map = {m: sorted(df[df["Make"]==m]["Model"].unique()) for m in makes}
    series_map = {(m,md): sorted(df[(df["Make"]==m)&(df["Model"]==md)]["Series"].unique())
                  for m in makes for md in models_map[m]}
    engine_map = {(m,md,ser): sorted(df[(df["Make"]==m)&(df["Model"]==md)&(df["Series"]==ser)]["Engine Code"].unique())
                  for m in makes for md in models_map[m] for ser in series_map[(m,md)]}
    roof_map = {(m,md,ser): sorted(df[(df["Make"]==m)&(df["Model"]==md)&(df["Series"]==ser)]["Roof"].unique())
                for m in makes for md in models_map[m] for ser in series_map[(m,md)]}
    interior_map = {(m,md,ser): sorted(df[(df["Make"]==m)&(df["Model"]==md)&(df["Series"]==ser)]["Interior"].unique())
                    for m in makes for md in models_map[m] for ser in series_map[(m,md)]}
    regions = sorted(df["Auction Region"].unique())
    colors = sorted(df["Color"].unique())
    return makes, models_map, series_map, engine_map, roof_map, interior_map, regions, colors

makes, models_map, series_map, engine_map, roof_map, interior_map, regions, colors = build_mappings(df_data)

# 6) Sidebar inputs
st.sidebar.header("Input Vehicle Specs")

vin = st.sidebar.text_input("Enter VIN (optional)").strip()
decoded = {}
if vin:
    decoded = decode_vin(vin)
    if decoded["Year"] > 0:
        st.sidebar.subheader("Decoded VIN Vehicle")
        st.sidebar.write(f"**Year:** {decoded['Year']}")
        st.sidebar.write(f"**Make:** {decoded['Make']}")
        st.sidebar.write(f"**Model:** {decoded['Model']}")
        st.sidebar.write(f"**Trim:** {decoded['Series']}")
    else:
        st.sidebar.error("VIN could not be decoded. Please select manually.")
        vin = ""

if not vin:
    model_year = st.sidebar.number_input("Model Year", 1990, 2025, 2021)
    make       = st.sidebar.selectbox("Make", makes)
    model      = st.sidebar.selectbox("Model", models_map[make])
    series     = st.sidebar.selectbox("Series/Trim", series_map[(make, model)])
else:
    model_year = decoded["Year"]
    make       = decoded["Make"]
    model      = decoded["Model"]
    series     = decoded["Series"]

# always-visible inputs
engine   = st.sidebar.selectbox("Engine Type", engine_map[(make, model, series)])
grade    = st.sidebar.slider("Grade", 1.0, 5.0, 3.0, step=0.1)
mileage  = st.sidebar.number_input("Mileage", 0, 300000, 50000)
drivable = st.sidebar.selectbox("Drivable", ["Yes","No"])
region   = st.sidebar.selectbox("Auction Region", regions)
color    = st.sidebar.selectbox("Exterior Color", colors)
roof     = st.sidebar.selectbox("Roof Type", roof_map[(make, model, series)])
interior = st.sidebar.selectbox("Interior Type", interior_map[(make, model, series)])

# 7) Main area: title & tabs
st.title("🚗 Carolina Auto Auction Wholesale Evaluator")
tab1, tab2 = st.tabs(["Estimate", "History & Recent Sales"])

# 8) Helper to filter last-60-day records
def get_recent_df():
    cutoff = pd.to_datetime("today") - pd.Timedelta(days=60)
    return df_data[
        (df_data["Make"]==make) &
        (df_data["Model"]==model) &
        (df_data["Series"]==series) &
        (df_data[date_col] >= cutoff)
    ]

with tab1:
    if st.button("Estimate Wholesale Value"):
        dfh = get_recent_df()
        if not dfh.empty:
            median_price = dfh["Sale Price"].median()
            st.metric("Estimated Value", f"${median_price:,.2f}")
            recent = dfh.sort_values(date_col, ascending=False).head(5)
            st.write("### Last 5 Transactions")
            st.dataframe(
                recent[[date_col, "Make", "Model", "Series", "Sale Price"]]
                .rename(columns={date_col: "Sale Date"}),
                use_container_width=True
            )
        else:
            st.warning("No historical data for this configuration in the last 60 days.")

with tab2:
    dfh = get_recent_df()
    if not dfh.empty:
        df_plot = (
            dfh.set_index(date_col)["Sale Price"]
            .resample("D").median()
            .reset_index()
        )
        chart = alt.Chart(df_plot).mark_line(color="#6a0dad").encode(
            x=alt.X(f"{date_col}:T", title="Date"),
            y=alt.Y("Sale Price:Q", title="Median Daily Price")
        ).properties(height=300)
        st.altair_chart(chart, use_container_width=True)
    else:
        st.write("No sales to chart in the last 60 days.")
