import streamlit as st
import pandas as pd
import numpy as np
import pickle
import altair as alt

# Configure page (first Streamlit command)
st.set_page_config(
    page_title="Carolina Auto Auction Wholesale Evaluator",
    layout="wide"
)

# Cache data and model loading for performance
df_data = st.cache_data(lambda: pd.read_excel("226842_196b62e8465_127512.xlsx"))()
for col in ['Make','Model','Series','Engine Code','Auction Region','Color','Roof','Interior']:
    if col in df_data.columns:
        df_data[col] = df_data[col].astype(str)
date_col = next((c for c in df_data.columns if 'date' in c.lower()), None)
if date_col:
    df_data[date_col] = pd.to_datetime(df_data[date_col])

pipeline = st.cache_resource(lambda: pickle.load(open("model_pipeline.pkl", 'rb')))()

# Cache dropdown mapping builds
def build_mappings(df):
    makes = sorted(df['Make'].dropna().unique())
    models_map = {mk: sorted(df[df['Make']==mk]['Model'].dropna().unique()) for mk in makes}
    series_map = {(mk,mdl): sorted(df[(df['Make']==mk)&(df['Model']==mdl)]['Series'].dropna().unique()) for mk in makes for mdl in models_map[mk]}
    engine_map = {(mk,mdl): sorted(df[(df['Make']==mk)&(df['Model']==mdl)]['Engine Code'].dropna().unique()) for mk in makes for mdl in models_map[mk]}
    roof_map = {(mk,mdl,ser): sorted(df[(df['Make']==mk)&(df['Model']==mdl)&(df['Series']==ser)]['Roof'].dropna().unique()) for mk in makes for mdl in models_map[mk] for ser in series_map[(mk,mdl)]}
    interior_map = {(mk,mdl,ser): sorted(df[(df['Make']==mk)&(df['Model']==mdl)&(df['Series']==ser)]['Interior'].dropna().unique()) for mk in makes for mdl in models_map[mk] for ser in series_map[(mk,mdl)]}
    regions = sorted(df['Auction Region'].dropna().unique())
    colors = sorted(df['Color'].dropna().unique())
    sale_months = [str(m) for m in range(1,13)]
    return makes, models_map, series_map, engine_map, roof_map, interior_map, regions, colors, sale_months

makes, models_map, series_map, engine_map, roof_map, interior_map, regions, colors, sale_months = st.cache_data(build_mappings)(df_data)

# App title
st.title("🚗 Carolina Auto Auction Wholesale Evaluator")

# Sidebar inputs
st.sidebar.header("Vehicle Input Specs")
model_year = st.sidebar.number_input("Model Year", 1990, 2025, 2021)
make = st.sidebar.selectbox("Make", makes)
model = st.sidebar.selectbox("Model", models_map[make])
series = st.sidebar.selectbox("Series/Trim", series_map[(make,model)])
engine = st.sidebar.selectbox("Engine Type", engine_map[(make,model)])
roof = st.sidebar.selectbox("Roof Type", roof_map[(make,model,series)])
interior = st.sidebar.selectbox("Interior Type", interior_map[(make,model,series)])
grade = st.sidebar.slider("Grade", 1.0, 5.0, 3.0, 0.1)
mileage = st.sidebar.number_input("Mileage", 0, 300000, 50000)
drivable = st.sidebar.selectbox("Drivable", ["Yes","No"])
region = st.sidebar.selectbox("Auction Region", regions)
color = st.sidebar.selectbox("Exterior Color", colors)
sale_month = st.sidebar.selectbox("Sale Month", sale_months)
sale_year = st.sidebar.number_input(
    "Sale Year",
    min_value=2000,
    max_value=int(pd.to_datetime("today").year),
    value=int(pd.to_datetime("today").year)
)
age = sale_year - model_year

# Main layout: tabs
tab1, tab2 = st.tabs(["Estimate", "History"])
with tab1:
    if st.button("Estimate Wholesale Value"):
        df_in = pd.DataFrame([{
            'Year': model_year,
            'Make': make,
            'Model': model,
            'Series': series,
            'Engine Code': engine,
            'Roof': roof,
            'Interior': interior,
            'Mileage': mileage,
            'Grade': grade,
            'Drivable': drivable,
            'Auction Region': region,
            'Color': color,
            'sale_month': sale_month,
            'age': age
        }])
        pred_log = pipeline.predict(df_in)[0]
        pred_val = np.expm1(pred_log)
        st.metric(label="Estimated Value", value=f"${pred_val:,.2f}")
with tab2:
    cutoff = pd.to_datetime("today") - pd.Timedelta(days=60)
    dfh = df_data[
        (df_data['Make']==make) &
        (df_data['Model']==model) &
        (df_data['Series']==series) &
        (df_data[date_col]>=cutoff)
    ]
    df_plot = dfh.set_index(date_col)['Sale Price'].resample('D').median().reset_index()
    if not df_plot.empty:
        chart = alt.Chart(df_plot).mark_line(color="#6a0dad").encode(
            x=alt.X(f'{date_col}:T', title='Date'),
            y=alt.Y('Sale Price:Q', title='Median Sale Price')
        ).properties(height=300)
        st.altair_chart(chart, use_container_width=True)
    else:
        st.write("No historical sales data in the last 60 days for this configuration.")