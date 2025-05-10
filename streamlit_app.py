import streamlit as st
import pandas as pd
import numpy as np
import pickle
import requests
import altair as alt
import difflib

# --- Page config ---
st.set_page_config(
    page_title="Carolina Auto Auction Wholesale Evaluator",
    layout="wide",
    initial_sidebar_state="expanded",
)

# --- Caching data & model ---
@st.cache_data
def load_data():
    df = pd.read_excel(
        "226842_196b62e8465_127512.xlsx",
        parse_dates=["Sold Date"], engine="openpyxl"
    )
    df['sale_month'] = df['Sold Date'].dt.month
    df['age'] = pd.Timestamp.now().year - df['Year']
    for col in ['Make','Model','Series','Engine Code','Roof','Interior']:
        df[col] = df[col].astype(str)
    return df

@st.cache_resource
def load_model():
    with open("model_pipeline.pkl", "rb") as f:
        return pickle.load(f)

# Load once
df = load_data()
pipeline = load_model()

# --- Precompute lookups ---
makes = sorted(df['Make'].unique())
models_by_make = {mk: sorted(df[df['Make']==mk]['Model'].unique()) for mk in makes}
series_by_model = {
    (mk,mo): sorted(df[(df['Make']==mk)&(df['Model']==mo)]['Series'].unique())
    for mk in makes for mo in models_by_make[mk]
}
engines_by_series = {
    (mk,mo,ser): sorted(
        df[(df['Make']==mk)&(df['Model']==mo)&(df['Series']==ser)]['Engine Code'].unique()
    )
    for (mk,mo), sels in series_by_model.items() for ser in sels
}
engines_by_model = {
    (mk,mo): sorted(df[(df['Make']==mk)&(df['Model']==mo)]['Engine Code'].unique())
    for mk in makes for mo in models_by_make[mk]
}
roof_by_series = {
    (mk,mo,ser): sorted(
        df[(df['Make']==mk)&(df['Model']==mo)&(df['Series']==ser)]['Roof'].unique()
    )
    for (mk,mo), sels in series_by_model.items() for ser in sels
}
roof_by_model = {
    (mk,mo): sorted(df[(df['Make']==mk)&(df['Model']==mo)]['Roof'].unique())
    for mk in makes for mo in models_by_make[mk]
}
interior_by_series = {
    (mk,mo,ser): sorted(
        df[(df['Make']==mk)&(df['Model']==mo)&(df['Series']==ser)]['Interior'].unique()
    )
    for (mk,mo), sels in series_by_model.items() for ser in sels
}
interior_by_model = {
    (mk,mo): sorted(df[(df['Make']==mk)&(df['Model']==mo)]['Interior'].unique())
    for mk in makes for mo in models_by_make[mk]
}
regions = sorted(df['Auction Region'].dropna().unique())
colors = sorted(df['Color'].dropna().unique())

# --- VIN decode ---
def decode_vin(vin: str) -> dict:
    resp = requests.get(
        f"https://vpic.nhtsa.dot.gov/api/vehicles/DecodeVinValues/{vin}?format=json"
    ).json()
    r = resp['Results'][0]
    return {
        'Year': int(r.get('ModelYear') or 0),
        'Make': r.get('Make','').upper(),
        'Model': r.get('Model','').upper(),
        'Series': r.get('Trim','').upper(),
        'Disp': r.get('Engine Displacement (L)') or '',
        'EngMod': r.get('Engine Model') or ''
    }

# --- Sidebar ---
st.sidebar.header("Input Vehicle Specs")
vin = st.sidebar.text_input("Enter VIN (optional)")
use_vin = False
decoded = {}

if vin:
    try:
        decoded = decode_vin(vin)
        if decoded['Year'] > 0:
            use_vin = True
            st.sidebar.subheader("Decoded VIN Vehicle")
            for k in ['Year','Make','Model','Series']:
                st.sidebar.write(f"{k}: {decoded[k]}")
    except:
        st.sidebar.error("VIN decode failed — please select manually.")

if not use_vin:
    decoded['Year'] = st.sidebar.number_input(
        "Model Year", 1980, pd.Timestamp.now().year, pd.Timestamp.now().year
    )
    decoded['Make'] = st.sidebar.selectbox("Make", makes)
    decoded['Model'] = st.sidebar.selectbox(
        "Model", models_by_make[decoded['Make']]
    )
    decoded['Series'] = st.sidebar.selectbox(
        "Series/Trim", series_by_model[(decoded['Make'],decoded['Model'])]
    )

# Engine selection with fallback
disp = decoded.get('Disp','')
eng_list = engines_by_series.get(
    (decoded['Make'],decoded['Model'],decoded['Series']), []
) or engines_by_model.get((decoded['Make'],decoded['Model']), [])
suggest = []
if use_vin and disp:
    suggest = difflib.get_close_matches(disp, eng_list, n=1)
if use_vin and not suggest and decoded.get('EngMod'):
    suggest = difflib.get_close_matches(decoded['EngMod'], eng_list, n=1)
if suggest:
    decoded['Engine Code'] = suggest[0]
else:
    decoded['Engine Code'] = st.sidebar.selectbox("Engine Type", eng_list)

# Common inputs
decoded['Grade'] = st.sidebar.slider("Grade", 1.0, 5.0, 3.0, 0.1)
decoded['Mileage'] = st.sidebar.number_input("Mileage", 0, 300000, 50000)
decoded['Drivable'] = st.sidebar.selectbox("Drivable", ["Yes","No"])
decoded['Auction Region'] = st.sidebar.selectbox("Auction Region", regions)
decoded['Color'] = st.sidebar.selectbox("Exterior Color", colors)

# Roof selection fallback
r_list = roof_by_series.get(
    (decoded['Make'],decoded['Model'],decoded['Series']), []
) or roof_by_model.get((decoded['Make'],decoded['Model']), [])
decoded['Roof'] = st.sidebar.selectbox("Roof Type", r_list)

# Interior selection fallback
i_list = interior_by_series.get(
    (decoded['Make'],decoded['Model'],decoded['Series']), []
) or interior_by_model.get((decoded['Make'],decoded['Model']), [])
decoded['Interior'] = st.sidebar.selectbox("Interior Type", i_list)

# Derive sale_month & age
dnow = pd.Timestamp.now()
decoded['sale_month'] = dnow.month
decoded['age'] = dnow.year - decoded['Year']

# --- Main ---
st.title("🚗 Carolina Auto Auction Wholesale Evaluator")
if st.button("Estimate Wholesale Value"):
    rec = pd.DataFrame([decoded])
    logp = pipeline.predict(rec)[0]
    val = np.exp(logp)
    st.success(f"💰 Estimated Wholesale Value: ${val:,.2f}")

# Tabs
tab1, tab2 = st.tabs(["Estimate","History & Recent Sales"])

@st.cache_data
def get_hist_df(make, model, series, cutoff):
    return df.loc[
        (df['Make']==make) & (df['Model']==model) &
        (df['Series']==series) & (df['Sold Date']>=cutoff)
    ]

@st.cache_data
def get_recent_df(make, model, series, low, high, cutoff):
    return df.loc[
        (df['Make']==make) & (df['Model']==model) &
        (df['Series']==series) & (df['Year']>=low) &
        (df['Year']<=high) & (df['Sold Date']>=cutoff)
    ]

with tab2:
    cutoff = pd.Timestamp.now() - pd.Timedelta(days=120)
    hist = get_hist_df(decoded['Make'], decoded['Model'], decoded['Series'], cutoff)
    st.subheader("Price History (Last 120 Days)")
    if not hist.empty:
        chart = alt.Chart(hist).mark_line(point=True).encode(
            x='Sold Date:T', y='Sale Price:Q'
        ).properties(width=600, height=300)
        st.altair_chart(chart, use_container_width=True)
    else:
        st.info("No recent sales in last 120 days.")

    st.subheader("Last 10 Transactions (±2 years)")
    low, high = decoded['Year']-2, decoded['Year']+2
    recent = get_recent_df(decoded['Make'], decoded['Model'], decoded['Series'], low, high, cutoff)
    recent = recent.sort_values('Sold Date', ascending=False).head(10)
    if not recent.empty:
        st.dataframe(
            recent[['Sold Date','Year','Make','Model','Series','Grade','Mileage','Sale Price']],
            hide_index=True, use_container_width=True
        )
    else:
        st.info("No historical transactions found.")
