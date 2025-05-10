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

# initialize session state for estimate button
if 'made_estimate' not in st.session_state:
    st.session_state['made_estimate'] = False

# --- Caching data & model ---
@st.cache_data(show_spinner=False)
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

@st.cache_resource(show_spinner=False)
def load_model():
    with open("model_pipeline.pkl", "rb") as f:
        return pickle.load(f)

@st.cache_data(show_spinner=False)
def build_dropdowns(df):
    makes = sorted(df['Make'].unique())
    models = {mk: sorted(df[df['Make']==mk]['Model'].unique()) for mk in makes}
    series = {(mk,mo): sorted(df[(df['Make']==mk)&(df['Model']==mo)]['Series'].unique())
              for mk in makes for mo in models[mk]}
    engines = {(mk,mo,ser): sorted(df[(df['Make']==mk)&(df['Model']==mo)&(df['Series']==ser)]['Engine Code'].unique())
               for (mk,mo), sels in series.items() for ser in sels}
    roofs = {(mk,mo,ser): sorted(df[(df['Make']==mk)&(df['Model']==mo)&(df['Series']==ser)]['Roof'].unique())
             for (mk,mo), sels in series.items() for ser in sels}
    interiors = {(mk,mo,ser): sorted(df[(df['Make']==mk)&(df['Model']==mo)&(df['Series']==ser)]['Interior'].unique())
                 for (mk,mo), sels in series.items() for ser in sels}
    regions = sorted(df['Auction Region'].dropna().unique())
    colors = sorted(df['Color'].dropna().unique())
    return makes, models, series, engines, roofs, interiors, regions, colors

# --- VIN decode ---
def decode_vin(vin: str) -> dict:
    try:
        resp = requests.get(
            f"https://vpic.nhtsa.dot.gov/api/vehicles/DecodeVinValues/{vin}?format=json",
            timeout=5
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
    except:
        return {}

# --- Prediction helper ---
@st.cache_data(show_spinner=False)
def predict_value(_pipeline, features):
    cols = ['Year','Make','Model','Series','Engine Code','Grade','Mileage',
            'Drivable','Auction Region','Color','Roof','Interior','sale_month','age']
    rec = pd.DataFrame([features], columns=cols)
    logp = _pipeline.predict(rec)[0]
    return float(np.exp(logp))

# --- Load once ---
df = load_data()
pipeline = load_model()
_makes, _models, _series, _engines, _roofs, _ints, _regions, _colors = build_dropdowns(df)

# --- Sidebar ---
st.sidebar.header("Input Vehicle Specs")
vin = st.sidebar.text_input("Enter VIN (optional)")
decoded = {}
use_vin = False
model_year = None
if vin:
    decoded = decode_vin(vin)
    if decoded.get('Year',0) > 0:
        use_vin = True
        model_year = decoded['Year']
        st.sidebar.subheader("Decoded VIN Vehicle")
        for k in ['Year','Make','Model','Series']:
            val = decoded.get(k)
            if val:
                st.sidebar.write(f"**{k}:** {val}")

if not use_vin:
    model_year = st.sidebar.number_input(
        "Model Year", 1980, pd.Timestamp.now().year,
        pd.Timestamp.now().year
    )

make = st.sidebar.selectbox(
    "Make", _makes,
    index=_makes.index(decoded.get('Make')) if decoded.get('Make') in _makes else 0
)
model = st.sidebar.selectbox(
    "Model", _models[make],
    index=_models[make].index(decoded.get('Model')) if decoded.get('Model') in _models[make] else 0
)
series = st.sidebar.selectbox(
    "Series/Trim", _series[(make,model)],
    index=_series[(make,model)].index(decoded.get('Series')) if decoded.get('Series') in _series[(make,model)] else 0
)

# Engine fallback
disp = decoded.get('Disp','')
eng_list = _engines.get((make,model,series), [])
suggest = []
if use_vin and disp:
    suggest = difflib.get_close_matches(str(disp), eng_list, n=1)
elif use_vin and decoded.get('EngMod'):
    suggest = difflib.get_close_matches(decoded['EngMod'], eng_list, n=1)
if suggest:
    st.sidebar.info(f"Using closest engine match: {suggest[0]}")
    engine = suggest[0]
else:
    engine = st.sidebar.selectbox("Engine Type", eng_list)

roof = st.sidebar.selectbox("Roof Type", _roofs.get((make,model,series), []))
interior = st.sidebar.selectbox("Interior Type", _ints.get((make,model,series), []))
grade = st.sidebar.slider("Grade", 1.0, 5.0, 3.0)
mileage = st.sidebar.number_input("Mileage", 0, 300000, 50000)
drivable = st.sidebar.selectbox("Drivable", ['Yes','No'])
region = st.sidebar.selectbox("Auction Region", _regions)
color = st.sidebar.selectbox("Exterior Color", _colors)
# derive
sale_month = pd.Timestamp.now().month
age = pd.Timestamp.now().year - model_year

# --- Main title ---
st.title("🚗 Carolina Auto Auction Wholesale Evaluator")

# Estimate button
def do_estimate():
    features = {
        'Year': model_year,
        'Make': make,
        'Model': model,
        'Series': series,
        'Engine Code': engine,
        'Grade': grade,
        'Mileage': mileage,
        'Drivable': drivable,
        'Auction Region': region,
        'Color': color,
        'Roof': roof,
        'Interior': interior,
        'sale_month': sale_month,
        'age': age
    }
    val = predict_value(pipeline, features)
    st.success(f"💰 Estimated Wholesale Value: ${val:,.2f}")
    st.session_state['made_estimate'] = True

if st.sidebar.button("Estimate Wholesale Value"):
    do_estimate()

# Post-estimate display
if st.session_state.get('made_estimate'):
    cutoff = pd.Timestamp.now() - pd.Timedelta(days=120)
    low, high = model_year-2, model_year+2
    hist = df[(df['Sold Date']>=cutoff) &
              (df['Year'].between(low, high)) &
              (df['Make']==make) & (df['Model']==model) & (df['Series']==series)]
    st.subheader("Price History & Recent Sales")
    if not hist.empty:
        chart = alt.Chart(hist).mark_line(point=True).encode(
            x='Sold Date:T', y='Sale Price:Q'
        ).properties(width=600, height=300)
        st.altair_chart(chart, use_container_width=True)
        last10 = hist.sort_values('Sold Date', ascending=False).head(10)
        st.subheader("Last 10 Transactions")
        st.dataframe(
            last10[['Sold Date','Year','Make','Model','Series','Engine Code','Roof','Interior','Drivable','Grade','Mileage','Sale Price']],
            hide_index=True, use_container_width=True
        )
    else:
        st.info("No recent transactions found.")
