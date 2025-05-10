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

# initialize session state
if "made_estimate" not in st.session_state:
    st.session_state["made_estimate"] = False

# --- Caching data & model ---
@st.cache_data(show_spinner=False)
def load_data():
    df = pd.read_excel(
        "226842_196b62e8465_127512.xlsx",
        parse_dates=["Sold Date"], engine="openpyxl"
    )
    df["sale_month"] = df["Sold Date"].dt.month
    df["age"] = pd.Timestamp.now().year - df["Year"]
    # ensure no nulls sneak into our categorical columns
    for c in ["Make","Model","Series","Engine Code","Roof","Interior"]:
        df[c] = df[c].fillna("").astype(str)
    return df

@st.cache_resource(show_spinner=False)
def load_model():
    with open("model_pipeline.pkl","rb") as f:
        return pickle.load(f)

@st.cache_data(show_spinner=False)
def build_dropdowns(df):
    makes = sorted(df["Make"].unique())
    models = {mk: sorted(df[df["Make"]==mk]["Model"].unique()) for mk in makes}
    series = {
        (mk,mo): sorted(df[(df["Make"]==mk)&(df["Model"]==mo)]["Series"].unique())
        for mk in makes for mo in models[mk]
    }
    engines = {
        (mk,mo,ser): sorted(
            df[(df["Make"]==mk)&(df["Model"]==mo)&(df["Series"]==ser)]["Engine Code"].unique()
        )
        for (mk,mo), sels in series.items() for ser in sels
    }
    roofs = {
        (mk,mo,ser): sorted(
            df[(df["Make"]==mk)&(df["Model"]==mo)&(df["Series"]==ser)]["Roof"].unique()
        )
        for (mk,mo), sels in series.items() for ser in sels
    }
    interiors = {
        (mk,mo,ser): sorted(
            df[(df["Make"]==mk)&(df["Model"]==mo)&(df["Series"]==ser)]["Interior"].unique()
        )
        for (mk,mo), sels in series.items() for ser in sels
    }
    regions = sorted(df["Auction Region"].dropna().unique())
    colors = sorted(df["Color"].dropna().unique())
    return makes, models, series, engines, roofs, interiors, regions, colors

# --- VIN decode ---
def decode_vin(vin:str) -> dict:
    try:
        r = requests.get(
            f"https://vpic.nhtsa.dot.gov/api/vehicles/DecodeVinValues/{vin}?format=json",
            timeout=5
        ).json()["Results"][0]
        return {
            "Year": int(r.get("ModelYear") or 0),
            "Make": r.get("Make","").upper(),
            "Model": r.get("Model","").upper(),
            "Series": r.get("Trim","").upper(),
            "Disp": r.get("Engine Displacement (L)") or "",
            "EngMod": r.get("Engine Model") or ""
        }
    except:
        return {}

# --- Prediction helper ---
@st.cache_data(show_spinner=False)
def predict_value(_pipeline, feat:dict):
    cols = [
        "Year","Make","Model","Series","Engine Code","Grade","Mileage",
        "Drivable","Auction Region","Color","Roof","Interior","sale_month","age"
    ]
    rec = pd.DataFrame([{k:feat[k] for k in cols}], columns=cols)
    # cast categoricals to str (defensive)
    for c in ["Make","Model","Series","Engine Code","Drivable","Auction Region","Color","Roof","Interior"]:
        rec[c] = rec[c].astype(str)
    lp = _pipeline.predict(rec)[0]
    return float(np.exp(lp))

# --- Load once ---
df = load_data()
pipeline = load_model()
_makes, _models, _series, _engines, _roofs, _ints, _regions, _colors = build_dropdowns(df)

# --- Sidebar inputs ---
st.sidebar.header("Input Vehicle Specs")
vin = st.sidebar.text_input("Enter VIN (optional)")
decoded, use_vin = {}, False

if vin:
    decoded = decode_vin(vin)
    if decoded.get("Year",0)>0:
        use_vin=True
        st.sidebar.subheader("Decoded VIN Vehicle")
        for field in ["Year","Make","Model","Series"]:
            st.sidebar.write(f"**{field}:** {decoded[field]}")

# Year & manual selects
year = decoded.get("Year") if use_vin else st.sidebar.number_input(
    "Model Year",1980,pd.Timestamp.now().year,pd.Timestamp.now().year
)
make = st.sidebar.selectbox(
    "Make", _makes,
    index=_makes.index(decoded.get("Make","")) if decoded.get("Make") in _makes else 0
)
model = st.sidebar.selectbox(
    "Model", _models[make],
    index=_models[make].index(decoded.get("Model","")) if decoded.get("Model") in _models[make] else 0
)
series = st.sidebar.selectbox(
    "Series/Trim", _series[(make,model)],
    index=_series[(make,model)].index(decoded.get("Series","")) if decoded.get("Series") in _series[(make,model)] else 0
)

# Engine fallback
disp = decoded.get("Disp","")
elist = _engines.get((make,model,series),[])
suggest = []
if use_vin and disp:
    suggest = difflib.get_close_matches(str(disp),elist,n=1)
elif use_vin and decoded.get("EngMod"):
    suggest = difflib.get_close_matches(decoded["EngMod"],elist,n=1)
if suggest:
    st.sidebar.info(f"Using closest engine match: {suggest[0]}")
    engine = suggest[0]
else:
    engine = st.sidebar.selectbox("Engine Type",elist)

roof = st.sidebar.selectbox("Roof Type",_roofs.get((make,model,series),[]))
interior = st.sidebar.selectbox("Interior Type",_ints.get((make,model,series),[]))
grade = st.sidebar.slider("Grade",1.0,5.0,3.0)
mileage = st.sidebar.number_input("Mileage",0,300000,50000)
drivable = st.sidebar.selectbox("Drivable",["Yes","No"])
region = st.sidebar.selectbox("Auction Region",_regions)
color = st.sidebar.selectbox("Exterior Color",_colors)

# derive
sale_month = pd.Timestamp.now().month
age = pd.Timestamp.now().year - year

# --- Main title ---
st.title("🚗 Carolina Auto Auction Wholesale Evaluator")

# Estimate button callback
def do_estimate():
    feats = {
        "Year":year,"Make":make,"Model":model,"Series":series,
        "Engine Code":engine,"Grade":grade,"Mileage":mileage,
        "Drivable":drivable,"Auction Region":region,"Color":color,
        "Roof":roof,"Interior":interior,"sale_month":sale_month,"age":age
    }
    val = predict_value(pipeline,feats)
    st.success(f"💰 Estimated Wholesale Value: ${val:,.2f}")
    st.session_state["made_estimate"] = True

if st.sidebar.button("Estimate Wholesale Value"):
    do_estimate()

# After estimate: history & table
if st.session_state["made_estimate"]:
    cutoff = pd.Timestamp.now() - pd.Timedelta(days=120)
    low,high = year-2,year+2
    subset = df[
        (df["Sold Date"]>=cutoff) &
        (df["Year"].between(low,high)) &
        (df["Make"]==make) &
        (df["Model"]==model) &
        (df["Series"]==series)
    ]
    st.subheader("Price History & Recent Sales")
    if not subset.empty:
        chart = (
            alt.Chart(subset).mark_line(point=True)
               .encode(x="Sold Date:T",y="Sale Price:Q")
               .properties(width=700,height=300)
        )
        st.altair_chart(chart,use_container_width=True)
        st.subheader("Last 10 Transactions")
        last10 = subset.sort_values("Sold Date",ascending=False).head(10)
        st.dataframe(
            last10[["Sold Date","Year","Make","Model","Series","Engine Code","Drivable","Roof","Interior","Grade","Mileage","Sale Price"]],
            hide_index=True,use_container_width=True
        )
    else:
        st.info("No recent transactions found.")
