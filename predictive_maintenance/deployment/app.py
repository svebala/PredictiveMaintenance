import streamlit as st
import pandas as pd
import joblib
from huggingface_hub import hf_hub_download
from datetime import datetime

st.set_page_config(page_title="Predictive Maintenance",page_icon="🚗",layout="wide")
HF_REPO="BalaSVenkat/predictive-maintenance-model"
MODEL_FILE="mlops_predictive_maintenance_model.joblib"
THRESHOLD=0.35
CFG={
"engine_rpm":{"label":"Engine RPM","physical":(61.0,2239.0),"operating":(380.0,1565.0),"default":791.24,"step":1.0,"unit":"RPM"},
"lub_oil_pressure":{"label":"Lubricating Oil Pressure","physical":(0.003,7.266),"operating":(0.86,5.61),"default":3.304,"step":0.01,"unit":"bar"},
"fuel_pressure":{"label":"Fuel Pressure","physical":(0.003,21.138),"operating":(1.40,16.16),"default":6.656,"step":0.01,"unit":"bar"},
"coolant_pressure":{"label":"Coolant Pressure","physical":(0.002,7.479),"operating":(0.72,5.95),"default":2.335,"step":0.01,"unit":"bar"},
"lub_oil_temp":{"label":"Lubricating Oil Temperature","physical":(71.322,89.581),"operating":(73.41,87.35),"default":77.643,"step":0.1,"unit":"°C"},
"coolant_temp":{"label":"Coolant Temperature","physical":(61.673,195.528),"operating":(65.74,91.78),"default":78.427,"step":0.1,"unit":"°C"}}
@st.cache_resource
def load_model():
 p=hf_hub_download(repo_id=HF_REPO,filename=MODEL_FILE);return joblib.load(p)
def validate(v):
 e=[];w=[]
 for k,x in v.items():
  p=CFG[k]["physical"];o=CFG[k]["operating"]
  if x<p[0] or x>p[1]:e.append(CFG[k]["label"])
  elif x<o[0] or x>o[1]:w.append(CFG[k]["label"])
 return e,w
model=load_model()
with st.sidebar:
 st.header("Model");st.write("Class 1 = Faulty");st.write(f"Threshold: {THRESHOLD:.0%}")
st.title("🚗 Engine Predictive Maintenance")
vals={}
c1,c2=st.columns(2)
for i,(k,c) in enumerate(CFG.items()):
 col=c1 if i<3 else c2
 with col:
  st.markdown(f"**{c['label']} ({c['unit']})**")
  vals[k]=st.slider(c["label"],float(c["physical"][0]),float(c["physical"][1]),float(c["default"]),step=float(c["step"]),label_visibility="collapsed")
if st.button("Run Diagnostics",use_container_width=True):
 err,warn=validate(vals)
 if err:
  st.error("Invalid physical readings: "+", ".join(err));st.stop()
 X=pd.DataFrame([vals])
 p=float(model.predict_proba(X)[0,1]);fault=p>=THRESHOLD
 a,b=st.columns([2,1])
 with a:
  st.subheader("Diagnostic Result")
  st.error(f"⚠️ Faulty Engine ({p:.1%})") if fault else st.success(f"✅ Healthy Engine ({p:.1%})")
  if warn: st.warning("Outside operating range: "+", ".join(warn))
  st.subheader("Recommended Action")
  st.write("- Schedule maintenance\n- Inspect lubrication\n- Inspect cooling") if fault else st.write("- Continue monitoring")
 with b:
  st.metric("Fault Probability",f"{p:.1%}",delta=f"{p-THRESHOLD:+.1%}")
  st.progress(min(p,1.0))
 with st.expander("Sensor Readings"):
  st.dataframe(pd.DataFrame([vals]).T.rename(columns={0:"Value"}))
 st.caption(datetime.now().strftime("%d-%b-%Y %H:%M:%S"))
st.divider();st.caption("Decision support only.")
