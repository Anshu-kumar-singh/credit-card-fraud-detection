import streamlit as st
import joblib
import numpy as np

model = joblib.load('fraud_model.pkl')

st.title("💳 Credit Card Fraud Detection")

fraud_sample = [-1.5487881, 1.80869795, -0.95350903, 2.21308539, -2.01572779,
-0.91345684, -2.35601298, 1.19716897, -1.67837406, -3.53865023,
3.10208993, -3.99337305, -1.93741062, -3.82289411, 0.83097011,
-2.47535885, -5.21187517, -0.41387168, 0.93326216, 0.39078596,
0.85513826, 0.77474482, 0.05903715, 0.34319981, -0.46893793,
-0.27833799, 0.62592222, 0.39557338, -0.04605724, -0.43495489]

normal_sample = [1.19185711, 0.26615071, 0.16648011, 0.44815408, 0.06001765,
-0.08236081, -0.07880298, 0.08510165, -0.25542513, -0.16697441,
1.61272666, 1.06523531, 0.48909502, -0.14377230, 0.63555809,
0.46391704, -0.11480466, -0.18336127, -0.14578304, -0.06908314,
-0.22577525, -0.63867195, 0.10128802, -0.33984648, 0.16717041,
0.12589453, -0.00898310, 0.01472417, -0.64421920, -1.99658437]

col1, col2 = st.columns(2)
with col1:
    load_fraud = st.button("🔴 Load Fraud Sample")
with col2:
    load_normal = st.button("🟢 Load Normal Sample")

# Store which sample to use
if load_fraud:
    st.session_state['defaults'] = fraud_sample
elif load_normal:
    st.session_state['defaults'] = normal_sample
elif 'defaults' not in st.session_state:
    st.session_state['defaults'] = [0.0] * 30

defaults = st.session_state['defaults']

st.write("### V1 to V28 Features")
v_values = []
cols = st.columns(4)
for i in range(1, 29):
    col = cols[(i-1) % 4]
    val = col.number_input(f"V{i}", value=float(defaults[i-1]), key=f"v{i}_{defaults[i-1]}")
    v_values.append(val)

amount = st.number_input("Amount (scaled)", value=float(defaults[28]), key=f"amt_{defaults[28]}")
time = st.number_input("Time (scaled)", value=float(defaults[29]), key=f"time_{defaults[29]}")

if st.button("🔍 Check Transaction"):
    input_data = np.array([[*v_values, amount, time]])
    prediction = model.predict(input_data)[0]
    probability = model.predict_proba(input_data)[0][1]

    if prediction == 1:
        st.error(f"🚨 FRAUD DETECTED! Probability: {probability:.2%}")
    else:
        st.success(f"✅ Transaction is NORMAL. Fraud Probability: {probability:.2%}")