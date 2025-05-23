

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pickle

# Load model
model = pickle.load(open('model.pkl', 'rb'))

# Streamlit page config
st.set_page_config(
    page_title="AeroReach Ticket Purchase Prediction 🎯",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.title("🎯 AeroReach Ticket Purchase Prediction")
st.subheader("Predict the likelihood of a user purchasing a ticket based on their engagement metrics.")

# Sidebar info
with st.sidebar:
    st.header("📘 About")
    st.write("""
        This app uses machine learning to predict whether a user is likely to purchase a ticket.
        Fill in the details below and get predictions instantly.
    """)

# Form layout
with st.form("prediction_form"):
    st.markdown("### 📝 User Input Features")

    col1, col2 = st.columns(2)

    with col1:
        preferred_device = st.selectbox("Preferred Device", [9, 8])
        preferred_location_type = st.selectbox("Preferred Location Type", [0, 1, 2])
        following_company_page = st.selectbox("Following Company Page", [0, 1])
        working_flag = st.selectbox("Working Flag", [0, 1])
        Adult_flag = st.selectbox("Adult Flag", [0, 1])
        travelling_network_rating = st.slider("Travelling Network Rating", 1, 4, 2)

    with col2:
        Yearly_avg_view_on_travel_page = st.number_input("Yearly Avg Views on Travel Page", min_value=0.0)
        total_likes_on_outstation_checkin_given = st.number_input("Total Likes Given on Outstation Check-ins", min_value=0.0)
        yearly_avg_Outstation_checkins = st.number_input("Yearly Avg Outstation Check-ins", min_value=0.0)
        member_in_family = st.number_input("Family Members", min_value=0)
        Yearly_avg_comment_on_travel_page = st.number_input("Yearly Avg Comments on Travel Page", min_value=0.0)
        total_likes_on_outofstation_checkin_received = st.number_input("Total Likes Received on Outstation Check-ins", min_value=0)
        week_since_last_outstation_checkin = st.number_input("Weeks Since Last Outstation Check-in", min_value=0)
        montly_avg_comment_on_company_page = st.number_input("Monthly Avg Comments on Company Page", min_value=0)
        Daily_Avg_mins_spend_on_traveling_page = st.number_input("Daily Avg Mins on Traveling Page", min_value=0)

    submitted = st.form_submit_button("🔍 Predict")

# Run prediction
if submitted:
    input_data = pd.DataFrame([[
        Yearly_avg_view_on_travel_page,
        preferred_device,
        total_likes_on_outstation_checkin_given,
        yearly_avg_Outstation_checkins,
        member_in_family,
        preferred_location_type,
        Yearly_avg_comment_on_travel_page,
        total_likes_on_outofstation_checkin_received,
        week_since_last_outstation_checkin,
        following_company_page,
        montly_avg_comment_on_company_page,
        working_flag,
        travelling_network_rating,
        Adult_flag,
        Daily_Avg_mins_spend_on_traveling_page
    ]], columns=[
        'Yearly_avg_view_on_travel_page',
        'preferred_device',
        'total_likes_on_outstation_checkin_given',
        'yearly_avg_Outstation_checkins',
        'member_in_family',
        'preferred_location_type',
        'Yearly_avg_comment_on_travel_page',
        'total_likes_on_outofstation_checkin_received',
        'week_since_last_outstation_checkin',
        'following_company_page',
        'montly_avg_comment_on_company_page',
        'working_flag',
        'travelling_network_rating',
        'Adult_flag',
        'Daily_Avg_mins_spend_on_traveling_page'
    ])

    prediction = model.predict(input_data)[0]
    proba = model.predict_proba(input_data)[0]

    # Display result
    if prediction == 1:
        st.success("✅ The user is likely to purchase a ticket!")
    else:
        st.error("❌ The user is unlikely to purchase a ticket.")

    # Display prediction probabilities as pie chart
    fig1, ax1 = plt.subplots()
    ax1.pie(proba, labels=['No', 'Yes'], autopct='%1.1f%%', startangle=90, colors=['#ff9999','#66b3ff'])
    ax1.axis('equal')  # Equal aspect ratio ensures pie is drawn as a circle.
    st.markdown("#### 🎯 Prediction Confidence")
    st.pyplot(fig1)

    # Display input features as bar chart
    st.markdown("#### 📊 Input Features Overview")
    fig2, ax2 = plt.subplots(figsize=(10, 4))
    ax2.bar(input_data.columns, input_data.iloc[0], color='skyblue')
    plt.xticks(rotation=45, ha='right')
    ax2.set_ylabel("Value")
    ax2.set_title("User Engagement Features")
    st.pyplot(fig2)
