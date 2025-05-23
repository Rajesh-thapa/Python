import streamlit as st
import pandas as pd
import numpy as np
import pickle

# Load the trained model
model = pickle.load(open('model.pkl', 'rb'))

st.title("AeroReach Ticket Purchase Prediction")

# Collect inputs from user
preferred_device = st.selectbox("Preferred Device", [9, 8])  # Use encoded values
preferred_location_type = st.selectbox("Preferred Location Type", [0, 1, 2])
following_company_page = st.selectbox("Following Company Page", [0, 1])
working_flag = st.selectbox("Working Flag", [0, 1])
Adult_flag = st.selectbox("Adult Flag", [0, 1])
Yearly_avg_view_on_travel_page = st.number_input("Yearly Avg Views on Travel Page", min_value=0.0)
total_likes_on_outstation_checkin_given = st.number_input("Total Likes on Outstation Checkin Given", min_value=0.0)
yearly_avg_Outstation_checkins = st.number_input("Yearly Avg Outstation Checkins", min_value=0.0)
member_in_family = st.number_input("Member in Family", min_value=0)
Yearly_avg_comment_on_travel_page = st.number_input("Yearly Avg Comment on Travel Page", min_value=0.0)
total_likes_on_outofstation_checkin_received = st.number_input("Total Likes on Outstation Checkin Received", min_value=0)
week_since_last_outstation_checkin = st.number_input("Weeks Since Last Outstation Checkin", min_value=0)
montly_avg_comment_on_company_page = st.number_input("Monthly Avg Comment on Company Page", min_value=0)
travelling_network_rating = st.number_input("Travelling Network Rating", min_value=0)
Daily_Avg_mins_spend_on_traveling_page = st.number_input("Daily Avg Mins on Traveling Page", min_value=0)

# Combine into a DataFrame
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

# Make prediction
if st.button("Predict"):
    prediction = model.predict(input_data)[0]
    if prediction == 1:
        st.success("✅ User is likely to purchase a ticket!")
    else:
        st.error("❌ User is unlikely to purchase a ticket.")

