import streamlit as st
import numpy as np
import joblib

# ==============================
# LOAD MODEL
# ==============================
model = joblib.load("model_inr.pkl")
encoders = joblib.load("encoders_inr.pkl")

st.set_page_config(page_title="Movie Success Predictor", layout="centered")

st.title(" Movie Success Predictor")
st.write("Model Prediction + Real-world Business Analysis")

# ==============================
# INPUTS
# ==============================
genre = st.selectbox("Genre", encoders['Genre'].classes_)
stability = st.selectbox("Screen Stability", encoders['Screen_Stability'].classes_)

budget = st.slider("Budget (₹ Crore)", 10, 500, 100)
marketing_budget = st.slider("Marketing Budget (₹ Crore)", 5, 200, 50)

screens = st.slider("Screens", 200, 30000, 3000)
seats = st.slider("Seats per Screen", 50, 300, 120)

weeks = st.slider("Weeks in Theatre", 2, 12, 6)
shows = st.slider("Shows per Day", 2, 15, 5)

ticket_price = st.slider("Ticket Price (₹)", 100, 800, 250)

actor_popularity = st.slider("Actor Popularity (1-10)", 1, 10, 6)
director_success = st.slider("Director Success (1-10)", 1, 10, 6)

occupancy = st.slider("Audience Occupancy (%)", 20, 100, 60) / 100

# ==============================
# ENCODING
# ==============================
genre_encoded = encoders['Genre'].transform([genre])[0]
stability_encoded = encoders['Screen_Stability'].transform([stability])[0]

feature_order = [
    'Genre','Budget','Screens','Weeks','Ticket_Price','Shows',
    'Screen_Stability','Actor_Popularity','Director_Success',
    'Marketing_Budget','Occupancy','Seats_Per_Screen'
]

input_dict = {
    'Genre': genre_encoded,
    'Budget': budget,
    'Screens': screens,
    'Weeks': weeks,
    'Ticket_Price': ticket_price,
    'Shows': shows,
    'Screen_Stability': stability_encoded,
    'Actor_Popularity': actor_popularity,
    'Director_Success': director_success,
    'Marketing_Budget': marketing_budget,
    'Occupancy': occupancy,
    'Seats_Per_Screen': seats
}

input_data = np.array([[input_dict[col] for col in feature_order]])

# ==============================
# PREDICTION
# ==============================
if st.button("Predict"):

    #  MODEL
    prediction = model.predict(input_data)[0]
    probability = model.predict_proba(input_data)[0][prediction]

    st.subheader(" AI Model Prediction")

    if prediction == 1:
        st.success(f"HIT  (Confidence: {probability*100:.2f}%)")
    else:
        st.error(f"FLOP  (Confidence: {probability*100:.2f}%)")

    # ==============================
    #  BUSINESS CALCULATION
    # ==============================
    days = weeks * 7

    total_shows = screens * shows * days
    total_seats = total_shows * seats
    tickets_sold = total_seats * occupancy

    revenue = (tickets_sold * ticket_price) / 10000000  # ₹ Crore
    total_cost = budget + marketing_budget

    st.subheader(" Business Analysis")

    st.write(f"Estimated Revenue: ₹{revenue:.2f} Cr")
    st.write(f"Total Cost: ₹{total_cost:.2f} Cr")

    # ==============================
    # FINAL VERDICT
    # ==============================
    st.subheader(" Business Verdict")

    if revenue >= 1.5 * budget:
        st.success(" HIT")
    elif revenue >= budget:
        st.warning(" AVERAGE")
    else:
        st.error(" FLOP")

    
    # COMPARISON
    
    st.subheader(" Model vs Business")

    if prediction == 1 and revenue < budget:
        st.warning("Model says HIT but business shows FLOP ")
    elif prediction == 0 and revenue >= budget:
        st.warning("Model says FLOP but business looks good ")
    else:
        st.success("Model & Business agree ")
