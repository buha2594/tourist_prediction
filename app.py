import streamlit as st
import pandas as pd
import numpy as np
import pickle
import joblib
import scipy.sparse

import base64

# Background Image CSS
def set_background(image_path):
    image_url = f"data:image/png;base64,{image_path}"
    bg_style = f"""
    <style>
    .stApp {{
        background-image: url("{image_url}");
        background-size: cover;
        background-position: center;
        background-repeat: no-repeat;
    }}
    </style>
    """
    st.markdown(bg_style, unsafe_allow_html=True)

# Load and Encode Image
import base64
def get_base64_image(image_path):
    with open(image_path, "rb") as img_file:
        return base64.b64encode(img_file.read()).decode()

# Apply Background
image_base64 = get_base64_image("11.png")  # Change this to your image filename
set_background(image_base64)



# Load Models
with open("visit_mode_model_16.pkl", "rb") as f:
    rating_model = pickle.load(f)


# -------------------- Load Data and Models --------------------

@st.cache_data
def load_data():
    """Load the dataset."""
    df = pd.read_csv("debug6-output.csv")
    return df

@st.cache_data
def load_visit_mode_model():
    """Load the visit mode prediction model."""
    with open("visit_mode_model_16.pkl", "rb") as f:
        return pickle.load(f)

@st.cache_data
def load_sparse_similarity():
    """Load the sparse cosine similarity matrix."""
    return scipy.sparse.load_npz("sparse_cosine_similarity_16-03.npz")

@st.cache_data
def load_vectorizer():
    """Load the TF-IDF vectorizer."""
    return joblib.load("tfidf_vectorizer_16-03.pkl")


# Load all required data and models
df = load_data()
rating_model = load_visit_mode_model()
cosine_sim_sparse = load_sparse_similarity()
vectorizer = load_vectorizer()

#------------------Sidebar Navigation--------------------------------
st.sidebar.title("🔍 Navigation")
page = st.sidebar.radio("Go to", ["Visit Mode Prediction", "Attraction Recommendations"])
# Ensure required columns exist


# -------------------------- PAGE: VISIT MODE PREDICTION --------------------------
if page == "Visit Mode Prediction":
    st.title("🚀 Visit Mode Prediction")

    # Country selection
    countries = df["AttractionCountry"].dropna().unique().tolist()
    selected_country = st.selectbox("🏳️ Select Country", ["Select"] + countries)

    # Filter regions based on selected country
    filtered_regions = df.loc[df["AttractionCountry"] == selected_country, "AttractionRegion"].dropna().unique().tolist() if selected_country != "Select" else []
    selected_region = st.selectbox("🌍 Select Region", ["Select"] + filtered_regions)

    # Filter cities based on selected country & region
    filtered_cities = df.loc[
        (df["AttractionCountry"] == selected_country) & (df["AttractionRegion"] == selected_region),
        "AttractionCityName"
    ].dropna().unique().tolist() if selected_region != "Select" else []
    selected_city = st.selectbox("🏙️ Select City", ["Select"] + filtered_cities)

    # Filter attractions based on selected city
    filtered_attractions = df.loc[
        (df["AttractionCountry"] == selected_country) & 
        (df["AttractionRegion"] == selected_region) & 
        (df["AttractionCityName"] == selected_city), 
        "attraction"
    ].dropna().unique().tolist() if selected_city != "Select" else []
    selected_attraction = st.selectbox("🎡 Select Attraction", ["Select"] + filtered_attractions)

    # Numeric Inputs for Visit Year and Month
    visit_year = st.number_input("📅 Visit Year", min_value=2000, max_value=2025, step=1)
    visit_month = st.number_input("📆 Visit Month", min_value=1, max_value=12, step=1)

    if st.button("Predict Visit Mode"):
        try:
            # Extract corresponding IDs safely
            country_id = df.loc[df["AttractionCountry"] == selected_country, "countryid"].values[0]
            region_id = df.loc[df["AttractionRegion"] == selected_region, "regionid"].values[0]
            city_id = df.loc[df["AttractionCityName"] == selected_city, "cityid"].values[0]
            attraction_id = df.loc[df["attraction"] == selected_attraction, "attractionid"].values[0]
            continent_id = df.loc[df["AttractionCountry"] == selected_country, "contenentid"].values[0]
            attraction_city_id = df.loc[df["AttractionCityName"] == selected_city, "AttractionCityId"].values[0]
            

            # Create DataFrame for prediction
            input_data = pd.DataFrame(
                [[visit_year, visit_month, attraction_id, continent_id, region_id, country_id, city_id, attraction_city_id]],
                columns=['visityear', 'visitMonth', 'attractionid', 'contenentid', 'regionid', 'countryid', 'cityid', 'AttractionCityId']
            )

            # Convert Data Types
            input_data = input_data.astype(int).astype(np.float32)

            
            # Make prediction
            prediction = rating_model.predict(input_data.values.reshape(1, -1))[0]

            # Map prediction to visit mode labels
            visit_mode_mapping = {0: "Business", 1: "Family", 2: "Couples", 3: "Friends"}  # Adjust as per dataset
            predicted_visit_mode = visit_mode_mapping.get(prediction, "Unknown")

            # Display the result
            st.success(f"🎯 Predicted Visit Mode: **{predicted_visit_mode}**")

        except IndexError:
            st.error("⚠️ Unable to retrieve IDs. Please check the selected inputs.")
       
            # Make prediction
            prediction = rating_model.predict(input_data.values.reshape(1, -1))[0]


            # Map prediction to visit mode label
            visit_modes = {0: "Unknown", 1: "Business", 2: "couples", 3:"Family",4:'Friends', 5:"Solo"}  
            predicted_mode = visit_modes.get(prediction, "Unknown")

            st.success(f"🚗 Predicted Visit Mode: **{predicted_mode}**")

        except IndexError:
            st.error("⚠️ Please make sure all selections are valid.")

    # Display selections
    st.write(f"You selected: {selected_country} > {selected_region} > {selected_city} > {selected_attraction} ")
        
    # -------------------- Page: Attraction Recommendations --------------------
elif page == "Attraction Recommendations":
    st.title("🌍 Travel Attraction Recommendation")

    # Dropdowns for country & city selection
    country_list = df["AttractionCountry"].unique().tolist()
    selected_country = st.selectbox("🏳️ Select Country", ["Select"] + country_list, key="country_rec")

    if selected_country != "Select":
        city_list = df[df["AttractionCountry"] == selected_country]["AttractionCityName"].unique().tolist()
    else:
        city_list = []

    selected_city = st.selectbox("🏙️ Select City", ["Select"] + city_list, key="city_rec")

    if selected_city != "Select":
        attractions = df[df["AttractionCityName"] == selected_city]["attraction"].unique().tolist()
    else:
        attractions = []

    selected_attraction = st.selectbox("🎡 Select Attraction", ["Select"] + attractions, key="attraction_rec")

    if st.button("🔍 Get Recommendations"):
        if selected_attraction != "Select":
            try:
                # Find index of selected attraction
                attraction_idx = df[df["attraction"] == selected_attraction].index[0]

                # Get similarity scores
                sim_scores = cosine_sim_sparse[attraction_idx].toarray().flatten()

                # Get top 10 similar attractions (ensuring we get enough unique ones)
                top_indices = np.argsort(sim_scores)[::-1]  # Sort in descending order

                # Store unique recommendations
                recommended_attractions = []
                seen = set()

                for i in top_indices:
                    attraction_name = df.iloc[i]["attraction"]
                    country_name = df.iloc[i]["AttractionCountry"]

                    if attraction_name != selected_attraction and attraction_name not in seen:
                        recommended_attractions.append((attraction_name, country_name))
                        seen.add(attraction_name)

                    if len(recommended_attractions) == 5:  # Ensure exactly 5 recommendations
                        break

                # Display recommendations
                if recommended_attractions:
                    st.success("✅ Recommended Attractions:")
                    for attraction, country in recommended_attractions:
                        st.write(f"🔹 {attraction} ({country})")
                else:
                    st.error("⚠️ No recommendations found. Try another attraction.")

            except IndexError:
                st.error("⚠️ Please make sure the selection is valid.")
        else:
            st.error("⚠️ Please select an attraction first.")

                