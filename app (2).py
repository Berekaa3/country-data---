
import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import joblib

st.set_page_config(layout='wide')
st.title('Country Clusters Analysis')

# Define the exact features used for scaling in the notebook
# This ensures consistency even if CSV column order changes or intermediate columns are present
# Based on the notebook's df after dropping 'country' and adding 'country_encoded'
SCALED_FEATURES = ['child_mort', 'exports', 'health', 'imports', 'income', 'inflation', 'life_expec', 'total_fer', 'gdpp', 'country_encoded']

# Load the models and scaler
try:
    kmeans_model = joblib.load('kmeans_model.joblib')
    scaler = joblib.load('scaler.joblib')
except FileNotFoundError:
    st.error("Error: Model files (kmeans_model.joblib or scaler.joblib) not found. Please ensure they are saved in the same directory.")
    st.stop()


st.write('### Overview of Clustered Countries')

# Load the clustered data
try:
    df_clustered = pd.read_csv('clustered_country_data.csv')
except FileNotFoundError:
    st.error("Error: clustered_country_data.csv not found. Please ensure it is saved in the same directory.")
    st.stop()


st.write('First 5 rows of the clustered data:')
st.dataframe(df_clustered.head())

st.write('### Clusters based on GDP per Capita and Income')

# Create the scatter plot
fig, ax = plt.subplots(figsize=(10, 7))
sns.scatterplot(data=df_clustered, x='gdpp', y='income', hue='cluster_label', palette='viridis', s=100, alpha=0.8, ax=ax)
ax.set_title('Clusters of Countries (GDP per Capita vs Income)')
ax.set_xlabel('GDP per Capita')
ax.set_ylabel('Income')
ax.grid(True)

st.pyplot(fig)

st.write("The scatter plot above visualizes the countries grouped into clusters based on their GDP per capita and income levels. Each color represents a different cluster, helping us understand the economic profiles of various countries.")

# Display Cluster Characteristics
st.write('### Cluster Characteristics (Centroids)')

# Get centroids and inverse transform them
centroids_scaled = kmeans_model.cluster_centers_

# Ensure centroids_df uses the correct feature names
centroids_df = pd.DataFrame(scaler.inverse_transform(centroids_scaled), columns=SCALED_FEATURES)
centroids_df.index.name = 'Cluster Label'
st.dataframe(centroids_df)

st.write("These are the average characteristics (centroids) for each cluster, transformed back to their original scales. This helps in understanding what defines each cluster.")


# Interactive Classification for a new country
st.write('### Classify a New Country')
st.write('Enter values for a hypothetical country to see which cluster it belongs to:')

input_data = {}
# Features that the user will input
user_input_features = ['child_mort', 'exports', 'health', 'imports', 'income', 'inflation', 'life_expec', 'total_fer', 'gdpp']

for feature in user_input_features:
    mean_val = df_clustered[feature].mean()
    input_data[feature] = st.number_input(f'Enter value for {feature}', value=float(f'{mean_val:.2f}'), step=0.1)

# Add 'country_encoded' with a placeholder value, as it was part of the scaled features
# This ensures the input DataFrame has the same number and order of columns as SCALED_FEATURES
input_data['country_encoded'] = 0 # This value won't affect clustering based on other features much for a single prediction

if st.button('Predict Cluster'):
    # Create a DataFrame from the input data, explicitly ensuring column order matches SCALED_FEATURES
    # This is critical for the scaler.transform to work correctly
    input_df = pd.DataFrame([input_data], columns=SCALED_FEATURES)

    # Scale the input data
    input_scaled = scaler.transform(input_df)

    # Predict the cluster
    predicted_cluster = kmeans_model.predict(input_scaled)[0]
    st.success(f'The hypothetical country belongs to Cluster: {predicted_cluster}')
