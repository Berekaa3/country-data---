
import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import joblib

st.set_page_config(layout='wide')
st.title('Country Clusters Analysis')

# Load the models and scaler
kmeans_model = joblib.load('kmeans_model.joblib')
scaler = joblib.load('scaler.joblib')

st.write('### Overview of Clustered Countries')

# Load the clustered data
df_clustered = pd.read_csv('clustered_country_data.csv')

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
# We need the original column names for the inverse transform
# The original df columns were: child_mort, exports, health, imports, income, inflation, life_expec, total_fer, gdpp, country_encoded
# When scaling, 'country_encoded' was included. So, we need to create a dummy dataframe with the same columns for inverse_transform.

# Create a dummy dataframe to inverse transform the centroids
# Assuming the order of columns remains the same as when the scaler was fitted
dummy_df_for_scaler = df_clustered.drop(columns=['cluster_label']).iloc[0:0] # empty dataframe with correct columns

# Prepare centroids for inverse transformation: need to match the original df shape for columns
# The scaler was fitted on df (which had 'country_encoded'). The app takes numeric inputs.
# Let's assume for this demonstration, we'll inverse transform all features used in scaling.
# The order of features in the scaler is crucial. Based on the notebook, it's all columns of `df` after dropping 'country' and before adding 'cluster_label'.

# This is a bit tricky if 'country_encoded' is also scaled. Let's make sure the inverse transform is on the relevant columns.
# For now, let's inverse transform all scaled features for simplicity and then just display the relevant ones.

# Get feature names from the original dataframe before scaling to match the scaler's features
# This assumes the order of columns is preserved when X_scaled was created.
original_features = pd.read_csv('clustered_country_data.csv').drop(columns=['country_encoded', 'cluster_label']).columns.tolist() + ['country_encoded']

centroids_original_scale = scaler.inverse_transform(centroids_scaled)
centroids_df = pd.DataFrame(centroids_original_scale, columns=original_features)
centroids_df.index.name = 'Cluster Label'
st.dataframe(centroids_df)

st.write("These are the average characteristics (centroids) for each cluster, transformed back to their original scales. This helps in understanding what defines each cluster.")


# Interactive Classification for a new country
st.write('### Classify a New Country')
st.write('Enter values for a hypothetical country to see which cluster it belongs to:')

# Create input fields for each feature (excluding country_encoded for user input)
input_data = {}
# Use the features from the original_features list, excluding 'country_encoded' for user input directly.
# The original data in df was child_mort, exports, health, imports, income, inflation, life_expec, total_fer, gdpp.
# Then country_encoded was added. The scaler was applied to all of these. 
# For user input, we'll ask for the numeric ones.

user_input_features = ['child_mort', 'exports', 'health', 'imports', 'income', 'inflation', 'life_expec', 'total_fer', 'gdpp']

for feature in user_input_features:
    # Using a default from the mean of the original df for better starting points
    mean_val = df_clustered[feature].mean()
    input_data[feature] = st.number_input(f'Enter value for {feature}', value=float(f'{mean_val:.2f}'), step=0.1)

# A dummy value for country_encoded, as it's not a user-inputted characteristic for classification
# We need to maintain the same number of features as the scaler was trained on.
input_data['country_encoded'] = 0 # Placeholder, as it's not a characteristic to classify upon

if st.button('Predict Cluster'):
    # Create a DataFrame from the input data, ensuring correct column order
    input_df = pd.DataFrame([input_data], columns=original_features)

    # Scale the input data
    input_scaled = scaler.transform(input_df)

    # Predict the cluster
    predicted_cluster = kmeans_model.predict(input_scaled)[0]
    st.success(f'The hypothetical country belongs to Cluster: {predicted_cluster}')
