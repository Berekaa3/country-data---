
import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

st.set_page_config(layout='wide')
st.title('Country Clusters Analysis')

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
