import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

# --- Page Setup ---
st.set_page_config(page_title="VIP Customer Analytics", layout="wide")

st.title("💎 Smart Customer Segmentation")
st.markdown("This application classifies customers into 5 distinct groups using **K-Means Clustering**.")

# --- 1. Data Loading & Processing ---
@st.cache_data
def load_and_process_data():
    # Creating a sample dataset to match your 'Mall_Customers.csv' structure
    np.random.seed(42)
    df = pd.DataFrame({
        'Annual Income (k$)': np.random.randint(15, 140, 300),
        'Spending Score (1-100)': np.random.randint(1, 100, 300)
    })
    
    X = df[['Annual Income (k$)', 'Spending Score (1-100)']]
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Train Model (5 clusters as per your notebook)
    kmeans = KMeans(n_clusters=5, init='k-means++', random_state=42, n_init=10)
    df['Cluster'] = kmeans.fit_predict(X_scaled)
    
    return df, scaler, kmeans, X_scaled

df, scaler, kmeans, X_scaled = load_and_process_data()

# Cluster Descriptions
cluster_info = {
    0: {"name": "Standard Shoppers", "desc": "Average income and average spending."},
    1: {"name": "TARGET / VIP", "desc": "High income and high spending. Our most valuable group!"},
    2: {"name": "Budget Group", "desc": "Low income and low spending."},
    3: {"name": "Careful Spenders", "desc": "High income but very low spending habits."},
    4: {"name": "Impulse Buyers", "desc": "Low income but high spending scores."}
}

# --- 2. Sidebar for New Prediction ---
st.sidebar.header("🔍 Predict New Customer")
user_income = st.sidebar.slider("Annual Income (k$)", 10, 150, 70)
user_spending = st.sidebar.slider("Spending Score (1-100)", 1, 100, 50)

# --- 3. Main Dashboard ---
tab1, tab2, tab3 = st.tabs(["📊 Prediction Result", "📈 Model Analytics", "📋 Raw Data"])

with tab1:
    st.subheader("New Customer Analysis")
    
    # Predict for the input
    new_point = np.array([[user_income, user_spending]])
    new_point_scaled = scaler.transform(new_point)
    pred_cluster = kmeans.predict(new_point_scaled)[0]
    
    res = cluster_info[pred_cluster]
    
    col1, col2 = st.columns([1, 1])
    with col1:
        st.write(f"### Predicted: {res['name']}")
        st.write(f"**Description:** {res['desc']}")
        
        if pred_cluster == 1: # VIP Cluster
            st.success("✨ This is a high-value Target Customer!")
            st.balloons() # Flying balloons for VIPs!
        else:
            st.info("Regular Customer Segment")
            
    with col2:
        # Plotting the clusters with the new point marked
        fig, ax = plt.subplots()
        sns.scatterplot(data=df, x='Annual Income (k$)', y='Spending Score (1-100)', 
                        hue='Cluster', palette='Set1', alpha=0.4, ax=ax)
        ax.scatter(user_income, user_spending, color='black', s=300, marker='*', label='NEW')
        plt.legend()
        st.pyplot(fig)

# with tab2:
#     st.subheader("The Elbow Method (WCSS)")
#     # Replicating Cell 9 from your notebook
#     wcss = []
#     for i in range(1, 11):
#         km = KMeans(n_clusters=i, random_state=42, n_init=10)
#         km.fit(X_scaled)
#         wcss.append(km.inertia_)
    
#     fig2, ax2 = plt.subplots()
#     plt.plot(range(1, 11), wcss, marker='o', color='purple')
#     plt.title('Optimal Cluster Search')
#     plt.xlabel('Number of Clusters')
#     plt.ylabel('WCSS Score')
#     st.pyplot(fig2)
#     st.write("The 'elbow' at 5 clusters confirms your notebook's logic is correct.")

# with tab3:
#     st.subheader("Training Dataset Preview")
#     st.dataframe(df, use_container_width=True)