Smart Customer Segmentation 💎

A Machine Learning web application that performs Customer Segmentation using the K-Means Clustering Algorithm.
This project helps businesses identify different customer groups based on their Annual Income and Spending Score.

The application is built using Python, Scikit-learn, and Streamlit with an interactive dashboard for customer analysis and prediction.

Live Demo link : https://anand-kmeanscluster.streamlit.app/

🚀 Features
Customer segmentation using K-Means Clustering
Interactive Streamlit dashboard
Predicts customer category instantly
Visual cluster representation
Real-time customer analytics
Beginner-friendly Machine Learning project
🛠️ Technologies Used
Python
Streamlit
Pandas
NumPy
Scikit-learn
Matplotlib
Seaborn
📂 Project Structure
├── app.py                  # Streamlit web application
├── Mall_Customers.csv      # Dataset
├── kmeans_model.pkl        # Trained KMeans model
├── scaler.pkl              # StandardScaler object
├── Project_KMeans.ipynb    # Jupyter Notebook for model training
├── requirements.txt        # Required libraries
└── README.md               # Project documentation
📊 Dataset

The dataset contains customer information such as:

Annual Income (k$)
Spending Score (1-100)

These features are used to group customers into different clusters.

🧠 Machine Learning Algorithm

This project uses the K-Means Clustering Algorithm to divide customers into 5 different segments.

Customer groups include:

VIP Customers
Standard Shoppers
Budget Customers
Careful Spenders
Impulse Buyers
⚙️ Installation
1. Clone the Repository
git clone https://github.com/your-username/customer-segmentation.git
cd customer-segmentation
2. Install Dependencies
pip install -r requirements.txt
▶️ Run the Application
streamlit run app.py

The application will automatically open in your browser.

📈 Application Workflow
User enters:
Annual Income
Spending Score
The model predicts the customer cluster
Application displays:
Customer Segment Name
Customer Description
Cluster Visualization

The Streamlit application implementation is available in app.py.

📊 Visualization

The dashboard includes:

Customer cluster scatter plots
New customer prediction marker
Segment-based analytics
📦 Requirements

Required libraries are listed in requirements.txt.

Main libraries include:

streamlit
pandas
matplotlib
seaborn
scikit-learn
🎯 Future Improvements
Add more clustering algorithms
Deploy using Streamlit Cloud
Add advanced analytics dashboard
Include customer recommendation system
Improve UI/UX design
👨‍💻 Author

Anand

📜 License

This project is open-source and available under the MIT License.
