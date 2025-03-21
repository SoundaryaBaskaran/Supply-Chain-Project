# 📦 Supply Chain Management Analysis – DataCo Dataset

## 📂 Project Presentation  
[📄 Click here to view the Supply Chain Project PPT](https://view.officeapps.live.com/op/view.aspx?src=https%3A%2F%2Fraw.githubusercontent.com%2FSoundaryaBaskaran%2FSupply-Chain-Project%2Frefs%2Fheads%2Fmain%2FSupply%2520Chain%2520Management%2520Analysis%2520%25E2%2580%2593%2520Dataco%2520Dataset%2520ppt.pptx&wdOrigin=BROWSELINK)


## 🚀 Introduction
Supply Chain Management (SCM) is essential for optimizing product flow from suppliers to customers. This project analyzes **DataCo's supply chain dataset** to identify inefficiencies, improve demand forecasting, and reduce late deliveries using **Machine Learning (ML) models**. The insights derived from this analysis help businesses make data-driven decisions and enhance operational efficiency.

---

## 🎯 Problem Statement
DataCo faces significant supply chain challenges:
- 📉 Inaccurate Sales Predictions & Order Quantities leading to inventory issues.
- ⏳ **Late Deliveries** causing customer dissatisfaction.
- 💰 **Fraudulent Transactions** reducing profit margins.
  
**Goal:** Optimize supply chain operations by mitigating these risks and improving efficiency.

---

## 📊 Project Objectives
✔ Improve **demand forecasting** to optimize inventory levels.
✔ Detect **fraudulent transactions** and mitigate risks.
✔ Identify **sales trends** to enhance product and regional performance insights.
✔ Reduce **delivery delays** and improve customer satisfaction.
✔ Support **data-driven decision-making** for supply chain management.

---

## 🗂 Dataset Overview
The dataset contains **2015-2018 supply chain records**, including:
- 📍 **Market Share by Region** – Sales distribution across different regions.
- 🎯 **Product Profitability** – Identifying profitable and loss-making products.
- 📅 **Sales Trends** – Weekly, monthly, and seasonal sales patterns.
- 💳 **Payment Modes** – Understanding customer payment preferences.
- 🕵 **Fraudulent Transactions** – Identifying fraud-prone customers and products.
- 🚚 **Late Deliveries** – Analyzing delays by product category and shipping method.
- 🎯 **Customer Segmentation** – Using RFM analysis for customer behavior insights.

---

## 🔍 Key Insights & Findings
### 📍 **Market Share by Region**
- 🌍 **Europe & LATAM lead** in sales per customer.
- 🌎 **Africa & USCA have lower sales**, indicating untapped growth potential.

### 💰 **Profitable & Loss-Making Products**
- ✅ **Fitness & Sports gear** (Nike, Under Armour) are most profitable.
- ❌ **Cleats & Footwear** show high losses, indicating pricing or overstock issues.

### 📈 **Sales Trends & Seasonal Patterns**
- 📅 **Peak Sales:** Fridays & November (Holiday promotions).
- 🔽 **Lowest Sales:** Tuesdays.
- ⏰ **High Activity:** Early mornings & late afternoons.

### 🔐 **Fraud Detection Insights**
- 🚩 Most fraud occurs in **Western Europe, Central & South America**.
- 🚨 **Men’s Footwear & Cleats** are the most targeted items.
- 👤 Customer "Mary Smith" shows **unusually high fraud cases**.

### 🚛 **Late Deliveries by Category & Shipping Method**
- **Cleats, Men’s Footwear, Women’s Apparel** face the most delays.
- **Standard Class Shipping** has the highest late deliveries, while **Same-Day** performs best.

### 🏆 **Customer Segmentation (RFM Analysis)**
- 👑 **Loyal Customers (10.5%)** – Strengthen relationships with loyalty programs.
- 🎖 **Champions (0.6%)** – VIP customers; leverage referrals.
- 🛍 **At-Risk Customers (11.4%)** – Require retention strategies.
- ❌ **Lost Customers (4.4%)** – Win-back offers needed.

---



## 🔄 ETL Pipeline Implementation

### 1️⃣ Stored Cleaned Data in MongoDB

- The cleaned supply chain dataset was stored in a MongoDB collection (cleaned_data) for processing.

### 2️⃣ ETL Pipeline Setup

- Organized the project into structured modules:

### 3️⃣ Extract Phase (src/extract.py)

- Extracted relevant fields from MongoDB, ignoring _id.

- Extracted 180,519 rows from MongoDB.

### 4️⃣ Transform Phase (src/transform.py)

- Applied meaningful transformations:

- Shipping Delay Calculation → (Days for shipping (real) - Days for shipment (scheduled))

- Customer Order Frequency → Count of orders per customer.

- Transformed 180,519 rows.

### 5️⃣ Load Phase (src/load.py)

- Stored transformed data in MongoDB under transformed_supply_chain collection.

- Loaded 180,519 records into transformed collection.

### 6️⃣ Final ETL Pipeline (src/etl_pipeline.py)

- Integrated Extract → Transform → Load into a single pipeline.

- Ensured data isn't reprocessed multiple times.

- Successfully executed full ETL pipeline!

---

## 🤖 Machine Learning Models Used
### **📌 Regression Models for Sales & Order Quantity Prediction**
- 🔹 **Models Trained** : Linear Regression, Ridge, Lasso, Random Forest, XGBoost
- 🔹 **Linear Regression** (Best for Sales, MAE: 0.0005, RMSE: 0.0014)
- 🔹 **Decision Tree** (Best for Quantity, MAE: 0.0040, RMSE: 0.006)

### **📌 Classification Models for Fraud & Late Deliveries**
- 🔹 **Models Trained** : Logistic Regression, Decision Tree, Random Forest, XGBoost, KNN, SVM
- 🔹 **Random Forest** – Best for **Fraud Detection** (Recall: 98.93%, Accuracy: 98.66%)
- 🔹 **Decision Tree** – Best for **Late Delivery Prediction** (Accuracy: 99.37%, F1 Score: 99.42%)

### **Model Improvement Techniques**
✔ **Cross-validation** for better generalization.
✔ **Feature Importance** analysis to refine prediction accuracy.

---

## 🚀 Deployment

- The project is deployed using Streamlit for interactive visualization.

- Docker is used for containerization, ensuring a portable and consistent environment.

---

## ⚙ How to Run This Project
### **🔧 Installation Steps**
1️⃣ Clone the repository:
   ```sh
   git clone https://github.com/SoundaryaBaskaran/Supply-Chain-Project.git
   cd Supply-Chain-Project
   ```
2️⃣ Install dependencies:
   ```sh
   pip install -r requirements.txt
   ```
### **📊 For Data Analysis & Model Training**
3️⃣ Run the Jupyter Notebook:
  ```sh
    jupyter notebook
  ```
    
Open the notebook file (.ipynb) and execute the scripts step by step to analyze the data and train models.

### **🚀 For Running the Deployed Application**
4️⃣ Run the Deployment (app.py):
  ```sh
    streamlit run app.py
  ```
    
Once the app is running, open the provided local URL in your browser to interact with the application.






---

## 📢 Business Recommendations
✔ **AI-driven demand forecasting** to prevent stock issues.  
✔ **Fraud detection models** to improve security & reduce financial risks.  
✔ **Optimized logistics strategies** to minimize late deliveries.  
✔ **Regional marketing expansion** in **Europe & LATAM**.  
✔ **Personalized retention offers** for at-risk customers.  
✔ **Multi-supplier strategy** for supply chain resilience.  

---

## 📬 Contact
🔗 **GitHub**: [https://github.com/SoundaryaBaskaran](https://github.com/SoundaryaBaskaran)  
🔗 **LinkedIn**: [SoundayaBaskaran](https://www.linkedin.com/in/soundaryabaskaran/)  
🔗 **Medium**: [SoundayaBaskaran](https://medium.com/@soundarya_baskaran)


🙌 **If you find this project useful, don’t forget to ⭐ the repository!** 🚀

