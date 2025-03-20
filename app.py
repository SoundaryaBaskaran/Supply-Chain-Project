import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import joblib

st.set_page_config(page_title="Supply Chain Dashboard", layout="wide")

# ---- Apply Dark Mode Theme ----
st.markdown("""
    <style>
    body {
        background-color: #121212;
        color: white;
    }
    .stApp {
        background-color: #1e1e1e;
    }
    </style>
""", unsafe_allow_html=True)

# Load the dataset
@st.cache_data
def load_data():
    df = pd.read_csv("data\cleaned_data.csv", encoding="ISO-8859-1")
    return df

df = load_data()

# ---- Load Trained Models ----
sales_model = joblib.load("regression_sales.pkl")  # Regression Model
fraud_model = joblib.load("classification_fraud.pkl")  # Classification Model

# # ---- Define Feature Lists ----
# feature_names = [
#     "Type", "Days for shipping (real)", "Days for shipment (scheduled)", "Benefit per order",
#     "Sales per customer", "Category Id", "Customer Id", "Customer Zipcode",
#     "Department Id", "Order Customer Id", "Order Id", "Order Item Cardprod Id",
#     "Order Item Discount", "Order Item Discount Rate", "Order Item Id",
#     "Order Item Product Price", "Order Item Profit Ratio", "Order Item Quantity",
#     "Sales", "Order Item Total", "Order Profit Per Order", "Product Card Id",
#     "Product Category Id", "Product Price", "order_year", "order_month",
#     "order_week_day", "order_hour", "TotalPrice","fraud","late_delivery"
# ]

# categorical_features = [
#     "Category Name", "Customer City", "Customer Country", "Customer Segment",
#     "Customer State", "Department Name", "Market", "Order City", "Order Country",
#     "Order Region", "Order State", "Product Name", "Shipping Mode"
# ]


# Streamlit App Title
st.title("📊 Supply Chain Management Dashboard")

# Sidebar for Navigation
st.sidebar.title("Navigation")
options = st.sidebar.radio("Select an Analysis", ["Overview", "Data Correlation", "Sales Analysis", "Product Performance", "Fraud Detection", "Sales Trend by Weekday, Hour, and Month", "Late Delivery Analysis", "Customer Segmentation", "Model Performance Comparison", "Predictive Analysis"])

# Overview Section
if options == "Overview":
    st.subheader("Dataset Overview")
    st.write("This dashboard provides insights into supply chain data, including sales trends, fraud detection, and late deliveries.")
    st.write("### Sample Data")
    st.write(df.head())
    
    st.write("### Dataset Information")
    st.write(df.describe())

# Data Correlation Heatmap
elif options == "Data Correlation":
    st.subheader("📊 Data Correlation Heatmap")

    # Select only numerical columns
    numeric_df = df.select_dtypes(include=['number'])
    corr_matrix = numeric_df.corr()

    # Plot heatmap
    plt.figure(figsize=(12, 6))
    sns.heatmap(corr_matrix, annot=True, cmap="coolwarm", fmt=".2f", linewidths=0.5)
    plt.title("Feature Correlation Heatmap")
    st.pyplot(plt)
    
   
# Sales Analysis
elif options == "Sales Analysis":
    st.subheader("📈 Sales Trends by Market & Region")
    
    # Market-wise Sales
    market_sales = df.groupby('Market')['Sales per customer'].sum().sort_values(ascending=False)
    plt.figure(figsize=(10,5))
    sns.barplot(x=market_sales.index, y=market_sales.values, palette="coolwarm")
    plt.xticks(rotation=45)
    plt.title("Total Sales per Customer by Market")
    st.pyplot(plt)
    
    # Region-wise Sales
    region_sales = df.groupby('Order Region')['Sales per customer'].sum().sort_values(ascending=False)
    plt.figure(figsize=(10,5))
    sns.barplot(x=region_sales.index, y=region_sales.values, palette="Spectral")
    plt.xticks(rotation=45)
    plt.title("Total Sales per Customer by Region")
    st.pyplot(plt)

# Product Performance
elif options == "Product Performance":
    st.subheader("📦 Best & Worst Performing Products")
    
    # Top 10 Products by Quantity Sold
    top10_product_by_quantity = df.groupby('Product Name')['Order Item Quantity'].sum().reset_index()
    top10_product_by_quantity = top10_product_by_quantity.sort_values(by='Order Item Quantity', ascending=False).head(10)
    
    # Plotting top 10 products by quantity sold
    plt.figure(figsize=(10, 6))
    sns.barplot(x='Order Item Quantity', y='Product Name', data=top10_product_by_quantity, palette='viridis')
    plt.title('Top 10 Products by Quantity Sold')
    plt.xlabel('Quantity Sold')
    plt.ylabel('Product Name')
    st.pyplot(plt)
    
    # Grouping by Category Name for sales and average price performance
    cat = df.groupby('Category Name')
    
    # Total sales for each category
    plt.figure(figsize=(12, 6))
    cat['Sales'].sum().sort_values(ascending=False).plot.bar(color=sns.color_palette('viridis', n_colors=len(cat)))
    plt.xlabel('Category')
    plt.ylabel('Total Sales')
    plt.title("Total Sales by Category")
    st.pyplot(plt)
    
    # Average sales for each category
    plt.figure(figsize=(12, 6))
    cat['Sales'].mean().sort_values(ascending=False).plot.bar(color=sns.color_palette('plasma', n_colors=len(cat)))
    plt.xlabel('Category')
    plt.ylabel('Average Sales')
    plt.title("Average Sales by Category")
    st.pyplot(plt)
    
    # Average product price for each category
    plt.figure(figsize=(12, 6))
    cat['Product Price'].mean().sort_values(ascending=False).plot.bar(color=sns.color_palette('cividis', n_colors=len(cat)))
    plt.xlabel('Category')
    plt.ylabel('Average Price')
    plt.title("Average Product Price by Category")
    st.pyplot(plt)


# Sales Trend by Weekday, Hour, and Month
elif options == "Sales Trend by Weekday, Hour, and Month":
    st.subheader("📅 Sales Trend by Weekday, Hour, and Month")

    
    # Sales by Weekday
    df['order date (DateOrders)'] = pd.to_datetime(df['order date (DateOrders)'])
    df['Weekday'] = df['order date (DateOrders)'].dt.weekday
    weekday_sales = df.groupby('Weekday')['Sales'].mean()  # Calculate mean sales
    
    plt.figure(figsize=(10,5))
    sns.lineplot(x=weekday_sales.index, y=weekday_sales.values, marker="o", palette="coolwarm")
    plt.title("Sales Trend by Weekday")
    plt.xlabel("Weekday")
    plt.ylabel("Average Sales")  # Average sales
    st.pyplot(plt)
    
    # Sales by Hour
    df['Hour'] = df['order date (DateOrders)'].dt.hour
    hour_sales = df.groupby('Hour')['Sales'].mean()  # Calculate mean sales
    
    plt.figure(figsize=(10,5))
    sns.lineplot(x=hour_sales.index, y=hour_sales.values, marker="o", palette="Spectral")
    plt.title("Sales Trend by Hour")
    plt.xlabel("Hour of the Day")
    plt.ylabel("Average Sales")  # Average sales
    st.pyplot(plt)
    
    # Sales by Month
    df['Month'] = df['order date (DateOrders)'].dt.month
    month_sales = df.groupby('Month')['Sales'].mean()  # Calculate mean sales
    
    plt.figure(figsize=(10,5))
    sns.lineplot(x=month_sales.index, y=month_sales.values, marker="o", palette="viridis")
    plt.title("Sales Trend by Month")
    plt.xlabel("Month")
    plt.ylabel("Average Sales")  # Average sales
    st.pyplot(plt)

# Late Delivery Analysis
elif options == "Late Delivery Analysis":
    st.subheader("⏳ Late Delivery Analysis")

    #  LATE DELIVERY BY REGION 
    st.subheader("📍 Late Deliveries by Region")

    # Count late deliveries by region
    late_deliveries = df[df['Delivery Status'] == 'Late delivery']['Order Region'].value_counts()

    # Plot bar chart
    plt.figure(figsize=(10,5))
    sns.barplot(x=late_deliveries.index, y=late_deliveries.values, palette="viridis")
    plt.xticks(rotation=45)
    plt.title("Late Deliveries by Region")
    st.pyplot(plt)

    #  LATE DELIVERY ANALYSIS BY PRODUCT CATEGORY 
    st.subheader("🚚 Late Deliveries by Product Category")

    # Filter late deliveries
    late_delivery = df[df['Delivery Status'] == 'Late delivery']

    # Get the top 10 product categories with most late deliveries
    top_late_deliveries = late_delivery['Category Name'].value_counts().nlargest(10)

    # Create a bar plot for product categories
    plt.figure(figsize=(12,6))
    top_late_deliveries.plot(kind='bar', color='skyblue')
    plt.title("Top 10 Product Categories with Most Late Deliveries")
    plt.xlabel("Product Category")
    plt.ylabel("Number of Late Deliveries")
    plt.xticks(rotation=45)
    st.pyplot(plt)

    #  LATE DELIVERY ANALYSIS BY SHIPPING METHOD 
    st.subheader("📦 Late Deliveries by Shipping Method")

    # Filtering late delivery orders for different shipping methods
    xyz1 = df[(df['Delivery Status'] == 'Late delivery') & (df['Shipping Mode'] == 'Standard Class')]
    xyz2 = df[(df['Delivery Status'] == 'Late delivery') & (df['Shipping Mode'] == 'First Class')]
    xyz3 = df[(df['Delivery Status'] == 'Late delivery') & (df['Shipping Mode'] == 'Second Class')]
    xyz4 = df[(df['Delivery Status'] == 'Late delivery') & (df['Shipping Mode'] == 'Same Day')]

    # Counting total values by order region for each shipping method
    count1 = xyz1['Order Region'].value_counts()
    count2 = xyz2['Order Region'].value_counts()
    count3 = xyz3['Order Region'].value_counts()
    count4 = xyz4['Order Region'].value_counts()

    # Index names (Regions)
    names = df['Order Region'].value_counts().keys()
    n_groups = len(names)

    # Creating the bar plot for shipping methods
    fig, ax = plt.subplots(figsize=(12,6))
    index = np.arange(n_groups)
    bar_width = 0.2
    opacity = 0.6

    type1 = plt.bar(index, count1, bar_width, alpha=opacity, color='b', label='Standard Class')
    type2 = plt.bar(index + bar_width, count2, bar_width, alpha=opacity, color='r', label='First Class')
    type3 = plt.bar(index + bar_width * 2, count3, bar_width, alpha=opacity, color='g', label='Second Class')
    type4 = plt.bar(index + bar_width * 3, count4, bar_width, alpha=opacity, color='y', label='Same Day')

    plt.xlabel('Order Regions')
    plt.ylabel('Number of Late Deliveries')
    plt.title('Late Deliveries by Shipping Method in Different Regions')
    plt.legend()
    plt.xticks(index + bar_width * 1.5, names, rotation=90)  
    plt.tight_layout()
    st.pyplot(fig)

# Fraud Detection Analysis
elif options == "Fraud Detection":
    st.subheader("🚨 Fraud Analysis by Region")
    
    fraud_df = df[df['Order Status'] == 'SUSPECTED_FRAUD']
    fraud_counts = fraud_df['Order Region'].value_counts()
    
    plt.figure(figsize=(10,5))
    sns.barplot(x=fraud_counts.index, y=fraud_counts.values, palette="Reds")
    plt.xticks(rotation=45)
    plt.title("Fraud Cases by Region")
    st.pyplot(plt)

    st.subheader("🚨 Fraud Analysis: Most Suspected Fraud Categories")

    # Filter fraud cases
    high_fraud1 = df[df['Order Status'] == 'SUSPECTED_FRAUD']
    high_fraud2 = df[(df['Order Status'] == 'SUSPECTED_FRAUD') & (df['Order Region'] == 'Western Europe')]

    # Plotting bar chart for top 10 fraud categories (All Regions)
    fraud1 = high_fraud1['Category Name'].value_counts().nlargest(10)
    fraud2 = high_fraud2['Category Name'].value_counts().nlargest(10)

    # Creating a figure with two subplots
    fig, ax = plt.subplots(1, 2, figsize=(20, 8))

    # Fraud categories across all regions
    fraud1.plot(kind='bar', color='orange', ax=ax[0])
    ax[0].set_title("Top 10 Products with Highest Fraud (All Regions)", fontsize=15)
    ax[0].set_xlabel("Products", fontsize=13)
    ax[0].set_ylabel("Fraud Cases", fontsize=13)
    ax[0].tick_params(axis='x', rotation=45)

    # Fraud categories in Western Europe
    fraud2.plot(kind='bar', color='green', ax=ax[1])
    ax[1].set_title("Top 10 Fraudulent Products (Western Europe)", fontsize=15)
    ax[1].set_xlabel("Products", fontsize=13)
    ax[1].set_ylabel("Fraud Cases", fontsize=13)
    ax[1].tick_params(axis='x', rotation=45)

    plt.tight_layout()
    st.pyplot(fig)

    # Insight
    st.write("🔹 **Insight:** Fraud cases are highest in certain product categories, especially in Western Europe. Businesses should implement stricter fraud detection mechanisms in these regions.")

    
    #  TOP 10 CUSTOMERS WITH MOST FRAUD 
    st.subheader("👤 Top 10 Fraudulent Customers")

    # Get top 10 customers involved in fraud
    top_fraud_customers = high_fraud1['Customer Full Name'].value_counts().nlargest(10)

    # Plot bar chart
    fig, ax = plt.subplots(figsize=(20, 8))
    top_fraud_customers.plot(kind='bar', color='red', ax=ax)
    ax.set_title("Top 10 Customers with Most Fraud Cases", fontsize=15)
    ax.set_xlabel("Customer Name", fontsize=13)
    ax.set_ylabel("Fraud Cases", fontsize=13)
    ax.tick_params(axis='x', rotation=45)
    st.pyplot(fig)

    # Insight
    st.write("🔹 **Insight:** Some customers have been involved in multiple fraudulent transactions. Businesses should flag these high-risk customers for further investigation.")

    # ---- MARY SMITH FRAUD ANALYSIS ----
    st.subheader("🕵️ Fraud Orders by Mary Smith")

    # Filter orders for Mary Smith with suspected fraud
    amount = df[(df['Customer Full Name'] == 'MarySmith')&(df['Order Status'] == 'SUSPECTED_FRAUD')]

    # Total fraud sales amount for Mary Smith
    total_fraud_sales = amount['Sales'].sum()
    
    # Display fraudulent sales amount
    st.write(f"🔹 **Mary Smith has been involved in fraudulent transactions totaling ${total_fraud_sales:,.2f}.**")



# Customer Segmentation
elif options == "Customer Segmentation":
    
    st.subheader("🛍️ Customer Segmentation  RFM Analysis")

    # Ensure 'TotalPrice' column is created
    df['TotalPrice'] = df['Order Item Quantity'] * df['Order Item Total']

    # Convert order date to datetime format
    df['order date (DateOrders)'] = pd.to_datetime(df['order date (DateOrders)'], errors='coerce')

    # Define present date (for Recency calculation)
    import datetime as dt
    present = dt.datetime(2018,2,1)

    # RFM Calculation
    Customer_seg = df.groupby('Order Customer Id').agg({
        'order date (DateOrders)': lambda x: (present - x.max()).days,  # Recency
        'Order Id': 'count',  # Frequency
        'TotalPrice': 'sum'  # Monetary Value
    })

    # Convert Recency to int
    Customer_seg['order date (DateOrders)'] = Customer_seg['order date (DateOrders)'].astype(int)

    # Rename columns
    Customer_seg.rename(columns={'order date (DateOrders)': 'R_Value', 
                                 'Order Id': 'F_Value', 
                                 'TotalPrice': 'M_Value'}, inplace=True)

    # Divide RFM into quartiles
    quantiles = Customer_seg.quantile(q=[0.25, 0.5, 0.75]).to_dict()

    # Function to calculate RFM Scores
    def R_Score(a, b, c):
        if a <= c[b][0.25]: return 1
        elif a <= c[b][0.50]: return 2
        elif a <= c[b][0.75]: return 3
        else: return 4

    def FM_Score(x, y, z):
        if x <= z[y][0.25]: return 4
        elif x <= z[y][0.50]: return 3
        elif x <= z[y][0.75]: return 2
        else: return 1

    # Assign R, F, M Scores
    Customer_seg['R_Score'] = Customer_seg['R_Value'].apply(R_Score, args=('R_Value', quantiles))
    Customer_seg['F_Score'] = Customer_seg['F_Value'].apply(FM_Score, args=('F_Value', quantiles))
    Customer_seg['M_Score'] = Customer_seg['M_Value'].apply(FM_Score, args=('M_Value', quantiles))

    # Calculate RFM Score
    Customer_seg['RFM_Score'] = (Customer_seg['R_Score'].astype(str) +
                                 Customer_seg['F_Score'].astype(str) +
                                 Customer_seg['M_Score'].astype(str))
    
    # Calculate total RFM score
    Customer_seg['RFM_Total_Score'] = Customer_seg[['R_Score', 'F_Score', 'M_Score']].sum(axis=1)

    # Define Customer Segmentation Categories
    def RFM_Total_Score(df):
        if df['RFM_Total_Score'] >= 11: return 'Champions'
        elif df['RFM_Total_Score'] == 10: return 'Loyal Customers'
        elif df['RFM_Total_Score'] == 9: return 'Recent Customers'
        elif df['RFM_Total_Score'] == 8: return 'Promising'
        elif df['RFM_Total_Score'] == 7: return 'Customers Needing Attention'
        elif df['RFM_Total_Score'] == 6: return 'Can’t Lose Them'
        elif df['RFM_Total_Score'] == 5: return 'At Risk'
        else: return 'Lost'

    # Apply Customer Segmentation
    Customer_seg['Customer_Segmentation'] = Customer_seg.apply(RFM_Total_Score, axis=1)

    # Pie Chart for Customer Segmentation

    fig, ax = plt.subplots(figsize=(6,6))
    Customer_seg['Customer_Segmentation'].value_counts().plot.pie(
        autopct='%1.1f%%', startangle=135, explode=(0,0,0,0.1,0,0,0,0),
        shadow=False, ax=ax, cmap="tab10")
    
    plt.title("Customer Segmentation", size=15)
    plt.ylabel("")
    st.pyplot(fig)

    # Display RFM Table
    st.subheader("📋 RFM Analysis Data")
    st.dataframe(Customer_seg[['R_Value', 'F_Value', 'M_Value', 'RFM_Score', 'Customer_Segmentation']])

    # Insights
    st.write("🔹 **Insight:** RFM analysis helps classify customers based on their purchase behavior. Businesses should focus on retaining 'Champions' and 'Loyal Customers' while re-engaging 'At Risk' and 'Lost' customers.")





# Model Performance Comparison
elif options == "Model Performance Comparison":
    st.subheader("📊 Model Performance: Regression vs Classification")
    
    

    # Regression Model Performance Data
    regression_data = {
        "Regression Model": ["Lasso", "Ridge", "Random Forest", "eXtreme gradient boosting", "Decision tree", "Linear Regression"],
        "MAE Value for Sales": [1.55, 0.75, 0.87, 0.15, 0.013, 0.0005],
        "RMSE Value for Sales": [2.33, 0.97, 3.679, 0.51, 0.87, 0.0014],
        "MAE Value for Quantity": [0.9, 0.34, 0.004, 0.0008, 3.69, 0.34],
        "RMSE Value for Quantity": [1.03, 0.52, 0.04, 0.008, 0.006, 0.52]
    }

    # Classification Model Performance Data
    classification_data = {
        "Classification Model": [
            "Logistic", "Gaussian Naive Bayes", "Support Vector Machines", 
            "K Nearest Neighbour", "Random Forest", "eXtreme Gradient Boosting", "Decision Tree"
        ],
        "Accuracy Score for Fraud Detection": [97.8, 87.84, 97.75, 97.36, 98.66, 99.03, 99.08],
        "Recall Score for Fraud Detection": [59.4, 16.23, 56.89, 41.9, 98.93, 92.39, 81.43],
        "F1 Score for Fraud Detection": [31.22, 27.92, 28.42, 35.67, 60.79, 75.86, 80.16],
        "Accuracy Score for Late Delivery": [98.84, 57.27, 98.84, 80.82, 99.6, 99.13, 99.37],
        "Recall Score for Late Delivery": [97.94, 56.2, 97.94, 83.45, 98.52, 98.45, 99.41],
        "F1 Score for Late Delivery": [98.96, 71.95, 98.96, 82.26, 99.74, 99.21, 99.42]
    }

    # Convert to DataFrame
    regression_df = pd.DataFrame(regression_data)
    classification_df = pd.DataFrame(classification_data)

    st.subheader("Regression Model Performance")
    st.dataframe(regression_df)

    st.subheader("Classification Model Performance")
    st.dataframe(classification_df)

#   PREDICTIVE ANALYSIS 
elif options == "Predictive Analysis":
    st.subheader("🔍 Predict Future Sales")

    #  SALES PREDICTION (REGRESSION) 
    st.markdown("### 📈 Sales Prediction")

    
    feature_names = [
    "Days for shipping (real)", "Days for shipment (scheduled)", "Benefit per order",
    "Sales per customer", "Category Id", "Customer Id", "Customer Zipcode",
    "Department Id", "Order Customer Id", "Order Id", "Order Item Cardprod Id",
    "Order Item Discount", "Order Item Discount Rate", "Order Item Id",
    "Order Item Product Price", "Order Item Profit Ratio", 
     "Order Item Total", "Order Profit Per Order", "Product Card Id",
    "Product Category Id", "Product Price", "order_year", "order_month",
    "order_week_day", "order_hour",  "Category Name", "Customer City", "Customer Country", "Customer Segment","Customer Full Name",
    "Customer State", "Department Name", "Market", "Order City", "Order Country",
    "Order Region", "Order State", "Product Name", "Shipping Mode","fraud","late_delivery"
]


    # Pre-filled Sample Inputs
    pre_filled_values = {
        "Type": "DEBIT",
        "Days for shipping (real)": 3,
        "Days for shipment (scheduled)": 4,
        "Benefit per order": 91.25,
        "Sales per customer": 314.64,
        "Category Id": 73,
        "Category Name": "Sporting Goods",
        "Customer City": "Caguas",
        "Customer Country": "Puerto Rico",
        "Customer Segment": "Consumer",
        "Customer State": "PR",
        "Customer Zipcode": 725,
        "Department Id": 2,
        "Department Name": "Fitness",
        "Market": "Pacific Asia",
        "Order City": "Bekasi",
        "Order Country": "Indonesia",
        "Order Region": "Southeast Asia",
        "Order State": "Java Occidental",
        "Product Name": "Smart watch",
        "Product Price": 327.75,
        "Order Item Discount": 13.11,
        "Order Item Discount Rate": 0.04,
        "Order Item Product Price": 327.75,
        "Order Item Profit Ratio": 0.29,
        "Order Item Quantity": 1,
        "Order Item Total": 327.75,
        "Order Profit Per Order": 91.25,
        "Shipping Mode": "Standard Class",
        "order_year": 2018,
        "order_month": 1,
        "order_week_day": 2,
        "order_hour": 22,
        "Customer Full Name":"MarySmith",
        "fraud": 0,
        "late_delivery": 0,
        "Customer Id": 20755,
        "Order Id": 77202,
        "Order Customer Id": 20755,
        "Order Item Cardprod Id": 1360,
        "Order Item Id": 180517,
        "Product Card Id": 1360,
        "Product Category Id": 73
    }



    
    # Collect User Inputs for Categorical Features
    user_inputs = {}
    
    # Loop Through Features
    for feature, value in pre_filled_values.items():
        if isinstance(value, str):
            user_inputs[feature] = st.selectbox(f"Select {feature}", [value], index=0)
        elif isinstance(value, float):
            user_inputs[feature] = st.number_input(f"Enter {feature}", value=value, format="%.2f")
        else:
            user_inputs[feature] = st.number_input(f"Enter {feature}", value=value)

    user_inputs["Sales"]=0

    # Convert to DataFrame
    sales_input = pd.DataFrame([user_inputs])

    # Encode Categorical Features
    categorical_features = [
        "Type", "Category Name", "Customer City", "Customer Country", "Customer Segment", "Customer State",
        "Department Name", "Market", "Order City", "Order Country", "Order Region", "Order State","Customer Full Name",
        "Product Name", "Shipping Mode"
    ]


    from sklearn.preprocessing import LabelEncoder
    for feature in categorical_features:
        if feature in sales_input.columns:
            encoder = LabelEncoder()
            sales_input[feature] = encoder.fit_transform(sales_input[feature])  # Encode dynamically

    # Load the saved MinMaxScaler
    scaler = joblib.load("scaler.pkl")

    # Apply scaling using the loaded scaler
    numerical_features = [col for col in sales_input.columns if col not in categorical_features]
    sales_input[numerical_features] = scaler.fit_transform(sales_input[numerical_features])  
    # Predict with Scaling
    if st.button("Predict Sales With MinMax Scaling"):
        prediction_with_scaling = sales_model.predict(sales_input)[0]
        st.success(f"✅ With MinMax Scaling Predicted Sales: ${prediction_with_scaling:,.2f}")

    
st.sidebar.write("📌 Select different options from the sidebar to explore more insights.")
