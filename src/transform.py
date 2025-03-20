import pandas as pd

def transform_data(df):
    """Transforms extracted data by adding new features."""
    
    # Shipping Delay (Late deliveries impact supply chain efficiency.)
    df["Shipping Delay"] = df["Days for shipping (real)"] - df["Days for shipment (scheduled)"]

    # Customer Order Frequency (Identify key customers & supply-demand patterns.)
    df["Customer Order Frequency"] = df.groupby("Customer Id")["Order Id"].transform("count")

    print(f"✅ Transformed {df.shape[0]} rows")
    return df

if __name__ == "__main__":
    from src.extract import extract_data
    
    df = extract_data()  # Extract data
    transformed_df = transform_data(df)  # Transform data

    # Print first few rows to check transformations
    print("\n📊 Sample Transformed Data:")
    print(transformed_df.head())

    # Print column names to verify transformations
    print("\n📌 Transformed Columns:")
    print(transformed_df.columns.tolist())
