import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
import os

def preprocess_data(input_path, output_dir):
    df = pd.read_csv(input_path)

    df = df.drop(columns=["customerID"])
    df["TotalCharges"] = df["TotalCharges"].replace(" ", np.nan)
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"])
    df = df.drop_duplicates()
    df["TotalCharges"] = df["TotalCharges"].fillna(df["TotalCharges"].median())

    le = LabelEncoder()
    for col in df.select_dtypes(include="object").columns:
        df[col] = le.fit_transform(df[col])

    X = df.drop(columns=["Churn"])
    y = df["Churn"]

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.2, random_state=42, stratify=y
    )

    os.makedirs(output_dir, exist_ok=True)
    pd.DataFrame(X_train, columns=X.columns).to_csv(f"{output_dir}/X_train.csv", index=False)
    pd.DataFrame(X_test, columns=X.columns).to_csv(f"{output_dir}/X_test.csv", index=False)
    y_train.to_csv(f"{output_dir}/y_train.csv", index=False)
    y_test.to_csv(f"{output_dir}/y_test.csv", index=False)
    print("Preprocessing selesai. Data tersimpan di", output_dir)

if __name__ == "__main__":
    preprocess_data(
        input_path="namadataset_raw/Telco-Customer-Churn.csv",
        output_dir="preprocessing/namadataset_preprocessing"
    )
