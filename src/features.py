from pathlib import Path

import typer
from loguru import logger
from tqdm import tqdm
import pandas as pd
from sklearn.preprocessing import StandardScaler, OneHotEncoder
import numpy as np

from src.config import PROCESSED_DATA_DIR

app = typer.Typer()

def normalize_numeric_features(df: pd.DataFrame, numeric_columns: list) -> pd.DataFrame:
    """
    Chuẩn hóa các biến số bằng StandardScaler
    """
    scaler = StandardScaler()
    df_copy = df.copy()
    df_copy[numeric_columns] = scaler.fit_transform(df[numeric_columns])
    return df_copy

def encode_categorical_features(df, categorical_columns):
    """
    Mã hóa one-hot cho các biến phân loại trong DataFrame.
    """
    df_copy = df.copy()
    encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
    encoded_features = encoder.fit_transform(df[categorical_columns])
    feature_names = encoder.get_feature_names_out(categorical_columns)
    encoded_df = pd.DataFrame(
        encoded_features,
        columns=feature_names,
        index=df.index
    )
    df_copy = df_copy.drop(columns=categorical_columns)
    df_copy = pd.concat([df_copy, encoded_df], axis=1)
    return df_copy

@app.command()
def main(
    input_path: Path = PROCESSED_DATA_DIR / "handled_missing_values.csv",
    output_path: Path = PROCESSED_DATA_DIR / "train_normalized.csv",
):
    logger.info(f"Đang đọc dữ liệu từ {input_path}...")
    df = pd.read_csv(input_path)
    
    numeric_columns = df.select_dtypes(include=['number']).columns.tolist()
    categorical_columns = df.select_dtypes(include=['category', 'object']).columns.tolist()

    if numeric_columns:
        logger.info("Đang chuẩn hóa các biến số...")
        df = normalize_numeric_features(df, numeric_columns)

    if categorical_columns:
        logger.info("Đang mã hóa các biến phân loại...")
        df = encode_categorical_features(df, categorical_columns)
    
    logger.info(f"Đang lưu features đã xử lý vào {output_path}...")
    df.to_csv(output_path, index=False)
    logger.success("Hoàn thành việc tạo features!")

if __name__ == "__main__":
    app()