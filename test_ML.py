import xgboost as xgb
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import os

def load_dataset(path):
    df = pd.read_csv(path)
    print("Dataset loaded successfully.")
    print("Dataset shape:", df.shape)
    return df



def main(path):
    file_path = os.path.join(path, "dataset_updated_renamed.csv")
    df = load_dataset(file_path)


if __name__ == "__main__":
    main("./dataset1/")