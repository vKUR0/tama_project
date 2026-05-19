import os
import pandas as pd


def main():
    dataset_dir = "./dataset_tmp"
    out_dir = "./dataset"
    os.makedirs(out_dir, exist_ok=True)
    dfs = []
    target_variable_distribution = {}

    for filename in sorted(os.listdir(dataset_dir)):
        if not filename.endswith(".csv"):
            continue
        file_path = os.path.join(dataset_dir, filename)
        # CSV files contain a metadata line before the header; skip it so pandas reads the header correctly
        try:
            df = pd.read_csv(file_path, skiprows=1)
        except Exception as e:
            print(f"Failed to read {filename}: {e}")
            continue
        if df.empty:
            print(f"Dataset: {filename} is empty after skipping metadata")
            continue
        # assume target is the last column
        target = pd.to_numeric(df.iloc[:, -1], errors='coerce').dropna().astype(int)
        if not target.empty:
            counts = target.value_counts().sort_index()
            target_variable_distribution[filename] = counts.to_dict()
        else:
            target_variable_distribution[filename] = {}

        print(f"Dataset: {filename}")
        print(f"has {len(df)} rows and {len(df.columns)} columns\n")

        dfs.append(df)

    if not dfs:
        print("No datasets were loaded; nothing to merge.")
        return

    # concatenate preserving columns; missing columns will be NaN
    merged_df = pd.concat(dfs, ignore_index=True, sort=False)
    out_path = os.path.join(out_dir, "merged_dataset.csv")
    merged_df.to_csv(out_path, index=False)
    print(f"Merged dataset saved to {out_path}")

def dataset_desc(path):
    print("Target variable distribution across datasets:")
    for filename in sorted(os.listdir(path)):
        if not filename.endswith(".csv"):
            continue
        file_path = os.path.join(path, filename)
        try:
            df = pd.read_csv(file_path, skiprows=1)
        except Exception as e:
            print(f"Failed to read {filename}: {e}")
            continue
        if df.empty:
            print(f"Dataset: {filename} is empty after skipping metadata")
            continue
        target = pd.to_numeric(df.iloc[:, -1], errors='coerce').dropna().astype(int)
        if not target.empty:
            print(f"Dataset: {filename}")
            print(f"has {len(df)} rows and {len(df.columns)} columns\n")
            print(df.head(3))
        else:
            print(f"Dataset: {filename}: No valid target variable found")

def update_dataset(path):
    #sort by timestamp and save in "path+1" with the same filename
    out_dir = path + "1"
    os.makedirs(out_dir, exist_ok=True)
    for filename in sorted(os.listdir(path)):
        if not filename.endswith(".csv"):
            continue
        file_path = os.path.join(path, filename)
        try:
            df = pd.read_csv(file_path)
        except Exception as e:
            print(f"Failed to read {filename}: {e}")
            continue
        if df.empty:
            print(f"Dataset: {filename} is empty after skipping metadata")
            continue
        # sort by timestamp (assume first column is timestamp)
        df_sorted = df.sort_values(by=df.columns[0])
        out_path = os.path.join(out_dir, filename)
        df_sorted.to_csv(out_path, index=False)
        print(f"Sorted dataset saved to {out_path}")
    


if __name__ == "__main__":
    Dataset_path = "./dataset"
    #main()
    dataset_desc(Dataset_path)
    update_dataset(Dataset_path)
    dataset_desc(Dataset_path + "1")