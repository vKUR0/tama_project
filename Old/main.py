import os
import pandas as pd


# def main():
#     dataset_dir = "./dataset_tmp"
#     out_dir = "./dataset"
#     os.makedirs(out_dir, exist_ok=True)
#     dfs = []
#     target_variable_distribution = {}

#     for filename in sorted(os.listdir(dataset_dir)):
#         if not filename.endswith(".csv"):
#             continue
#         file_path = os.path.join(dataset_dir, filename)
#         # CSV files contain a metadata line before the header; skip it so pandas reads the header correctly
#         try:
#             df = pd.read_csv(file_path, skiprows=1)
#         except Exception as e:
#             print(f"Failed to read {filename}: {e}")
#             continue
#         if df.empty:
#             print(f"Dataset: {filename} is empty after skipping metadata")
#             continue
#         # assume target is the last column
#         target = pd.to_numeric(df.iloc[:, -1], errors='coerce').dropna().astype(int)
#         if not target.empty:
#             counts = target.value_counts().sort_index()
#             target_variable_distribution[filename] = counts.to_dict()
#         else:
#             target_variable_distribution[filename] = {}

#         print(f"Dataset: {filename}")
#         print(f"has {len(df)} rows and {len(df.columns)} columns\n")

#         dfs.append(df)

#     if not dfs:
#         print("No datasets were loaded; nothing to merge.")
#         return

#     # concatenate preserving columns; missing columns will be NaN
#     merged_df = pd.concat(dfs, ignore_index=True, sort=False)
#     out_path = os.path.join(out_dir, "merged_dataset.csv")
#     merged_df.to_csv(out_path, index=False)
#     print(f"Merged dataset saved to {out_path}")

# def dataset_desc(path):
#     print("Target variable distribution across datasets:")
#     for filename in sorted(os.listdir(path)):
#         if not filename.endswith(".csv"):
#             continue
#         file_path = os.path.join(path, filename)
#         try:
#             df = pd.read_csv(file_path, skiprows=1)
#         except Exception as e:
#             print(f"Failed to read {filename}: {e}")
#             continue
#         if df.empty:
#             print(f"Dataset: {filename} is empty after skipping metadata")
#             continue
#         target = pd.to_numeric(df.iloc[:, -1], errors='coerce').dropna().astype(int)
#         if not target.empty:
#             print(f"Dataset: {filename}")
#             print(f"has {len(df)} rows and {len(df.columns)} columns\n")
#             print(df.head(3))
#         else:
#             print(f"Dataset: {filename}: No valid target variable found")

# def update_dataset(path):
#     #sort by timestamp and save in "path+1" with the same filename
#     out_dir = path + "1"
#     os.makedirs(out_dir, exist_ok=True)
#     for filename in sorted(os.listdir(path)):
#         if not filename.endswith(".csv"):
#             continue
#         file_path = os.path.join(path, filename)
#         try:
#             df = pd.read_csv(file_path)
#         except Exception as e:
#             print(f"Failed to read {filename}: {e}")
#             continue
#         if df.empty:
#             print(f"Dataset: {filename} is empty after skipping metadata")
#             continue
#         # sort by timestamp (assume first column is timestamp)
#         df_sorted = df.sort_values(by=df.columns[0])
#         out_path = os.path.join(out_dir, filename)
#         df_sorted.to_csv(out_path, index=False)
#         print(f"Sorted dataset saved to {out_path}")
    
# def main(path, searched_magnitude):
#     file_path = os.path.join(path, "merged_dataset.csv")
#     df = pd.read_csv(file_path)
#     print(df.columns)
#     matching_earthquakes = df[df["Magnitude"] == searched_magnitude]
#     if matching_earthquakes.empty:
#         print(f"No earthquakes found with magnitude {searched_magnitude}.")
#     else:
#         print(f"Earthquakes with magnitude {searched_magnitude}:")
#         print(matching_earthquakes)

def Modify_tsunamis_updated_4(path):
    file_path = os.path.join(path, "tsunamis_updated_4.csv")
    df = pd.read_csv(file_path)
    # remove Tsunami Event Validity and vol column
    if "Tsunami Event Validity" in df.columns:
        df = df.drop(columns=["Tsunami Event Validity"])
    if "Vol" in df.columns:
        df = df.drop(columns=["Vol"])
    if "Tsunami Cause Code" in df.columns:
        df = df.drop(columns=["Tsunami Cause Code"])
    if "More Info" in df.columns:
        df = df.drop(columns=["More Info"])
    if "Deposits" in df.columns:
        df = df.drop(columns=["Deposits"])
    if "Country" in df.columns:
        df = df.drop(columns=["Country"])
    if "Tsunami Intensity" in df.columns:
        df = df.drop(columns=["Tsunami Intensity"])
    if "Deaths" in df.columns:
        df = df.drop(columns=["Deaths"])
    if "Tsunami Magnitude (Iida)" in df.columns:
        df = df.drop(columns=["Tsunami Magnitude (Iida)"])
    if "Tsunami Magnitude (Abe)" in df.columns:
        df = df.drop(columns=["Tsunami Magnitude (Abe)"])
    if "Missing Description" in df.columns:
        df = df.drop(columns=["Missing Description"])
    if "Missing" in df.columns:
        df = df.drop(columns=["Missing"])
    if "Injuries" in df.columns:
        df = df.drop(columns=["Injuries"])
    if "Injuries Description" in df.columns:
        df = df.drop(columns=["Injuries Description"])
    if "Damage ($Mil)" in df.columns:
        df = df.drop(columns=["Damage ($Mil)"])
    if "Houses Destroyed" in df.columns:
        df = df.drop(columns=["Houses Destroyed"])
    if "Houses Destroyed Description" in df.columns:
        df = df.drop(columns=["Houses Destroyed Description"])
    if "Houses Damaged" in df.columns:
        df = df.drop(columns=["Houses Damaged"])
    if "Houses Damaged Description" in df.columns:
        df = df.drop(columns=["Houses Damaged Description"])
    if "Total Deaths" in df.columns:
        df = df.drop(columns=["Total Deaths"])
    if "Death Description" in df.columns:
        df = df.drop(columns=["Death Description"])
    if "Total Missing Description" in df.columns:
        df = df.drop(columns=["Total Missing Description"])
    if "Total Missing" in df.columns:
        df = df.drop(columns=["Total Missing"])
    if "Damage Description" in df.columns:
        df = df.drop(columns=["Damage Description"])
    if "Total Houses Destroyed Description" in df.columns:
        df = df.drop(columns=["Total Houses Destroyed Description"])
    if "Total Houses Destroyed" in df.columns:
        df = df.drop(columns=["Total Houses Destroyed"])
    if "Total Houses Damaged Description" in df.columns:
        df = df.drop(columns=["Total Houses Damaged Description"])
    if "Total Houses Damaged" in df.columns:
        df = df.drop(columns=["Total Houses Damaged"])
    
    df = df.rename(columns={"Origin Time": "Origin_Time"})
    df = df.rename(columns={"Location Name": "Location_Name"})
    df = df.rename(columns={"Maximum Water Height (m)": "Max_Height"})
    df = df.rename(columns={"Earthquake Magnitude": "Magnitude"})
    df = df.rename(columns={"Number of Runups": "Num_Runups"})
    df = df.rename(columns={"Total Damage Description": "Total_Damage_Description"})
    df = df.rename(columns={"Total Damage ($Mil)": "Total_Damage($Mil)"})
    df = df.rename(columns={"Total Injuries Description": "Total_Injuries_Description"})
    df = df.rename(columns={"Total Injuries": "Total_Injuries"})
    df = df.rename(columns={"Total Death Description": "Total_Death_Description"})

    #print all Tsunami_Magnitude_Abe$
    print(df["Total_Injuries"].unique())
    #save the updated dataset
    out_path = os.path.join(path, "tsunamis_updated_4_renamed.csv")
    df.to_csv(out_path, index=False)


# file_path = os.path.join(path, "dataset_updated_renamed.csv")
#     df = pd.read_csv(file_path)
#     print(f"Nombre de lignes dans le fichier : {len(df)}")
#     print("columns : ", df.columns)
#     # add tsunami colum to the dataset with 0 if Tsunami_Id is == 0  and 1 if Tsunami_Id is != 0
#     df["Tsunami"] = df["Tsunami_ID"].apply(lambda x: 0 if x == 0 else 1)
#     out_path = os.path.join(path, "dataset_updated_renamed_with_tsunami.csv")
#     df.to_csv(out_path, index=False)

    


if __name__ == "__main__":
    Dataset_path = "./dataset1/"

    main(Dataset_path)