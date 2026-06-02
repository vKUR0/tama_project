import pandas as pd
import os
import tkinter as tk

def main(path):
    file_path = os.path.join(path, "tsunamis-2026-05-25_16-23-01_+0900.csv")
    # remove the second line which is metadata
    df = pd.read_csv(file_path)
    df = df.drop(index=0).reset_index(drop=True)
    #show all the tsunamis in a tkinter window
    root = tk.Tk()
    root.title("Tsunamis")
    text = tk.Text(root)
    text.pack()
    for index, row in df.iterrows():
        text.insert(tk.END, f"{row['Year']} - {row['Mo']} - {row['Dy']} - {row['Location Name']} - {row['Earthquake Magnitude']}\n")
    root.mainloop()

if __name__ == "__main__":
    path = "dataset1/"
    main(path)