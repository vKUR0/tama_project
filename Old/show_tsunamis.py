import pandas as pd
import os
import tkinter as tk

def main(path):
    file_path = os.path.join(path, "tsunamis_updated_2.csv")
    # remove the second line which is metadata
    df = pd.read_csv(file_path)
    #show all the tsunamis in a tkinter window
    root = tk.Tk()
    root.title("Tsunamis")
    text = tk.Text(root)
    text.pack()
    # show the tsunamis in the text widget with their ID, Year, Mo, Dy, Location Name, Earthquake Magnitude
    for index, row in df.iterrows():
        #add a line with the name of the columns for the first line
        if index == 0:
            text.insert(tk.END, f"ID - Year - Mo - Dy - hour+7 - Location Name - Earthquake Magnitude\n")
        #add a line for each tsunami with the format "ID - Year - Mo - Dy - hour+7 - Location Name - Earthquake Magnitude"
        text.insert(tk.END, f"{row['ID']} - {row['Year']} - {row['Mo']} - {row['Dy']} - {row['Hr']+9} - {row['Location Name']} - {row['Earthquake Magnitude']}\n")
    root.mainloop()

if __name__ == "__main__":
    path = "dataset1/"
    main(path)