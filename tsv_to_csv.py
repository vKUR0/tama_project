import csv

with open("./dataset_tmp/tsunamis-2026-05-25_16-23-01_+0900.tsv", "r", encoding="utf-8") as tsv_file:
    reader = csv.reader(tsv_file, delimiter="\t")

    with open("tsunamis-2026-05-25_16-23-01_+0900.csv", "w", newline="", encoding="utf-8") as csv_file:
        writer = csv.writer(csv_file)

        for row in reader:
            writer.writerow(row)