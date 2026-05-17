import csv
from pathlib import Path


INPUT_CSV = "train.csv"
OUTPUT_CSV = "train_sin_extension.csv"
COLUMN_NAME = "Id"


def remove_extension(filename: str) -> str:
    return Path(filename).stem


with open(INPUT_CSV, "r", newline="", encoding="utf-8") as infile:
    reader = csv.DictReader(infile)
    fieldnames = reader.fieldnames

    if not fieldnames or COLUMN_NAME not in fieldnames:
        raise ValueError(f"No se encontró la columna '{COLUMN_NAME}' en {INPUT_CSV}.")

    with open(OUTPUT_CSV, "w", newline="", encoding="utf-8") as outfile:
        writer = csv.DictWriter(outfile, fieldnames=fieldnames)
        writer.writeheader()

        for row in reader:
            row[COLUMN_NAME] = remove_extension(row[COLUMN_NAME])
            writer.writerow(row)

print(f"Archivo generado: {OUTPUT_CSV}")
