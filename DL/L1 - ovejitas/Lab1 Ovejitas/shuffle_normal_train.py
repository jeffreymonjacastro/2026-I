import argparse
import csv
import random
from pathlib import Path


def shuffle_csv_rows(input_path: Path, output_path: Path, seed: int | None = None) -> None:
    with input_path.open("r", newline="", encoding="utf-8-sig") as input_file:
        reader = csv.reader(input_file)
        rows = list(reader)

    if not rows:
        raise ValueError(f"El archivo '{input_path}' está vacío.")

    header, *data_rows = rows
    random_generator = random.Random(seed)
    random_generator.shuffle(data_rows)

    with output_path.open("w", newline="", encoding="utf-8") as output_file:
        writer = csv.writer(output_file)
        writer.writerow(header)
        writer.writerows(data_rows)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Lee un CSV y mezcla (shuffle) sus filas manteniendo el encabezado."
    )
    parser.add_argument(
        "--input",
        default="normal_train.csv",
        help="Ruta del CSV de entrada (por defecto: normal_train.csv).",
    )
    parser.add_argument(
        "--output",
        default="normal_train_shuffled.csv",
        help="Ruta del CSV de salida (por defecto: normal_train_shuffled.csv).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Semilla opcional para reproducibilidad.",
    )

    args = parser.parse_args()
    input_path = Path(args.input)
    output_path = Path(args.output)

    if not input_path.exists():
        raise FileNotFoundError(f"No se encontró el archivo de entrada: {input_path}")

    shuffle_csv_rows(input_path, output_path, args.seed)
    print(f"Archivo mezclado creado en: {output_path}")


if __name__ == "__main__":
    main()
