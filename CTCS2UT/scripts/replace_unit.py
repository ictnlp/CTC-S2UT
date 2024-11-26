import argparse
import pandas as pd
from pathlib import Path
from examples.speech_to_text.data_utils import load_df_from_tsv, save_df_to_tsv
from examples.speech_to_speech.preprocessing.data_utils import load_units, process_units

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-tsv")
    parser.add_argument("--unit-txt")
    parser.add_argument("--output-tsv")
    args = parser.parse_args()
    df = load_df_from_tsv(args.input_tsv)
    data = list(df.T.to_dict().values())
    with open(args.unit_txt, "r") as f:
        unit_data = f.read().splitlines()
    for item, unit in zip(data, unit_data):
        item["tgt_text"] = unit
    df = pd.DataFrame.from_dict(data)
    save_df_to_tsv(df, args.output_tsv)

if __name__ == "__main__":
    main()