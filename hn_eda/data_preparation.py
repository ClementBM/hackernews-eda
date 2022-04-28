import json
from pathlib import Path

import pandas as pd
import requests
from tqdm import tqdm

ROOT = Path(__file__)

TOPSTORIES_NAME = "hn_topstories"
TOPSTORIES_ZIP = ROOT.parent / f"{TOPSTORIES_NAME}.zip"
TOPSTORIES_JSONL = TOPSTORIES_ZIP / f"{TOPSTORIES_NAME}.jsonl"


def save_topstories_as_zip():
    hn_topstories_url = (
        "https://hacker-news.firebaseio.com/v0/topstories.json?print=pretty"
    )
    hn_get_item_url = (
        "https://hacker-news.firebaseio.com/v0/item/{item_id}.json?print=pretty"
    )

    topstory_result = requests.get(hn_topstories_url)
    topstory_ids = json.loads(topstory_result.text)

    data = list()
    for topstory_id in tqdm(topstory_ids):
        result = requests.get(hn_get_item_url.format(item_id=topstory_id))
        data.append(json.loads(result.text))

    data_df = pd.json_normalize(data)
    data_df.to_pickle(TOPSTORIES_ZIP)


def save_to_json(file_path: Path):
    standard_df = pd.read_pickle(file_path)
    shuffled_df = standard_df.sample(frac=1, random_state=42).reset_index(drop=True)
    _save_df_as_jsonl(shuffled_df, file_path.parent / f"{TOPSTORIES_NAME}.jsonl")


def _save_df_as_jsonl(df, file_path: Path):
    if file_path.exists():
        file_path.unlink()

    with open(file_path, "ab") as json_file:
        df.apply(
            lambda x: json_file.write(f"{x.to_json()}\n".encode("utf-8")),
            axis=1,
        )


def load_topstories_from_zip():
    return pd.read_json(
        TOPSTORIES_ZIP,
        lines=True,
        compression="zip",
    )


if __name__ == "__main__":
    save_topstories_as_zip()
    save_to_json(TOPSTORIES_ZIP)
