import json
from pathlib import Path

import pandas as pd
import requests
from tqdm import tqdm

ROOT = Path(__file__)
TOPSTORIES_PATH = ROOT.parent / "hn_topstories.zip"


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
    data_df.to_pickle(TOPSTORIES_PATH)


def save_to_json(file_path: Path):
    standard_df = pd.read_pickle(file_path)
    shuffled_df = standard_df.sample(frac=1, random_state=42).reset_index(drop=True)
    _save_df_as_json(shuffled_df, file_path.parent / "hn_topstories.json")


def _save_df_as_json(df, file_path: Path):
    if file_path.exists():
        file_path.unlink()

    with open(file_path, "ab") as json_file:
        df.apply(
            lambda x: json_file.write(f"{x.to_json()}\n".encode("utf-8")),
            axis=1,
        )


def load_topstories_from_zip():
    return pd.read_pickle(TOPSTORIES_PATH)


if __name__ == "__main__":
    save_topstories_as_zip()
    save_to_json(TOPSTORIES_PATH)
