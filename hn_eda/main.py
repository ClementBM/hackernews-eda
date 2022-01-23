import pandas as pd

from hn_eda.corpus_metrics import CorpusMetrics
from hn_eda.data_preparation import load_topstories_from_zip
from hn_eda.story_corpus import StoryCorpusReader


def main():
    topstories = load_topstories_from_zip()
    topstories.describe()

    story_corpus = StoryCorpusReader()
    corpus_metric = CorpusMetrics(corpus=story_corpus)

    readme_df = pd.DataFrame(
        data=[corpus_metric.values()],
        columns=corpus_metric.metric_names(),
        index=["title"],
    )
    readme_df.transpose()

    corpus_metric.frequency_distribution.most_common(20)

    corpus_metric.story_text.collocations()
    corpus_metric.story_text.collocations(window_size=3)  # does not work ?

    corpus_metric.story_text.concordance("language")

    corpus_metric.story_text.dispersion_plot(
        [
            "Google",
            "Microsoft",
            "Apple",
            "Amazon",
            "Tesla",
        ]
    )

    corpus_metric.frequency_distribution.plot(20, cumulative=True)

    # Access the plaintext; outputs pure string/basestring.
    corpus_metric.items
