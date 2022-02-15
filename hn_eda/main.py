from pathlib import Path
import pandas as pd

from hn_eda.corpus_metrics import CorpusMetrics
from hn_eda.data_preparation import load_topstories_from_zip
from hn_eda.story_corpus import StoryCorpusReader
from wordcloud import WordCloud
from matplotlib import pyplot as plt

ROOT = Path(__file__).parent
GENERATED_DIR = ROOT.parent / "generated"


def main():
    topstories = load_topstories_from_zip()
    topstories.describe()

    story_corpus = StoryCorpusReader()
    corpus_metric = CorpusMetrics(corpus=story_corpus)

    metric_values = {name: value for name, value in corpus_metric.values()}

    readme_df = pd.DataFrame(
        data=[metric_values.values()],
        columns=metric_values.keys(),
        index=["title"],
    )
    readme_df.transpose().to_markdown(GENERATED_DIR / "readme.md")

    corpus_metric.frequency_distribution.most_common(20)

    corpus_metric.story_text.collocations()

    # When window_size > 2, count non-contiguous bigrams, in the
    # style of Church and Hanks's (1990) association ratio.
    corpus_metric.story_text.collocations(window_size=3)

    trigramer(story_corpus)

    corpus_metric.story_text.concordance("pfizer")

    plt.figure(figsize=(18, 12))
    corpus_metric.story_text.dispersion_plot(
        [
            "Google",
            "Microsoft",
            "Apple",
            "Amazon",
            "Tesla",
        ]
    )
    plt.savefig("saved_figure.png")

    plt.figure(figsize=(18, 12))
    corpus_metric.story_text.dispersion_plot(
        [
            "Rust",
            "Python",
            "JavaScript",
            "C",
        ]
    )
    plt.savefig("saved_figure.png")

    plt.figure(figsize=(18, 12))
    corpus_metric.frequency_distribution.plot(20, cumulative=True)
    plt.savefig("saved_figure.png")

    # Access the plaintext; outputs pure string/basestring.
    story_corpus.items()

    plot_word_cloud(corpus_metric, plot_path=GENERATED_DIR / "wordcloud.png")


def summary_files(dir_path: Path, corpus_metrics: CorpusMetrics):
    dir_path.mkdir(exist_ok=True)

    with open(dir_path / "in_vocab.txt", "w") as in_vocab_file:
        in_vocab_file.writelines(
            "{}\n".format(x) for x in sorted(corpus_metrics.in_vocab_tokens)
        )

    with open(dir_path / "out_of_vocab.txt", "w") as out_of_vocab_file:
        out_of_vocab_file.writelines(
            "{}\n".format(x) for x in sorted(corpus_metrics.out_of_vocab_tokens)
        )


def plot_word_cloud(corpus_metric: CorpusMetrics, plot_path: Path):
    # generating the wordcloud
    wordcloud = (
        WordCloud(
            background_color="black",
            max_words=150,
            include_numbers=False,
        )
        .generate_from_frequencies(corpus_metric.frequency_distribution)
        .recolor(random_state=1)
    )

    # plot the wordcloud
    plt.figure(figsize=(20, 20))
    plt.imshow(wordcloud)

    # to remove the axis value
    plt.axis("off")
    plt.savefig(plot_path)


def trigramer(story_corpus: StoryCorpusReader):
    """Finds trigram collocations."""
    from nltk.metrics import (
        TrigramAssocMeasures,
        ranks_from_scores,
        spearman_correlation,
    )
    from nltk.collocations import TrigramCollocationFinder
    from nltk.corpus import stopwords

    likelihood_ratio_scorer = TrigramAssocMeasures.likelihood_ratio
    raw_freq_scorer = TrigramAssocMeasures.raw_freq
    point_wise_mutual_information = TrigramAssocMeasures.pmi

    ignored_words = stopwords.words("english")
    word_filter = lambda w: len(w) < 3 or w.lower() in ignored_words

    words = [word.lower() for word in story_corpus.words()]

    cf = TrigramCollocationFinder.from_words(words)
    cf.apply_freq_filter(min_freq=2)
    cf.apply_word_filter(word_filter)

    corr_1 = spearman_correlation(
        ranks_from_scores(cf.score_ngrams(likelihood_ratio_scorer)),
        ranks_from_scores(cf.score_ngrams(raw_freq_scorer)),
    )

    print([" ".join(tup) for tup in cf.nbest(likelihood_ratio_scorer, 15)])
    print(f"Correlation to {raw_freq_scorer.__name__}: {corr_1:0.4f}")

    corr_2 = spearman_correlation(
        ranks_from_scores(cf.score_ngrams(point_wise_mutual_information)),
        ranks_from_scores(cf.score_ngrams(raw_freq_scorer)),
    )

    print([" ".join(tup) for tup in cf.nbest(point_wise_mutual_information, 15)])
    print(f"Correlation to {raw_freq_scorer.__name__}: {corr_2:0.4f}")
