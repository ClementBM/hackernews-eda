import pandas as pd

from hn_eda.corpus_metrics import CorpusMetrics
from hn_eda.data_preparation import load_topstories_from_zip
from hn_eda.story_corpus import StoryCorpusReader
from wordcloud import WordCloud
from matplotlib import pyplot as plt


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
    readme_df.transpose()

    len(corpus_metric.out_of_vocab_tokens)
    len(corpus_metric.numerical_tokens)

    corpus_metric.frequency_distribution.most_common(20)

    corpus_metric.story_text.collocations()

    corpus_metric.story_text.generate(length=10)
    corpus_metric.story_text.plot(30)

    corpus_metric.story_text[3:5]

    corpus_metric.story_text.findall("<.*><.*><Google>")

    corpus_metric.story_text.index(word="Google")

    # ?
    corpus_metric.story_text.common_contexts(words="oo")
    # ?
    corpus_metric.story_text.similar("Google")

    corpus_metric.story_text.vocab()["Google"]

    # When window_size > 2, count non-contiguous bigrams, in the
    # style of Church and Hanks's (1990) association ratio.
    corpus_metric.story_text.collocations(window_size=3)

    corpus_metric.story_text.concordance("language")

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

    plt.figure(figsize=(18, 12))
    corpus_metric.story_text.dispersion_plot(
        [
            "Rust",
            "Python",
            "JavaScript",
            "C",
        ]
    )

    plt.figure(figsize=(18, 12))
    corpus_metric.frequency_distribution.plot(20, cumulative=True)

    # Access the plaintext; outputs pure string/basestring.
    story_corpus.items()

    plot_word_cloud(corpus_metric)


def plot_word_cloud(corpus_metric: CorpusMetrics):
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
    plt.show()
