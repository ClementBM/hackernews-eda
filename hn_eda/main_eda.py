from pathlib import Path

from hn_eda.story_corpus import StoryCorpusReader
from matplotlib import pyplot as plt
from nltk.book import Text
from nltk.corpus import stopwords

ROOT = Path(__file__).parent
GENERATED_DIR = ROOT.parent / "generated"


def main():
    story_corpus = StoryCorpusReader()

    story_text = Text(story_corpus.sentences_tokens())

    story_text[3:5]
    story_text[3:7]
    story_text.tokens[3:5]

    plt.figure(figsize=(18, 12))
    story_text.plot(20)

    story_text.concordance("language")

    story_text.findall("<.*><.*><Google>")

    story_text.index(word="Apple")

    story_text.collocations(window_size=3)

    plt.figure(figsize=(18, 12))
    story_text.dispersion_plot(
        [
            "Google",
            "Microsoft",
            "Apple",
            "Amazon",
            "Tesla",
        ]
    )

    plt.figure(figsize=(18, 12))
    story_text.dispersion_plot(
        [
            "Rust",
            "Python",
            "JavaScript",
            "C",
        ]
    )

    story_vocab = story_text.vocab()

    story_vocab.most_common(20)

    stop_words = set(stopwords.words("english"))

    removable_vocab_keys = []
    for vocab_key in story_vocab.keys():
        if vocab_key.casefold() in stop_words:
            removable_vocab_keys.append(vocab_key)

    for removable_vocab_key in removable_vocab_keys:
        story_vocab.pop(removable_vocab_key)

    story_vocab["Google"]

    plt.figure(figsize=(18, 12))
    story_vocab.plot(20, cumulative=False, percents=False, show=False)
    plt.xticks(rotation=45)

    story_text.collocations()
    story_text.generate(length=10)

    story_text.common_contexts(words=["Google", "Apple"])
    story_text.similar("Google")
