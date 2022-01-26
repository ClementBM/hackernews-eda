import re
from pathlib import Path
from statistics import mean, median, stdev

from nltk import FreqDist
from nltk.book import Text
from nltk.corpus import stopwords, words
from nltk.stem import WordNetLemmatizer

from nltk.corpus.reader.api import CorpusReader

ROOT = Path(__file__).parent


class CorpusMetrics:
    def __init__(self, corpus: CorpusReader):
        self.story_corpus = corpus
        self.items = self.story_corpus.titles()

        self.story_text = Text(corpus.unique_words(self.items))
        self.sentences_tokens = corpus.sentences(self.items)
        self.unique_sentences_tokens = corpus.unique_sentences(self.items)
        self.sentence_lengths = [
            len(sentence_tokens) for sentence_tokens in self.unique_sentences_tokens
        ]
        self.uppercase_items = corpus.unique_uppercase_sentences(self.items)

        self._compute_dictionary()
        self._lemmatize_dictionary()
        self._compute_oov()

    def _compute_dictionary(self):
        stop_words = set(stopwords.words("english"))

        meaningful_words = [
            word for word in self.story_text if word.casefold() not in stop_words
        ]

        self.frequency_distribution = FreqDist(meaningful_words)

    def _lemmatize_dictionary(self):
        lemmatizer = WordNetLemmatizer()
        self.lemmatized_words = set()
        for word in self.frequency_distribution.keys():
            casefold_lemme = lemmatizer.lemmatize(word.casefold(), pos="n")
            if casefold_lemme.casefold() != word.casefold():
                self.lemmatized_words.add(casefold_lemme)
                continue

            casefoldv_lemme = lemmatizer.lemmatize(word.casefold(), pos="v")
            self.lemmatized_words.add(casefoldv_lemme)

    def _compute_oov(self):
        numerical_regex_pattern = r"^(([0-9]*)|(([0-9]*)[\.,]([0-9]*)))$"
        self.in_vocab_tokens = set()
        self.out_of_vocab_tokens = set()
        self.numerical_tokens = list()
        self.nltk_words = set(word.lower() for word in words.words())

        for token in self.lemmatized_words:
            if token.lower() in self.nltk_words:
                self.in_vocab_tokens.add(token)
            elif re.match(numerical_regex_pattern, token) is not None:
                self.numerical_tokens.append(token)
            else:
                self.out_of_vocab_tokens.add(token)

    def numerical_frequency(self):
        numerical_frequencies = [
            self.frequency_distribution.freq(token) for token in self.numerical_tokens
        ]
        return sum(numerical_frequencies)

    def duplicate_proportion(self):
        return (self.item_count() - self.unique_item_count()) / self.item_count()

    def item_count(self):
        return len(self.items)

    def unique_item_count(self):
        return len(self.unique_sentences_tokens)

    def average_item_length(self):
        return mean(self.sentence_lengths)

    def std_item_length(self):
        return stdev(self.sentence_lengths)

    def median_item_length(self):
        return median(self.sentence_lengths)

    def extremum_item_length(self):
        return min(self.sentence_lengths), max(self.sentence_lengths)

    def in_vocabulary_proportion(self):
        return len(self.in_vocab_tokens) / self.normalized_dictionary_length()

    def out_of_vocabulary_proportion(self):
        return len(self.out_of_vocab_tokens) / self.normalized_dictionary_length()

    def numerical_proportion(self):
        return len(self.numerical_tokens) / self.normalized_dictionary_length()

    def lexical_diversity(self):
        return self.dictionary_length() / self.frequency_distribution.N()

    def hapaxes_proportion(self):
        """
        Return the proportion of all samples that occur once (hapax legomena)
        """
        return len(self.frequency_distribution.hapaxes()) / self.dictionary_length()

    def dictionary_length(self):
        return len(self.frequency_distribution)

    def normalized_dictionary_length(self):
        return len(self.lemmatized_words)

    def uppercase_item_proportion(self):
        return len(self.uppercase_items) / self.unique_item_count()

    def values(self):
        return [
            ("duplicate proportion", round(self.duplicate_proportion(), 4)),
            ("count", self.item_count()),
            ("unique count", self.unique_item_count()),
            ("dictionary length", self.dictionary_length()),
            ("lem dictionary length", self.normalized_dictionary_length()),
            ("in vocabulary", round(self.in_vocabulary_proportion(), 4)),
            ("out of vocabulary", round(self.out_of_vocabulary_proportion(), 4)),
            ("numerical proportion", round(self.numerical_proportion(), 4)),
            ("numerical freqency", round(self.numerical_frequency(), 4)),
            ("lexical diversity", round(self.lexical_diversity(), 4)),
            ("hapaxes", round(self.hapaxes_proportion(), 4)),
            ("uppercase items", round(self.uppercase_item_proportion(), 4)),
            ("average length", round(self.average_item_length(), 2)),
            ("std length", round(self.std_item_length(), 2)),
            ("median length", round(self.median_item_length(), 2)),
            ("min max length", self.extremum_item_length()),
        ]
