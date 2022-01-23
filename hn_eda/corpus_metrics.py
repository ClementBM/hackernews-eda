import re
import string
from pathlib import Path
from statistics import mean, median, stdev

import fasttext
import numpy as np
from nltk import FreqDist
from nltk.book import Text
from nltk.corpus import stopwords, words
from nltk.stem import WordNetLemmatizer

from nltk.corpus.reader.api import CorpusReader
from hn_eda.story_corpus import StoryCorpusReader

ROOT = Path(__file__).parent


class NlpMetricI:
    def __init__(self):
        pass

    def value(self):
        pass

    def formula(self):
        pass


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

    def _compute_dictionary(self):
        """
        frequency_distribution.most_common(n=None)
        """
        stop_words = set(stopwords.words("english"))

        meaningful_words = [
            word
            for word in self.story_text
            if word.casefold() not in stop_words
            and re.match(f"[{re.escape(string.punctuation)}—]", word.casefold()) == None
        ]

        self.frequency_distribution = FreqDist(meaningful_words)

    def _lemmatize_dictionary(self):
        """
        "n" for nouns
        "v" for verbs
        "a" for adjectives
        "r" for adverbs
        "s" for satellite adjectives
        """
        lemmatizer = WordNetLemmatizer()
        self.lemmatized_words = set()
        for word in self.frequency_distribution.keys():
            lemme = lemmatizer.lemmatize(word, pos="n")
            if lemme.casefold() != word.casefold():
                self.lemmatized_words.add(lemme)
                continue

            v_lemme = lemmatizer.lemmatize(word, pos="v")
            if v_lemme.casefold() != word.casefold():
                self.lemmatized_words.add(v_lemme)
                continue

            casefold_lemme = lemmatizer.lemmatize(word.casefold(), pos="n")
            if casefold_lemme.casefold() != word.casefold():
                self.lemmatized_words.add(casefold_lemme)
                continue

            casefoldv_lemme = lemmatizer.lemmatize(word.casefold(), pos="v")
            self.lemmatized_words.add(casefoldv_lemme)

    def _compute_oov(self):
        self.in_vocab_tokens = set()
        self.out_of_vocab_tokens = set()
        nltk_words = set(word.lower() for word in words.words())

        for token in self.lemmatized_words:
            if token.lower() in nltk_words:
                self.in_vocab_tokens.add(token)
            else:
                self.out_of_vocab_tokens.add(token)

    def numerical_proportion(self):
        numerica_prop = 0
        numerical_regex_pattern = r"^(([0-9]*)|(([0-9]*)[\.,]([0-9]*)))$"
        self.numerical_tokens = list()
        for token in self.lemmatized_words:
            if re.match(numerical_regex_pattern, token) is not None:
                self.numerical_tokens.append(token)
                numerica_prop += self.frequency_distribution.freq(token)
        return numerica_prop

    def in_vocabulary_proportion(self):
        return len(self.in_vocab_tokens) / len(self.lemmatized_words)

    def out_of_vocabulary_proportion(self):
        return len(self.out_of_vocab_tokens) / len(self.lemmatized_words)

    def lexical_diversity(self):
        return self.dictionary_length() / self.frequency_distribution.N()

    def hapaxes_proportion(self):
        """
        Return the proportion of all samples that occur once (hapax legomena)
        """
        return len(self.frequency_distribution.hapaxes()) / self.dictionary_length()

    def dictionary_length(self):
        return len(self.frequency_distribution)

    def lem_dictionary_length(self):
        return len(self.lemmatized_words)

    def uppercase_item_proportion(self):
        return len(self.uppercase_items) / self.unique_item_count()

    def english_item_proportion(self):
        unique_sentences = [
            " ".join(sentence) for sentence in self.unique_sentences_tokens
        ]

        are_english_items = self.is_english(unique_sentences)
        self.non_english_sentences = np.array(unique_sentences)[
            np.logical_not(are_english_items)
        ]
        return are_english_items.sum() / self.unique_item_count()

    def is_english(self, texts: list):
        path_to_pretrained_model = ROOT / "data" / "lid.176.ftz"
        fmodel = fasttext.load_model(str(path_to_pretrained_model))
        return np.array(sum(fmodel.predict(texts)[0], [])) == "__label__en"

    def metric_names(self):
        return [
            "duplicate proportion",
            "count",
            "unique count",
            "dictionary length",
            "lem dictionary length",
            "in vocabulary",
            "out of vocabulary",
            "lexical diversity",
            "hapaxes",
            "uppercase items",
            "numerical proportion",
            "average length",
            "std length",
            "median length",
            "min max length",
            "english proportion",
        ]

    def values(self):
        return [
            round(self.duplicate_proportion(), 4),
            self.item_count(),
            self.unique_item_count(),
            self.dictionary_length(),
            self.lem_dictionary_length(),
            round(self.in_vocabulary_proportion(), 4),
            round(self.out_of_vocabulary_proportion(), 4),
            round(self.lexical_diversity(), 4),
            round(self.hapaxes_proportion(), 4),
            round(self.uppercase_item_proportion(), 4),
            round(self.numerical_proportion(), 4),
            round(self.average_item_length(), 2),
            round(self.std_item_length(), 2),
            round(self.median_item_length(), 2),
            self.extremum_item_length(),
            round(self.english_item_proportion(), 4),
        ]
