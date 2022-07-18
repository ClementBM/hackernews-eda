import re
from pathlib import Path
from statistics import mean, median, stdev
import pandas as pd

from nltk import FreqDist
from nltk.corpus import stopwords, words
from nltk.stem import WordNetLemmatizer

from hn_eda.story_corpus import CorpusReaderBase

ROOT = Path(__file__).parent


def corpus_metric(name, formula, decimal_round=0):
    def decorator(function):
        function.is_metric = True
        function.name = name
        function.formula = formula
        function.decimal_round = decimal_round
        return function

    return decorator


class CorpusMetrics:
    _unique_sentences = None
    _dictionary = None
    _uppercase_sentences_words = None
    _uppercase_sentences = None
    _sentence_lengths = None

    def __init__(self, corpus: CorpusReaderBase):
        self.corpus = corpus

        self._lemmatize_dictionary()
        self._compute_oov()

    def uppercase_sentences_words(
        self,
    ):
        if self._uppercase_sentences_words == None:
            tokenizer = self.corpus._word_tokenizer
            self._uppercase_sentences_words = [
                tuple(tokenizer.tokenize(t)) for t in self.uppercase_sentences()
            ]
        return self._uppercase_sentences_words

    def unique_sentences(self):
        """
        :return: a list of the text content of Stories as
            as a list of words.. and punctuation symbols.
        :rtype: list(list(str))
        """
        if self._unique_sentences == None:
            self._unique_sentences = set(self.corpus.titles())
        return self._unique_sentences

    def uppercase_sentences(self):
        if self._uppercase_sentences == None:
            regex_pattern = r"^[^a-z]*$"
            self._uppercase_sentences = [
                title
                for title in self.unique_sentences()
                if re.match(regex_pattern, title) is not None
            ]
        return self._uppercase_sentences

    def sentence_lengths(self):
        if self._sentence_lengths == None:
            self._sentence_lengths = [
                len(sentence_tokens) for sentence_tokens in self.unique_sentences()
            ]
        return self._sentence_lengths

    def dictionary(self):
        if self._dictionary == None:
            stop_words = set(stopwords.words("english"))

            meaningful_words = [
                word
                for word in self.corpus.words()
                if word.casefold() not in stop_words
            ]

            self._dictionary = FreqDist(meaningful_words)
        return self._dictionary

    def _lemmatize_dictionary(self):
        lemmatizer = WordNetLemmatizer()
        self.lemmatized_words = set()
        for word in self.dictionary().keys():
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

    @corpus_metric(
        name="numerical frequency",
        formula="\\vert \mathcal{T}_{numerical} \\vert \over \\vert \mathcal{T} \\vert",
        decimal_round=4,
    )
    def numerical_frequency(self):
        numerical_frequencies = [
            self.dictionary().freq(token) for token in self.numerical_tokens
        ]
        return sum(numerical_frequencies)

    @corpus_metric(
        name="numerical proportion",
        formula="\\vert \mathcal{D}_{numerical} \\vert \over \\vert \mathcal{D}_{lemme} \\vert",
        decimal_round=4,
    )
    def numerical_proportion(self):
        return len(self.numerical_tokens) / self.normalized_dictionary_length()

    @corpus_metric(
        name="duplicate proportion",
        formula="\\vert \mathcal{O} \\vert - \\vert \mathcal{O}_{unique} \\vert \over \\vert \mathcal{O} \\vert",
        decimal_round=4,
    )
    def duplicate_proportion(self):
        return (self.item_count() - self.unique_item_count()) / self.item_count()

    @corpus_metric(name="count", formula="\\vert \mathcal{O} \\vert")
    def item_count(self):
        return len(self.corpus.sentences())

    @corpus_metric(name="unique count", formula="\\vert \mathcal{O}_{unique} \\vert")
    def unique_item_count(self):
        return len(self.unique_sentences())

    @corpus_metric(name="average length", formula="\\bar{M_i}", decimal_round=2)
    def average_item_length(self):
        return mean(self.sentence_lengths())

    @corpus_metric(name="std length", formula="s_{M_i}", decimal_round=2)
    def std_item_length(self):
        return stdev(self.sentence_lengths())

    @corpus_metric(name="median length", formula="\\tilde{M_i}", decimal_round=2)
    def median_item_length(self):
        return median(self.sentence_lengths())

    @corpus_metric(name="min max length", formula="\{ min(M_i), max(M_i) \}")
    def extremum_item_length(self):
        return min(self.sentence_lengths()), max(self.sentence_lengths())

    @corpus_metric(
        name="in vocabulary",
        formula="\\vert \mathcal{D}_{lemme} \cap \mathcal{D}_{NLTK} \\vert \over \\vert \mathcal{D}_{lemme} \\vert",
        decimal_round=4,
    )
    def in_vocabulary_proportion(self):
        return len(self.in_vocab_tokens) / self.normalized_dictionary_length()

    @corpus_metric(
        name="out of vocabulary",
        formula="$\\vert \mathcal{D}_{lemme} \\vert - \\vert \mathcal{D}_{lemme} \cap \mathcal{D}_{NLTK} \\vert \over \\vert \mathcal{D}_{lemme} \\vert$",
        decimal_round=4,
    )
    def out_of_vocabulary_proportion(self):
        return len(self.out_of_vocab_tokens) / self.normalized_dictionary_length()

    @corpus_metric(
        name="lexical diversity",
        formula="\\vert \mathcal{D} \\vert \over \\vert \mathcal{T} \\vert",
        decimal_round=4,
    )
    def lexical_diversity(self):
        return self.dictionary_length() / self.dictionary().N()

    @corpus_metric(
        name="token count",
        formula="\\vert \mathcal{T} \\vert",
        decimal_round=4,
    )
    def token_count(self):
        return self.dictionary().N()

    @corpus_metric(
        name="hapaxes",
        formula="\\vert \mathcal{D}_{hapax} \\vert \over \\vert \mathcal{D} \\vert",
        decimal_round=4,
    )
    def hapaxes_proportion(self):
        """
        Return the proportion of all samples that occur once (hapax legomena)
        """
        return len(self.dictionary().hapaxes()) / self.dictionary_length()

    @corpus_metric(name="dictionary length", formula="\\vert \mathcal{D} \\vert")
    def dictionary_length(self):
        return len(self.dictionary())

    @corpus_metric(
        name="lem dictionary length", formula="\\vert \mathcal{D}_{lemme} \\vert"
    )
    def normalized_dictionary_length(self):
        return len(self.lemmatized_words)

    @corpus_metric(
        name="uppercase items",
        formula="\\vert \\mathcal{O}_{upper} \\vert \over \\vert \\mathcal{O} \\vert",
        decimal_round=4,
    )
    def uppercase_item_proportion(self):
        return len(self.uppercase_sentences()) / self.unique_item_count()

    def values(self):
        metrics = [
            getattr(self, method)
            for method in dir(self)
            if hasattr(getattr(self, method), "is_metric")
        ]

        metric_names = []
        metric_formulas = []
        metric_values = []
        for metric in metrics:
            metric_names.append(metric.name)
            metric_formulas.append(f"${metric.formula}$")
            if metric.decimal_round > 0:
                metric_values.append(round(metric(), metric.decimal_round))
            else:
                metric_values.append(metric())

        readme_df = pd.DataFrame(
            data=[metric_formulas, metric_values],
            columns=metric_names,
            index=["formula", "title"],
        )
        return readme_df.transpose()
