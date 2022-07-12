import re
from pathlib import Path
from statistics import mean, median, stdev

from nltk import FreqDist
from nltk.book import Text
from nltk.corpus import stopwords, words
from nltk.stem import WordNetLemmatizer

from nltk.corpus.reader.api import CorpusReaderBase

ROOT = Path(__file__).parent
from functools import wraps


def corpus_metric(name, formula, decimal_round=0):
    def decorate(fn):
        @wraps(fn)
        def wrapper(*args, **kwargs):
            fn.is_metric = True
            fn.name = name
            fn.formula = formula
            fn.decimal_round = decimal_round
            return fn

        return wrapper

    return decorate


class CorpusMetrics:
    def __init__(self, corpus: CorpusReaderBase):
        self.corpus = corpus

        self._lemmatize_dictionary()
        self._compute_oov()

    def uppercase_sentences_words(
        self,
    ):
        if self._uppercase_sentences_words == None:
            tokenizer = self._word_tokenizer
            self._uppercase_sentences_words = [
                tuple(tokenizer.tokenize(t)) for t in self.uppercase()
            ]
        return self._uppercase_sentences_words

    def unique_sentences(self):
        """
        :return: a list of the text content of Stories as
            as a list of words.. and punctuation symbols.
        :rtype: list(list(str))
        """
        if self._unique_sentences == None:
            self._unique_sentences = set(self.corpus.sentences())
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
        return [
            len(sentence_tokens) for sentence_tokens in self.uppercase_sentences_words()
        ]

    def dictionary(self):
        if self._dictionary == None:
            stop_words = set(stopwords.words("english"))

            meaningful_words = [
                word for word in self.story_text if word.casefold() not in stop_words
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
        name="numerical frequency", formula="$d_{numerical} \over d$", decimal_round=4
    )
    def numerical_frequency(self):
        numerical_frequencies = [
            self.dictionary().freq(token) for token in self.numerical_tokens
        ]
        return sum(numerical_frequencies)

    @corpus_metric(
        name="duplicate proportion",
        formula="$ \\vert \mathcal{C} \\vert - \\vert \mathcal{C}_{unique} \\vert \over \\vert \mathcal{C} \\vert $",
        decimal_round=4,
    )
    def duplicate_proportion(self):
        return (self.item_count() - self.unique_item_count()) / self.item_count()

    @corpus_metric(name="count", formula="$ \\vert \mathcal{C} \\vert $")
    def item_count(self):
        return len(self.corpus.items())

    @corpus_metric(
        name="unique count", formula="$ \\vert \mathcal{C}_{unique} \\vert $"
    )
    def unique_item_count(self):
        return len(self.unique_sentences_tokens)

    @corpus_metric(name="average length", formula="$\\bar{n}$", decimal_round=2)
    def average_item_length(self):
        return mean(self.sentence_lengths)

    @corpus_metric(name="std length", formula="$s_{n}$", decimal_round=2)
    def std_item_length(self):
        return stdev(self.sentence_lengths)

    @corpus_metric(name="median length", formula="$\\tilde{n}$", decimal_round=2)
    def median_item_length(self):
        return median(self.sentence_lengths)

    @corpus_metric(name="min max length", formula="$min(n), max(n)$")
    def extremum_item_length(self):
        return min(self.sentence_lengths), max(self.sentence_lengths)

    @corpus_metric(
        name="in vocabulary",
        formula="$\\vert \mathcal{D}_{lemme} \cap \mathcal{D}_{NLTK} \\vert \over \\vert \mathcal{D}_{lemme} \\vert$",
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
        name="numerical proportion", formula="$d_{numerical} \over d$", decimal_round=4
    )
    def numerical_proportion(self):
        return len(self.numerical_tokens) / self.normalized_dictionary_length()

    @corpus_metric(
        name="lexical diversity",
        formula="$\\vert \mathcal{D} \\vert \over \\vert \mathcal{T} \\vert$",
        decimal_round=4,
    )
    def lexical_diversity(self):
        return self.dictionary_length() / self.dictionary().N()

    @corpus_metric(
        name="hapaxes",
        formula="$\\vert \mathcal{D}_{hapax} \\vert \over \\vert \mathcal{D} \\vert$",
        decimal_round=4,
    )
    def hapaxes_proportion(self):
        """
        Return the proportion of all samples that occur once (hapax legomena)
        """
        return len(self.dictionary().hapaxes()) / self.dictionary_length()

    @corpus_metric(name="dictionary length", formula="$\\vert \mathcal{D} \\vert$")
    def dictionary_length(self):
        return len(self.dictionary())

    @corpus_metric(
        name="lem dictionary length", formula="$\\vert \mathcal{D}_{lemme} \\vert$"
    )
    def normalized_dictionary_length(self):
        return len(self.lemmatized_words)

    @corpus_metric(
        name="uppercase items", formula="$n_{upper} \over n_{unique}$", decimal_round=4
    )
    def uppercase_item_proportion(self):
        return len(self.uppercase_items) / self.unique_item_count()
