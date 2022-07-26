import re
from pathlib import Path
from statistics import mean, median, stdev
import pandas as pd

from nltk import FreqDist
from nltk.corpus import stopwords, words
from nltk.stem import WordNetLemmatizer

from hn_eda.story_corpus import CorpusReaderBase

ROOT = Path(__file__).parent


def corpus_metric(name, formula, description="", order=0, decimal_round=0):
    def decorator(function):
        function.is_metric = True
        function.name = name
        function.formula = formula
        function.decimal_round = decimal_round
        function.description = description
        function.order = order
        return function

    return decorator


class CorpusMetrics:
    _dictionary = None
    _unique_sentences = None
    _uppercase_sentences = None
    _sentence_lengths = None
    _uppercased_tokens = None

    def __init__(self, corpus: CorpusReaderBase, item_name):
        self.corpus = corpus
        self.item_name = item_name

        self._lemmatize_dictionary()
        self._compute_oov()

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
                sentence
                for sentence in self.unique_sentences()
                if re.match(regex_pattern, sentence) is not None
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

    def uppercased_tokens(self):
        if self._uppercased_tokens == None:
            self._uppercased_tokens = []
            for sentence in self.unique_sentences():
                if sentence in self.uppercase_sentences():
                    continue

                tokens = re.findall(r"[A-Z]{2,}", sentence)
                self._uppercased_tokens += tokens

        return self._uppercased_tokens

    @corpus_metric(
        order=0,
        name="Count",
        description="Number of {}",
        formula="\\vert \mathcal{O} \\vert",
    )
    def item_count(self):
        return len(self.corpus.sentences())

    @corpus_metric(
        order=1,
        name="Unique count",
        description="Number of unique {}",
        formula="\\vert \mathcal{O}_{unique} \\vert",
    )
    def unique_item_count(self):
        return len(self.unique_sentences())

    @corpus_metric(
        order=2,
        description="Total number of tokens",
        name="Token count",
        formula="\\vert \mathcal{T} \\vert",
        decimal_round=4,
    )
    def token_count(self):
        return self.dictionary().N()

    @corpus_metric(
        order=3,
        description="Total number of unique tokens",
        name="Dictionary length",
        formula="\\vert \mathcal{D} \\vert",
    )
    def dictionary_length(self):
        return len(self.dictionary())

    @corpus_metric(
        order=4,
        description="Total number of unique lemmetized tokens",
        name="Lem dictionary length",
        formula="\\vert \mathcal{D}_{lemme} \\vert",
    )
    def alpha_num_dictionary_length(self):
        return len(self.lemmatized_words)

    @corpus_metric(
        order=5,
        description="Total number of unique alpha lemmetized tokens",
        name="Alpha lem dictionary length",
        formula="\\vert \mathcal{D}_{\\alpha-lemme} \\vert",
    )
    def alpha_dictionary_length(self):
        return len(self.out_of_vocab_tokens) + len(self.in_vocab_tokens)

    @corpus_metric(
        order=6,
        description="Average number of tokens",
        name="Average length",
        formula="\\bar{M_i}",
        decimal_round=2,
    )
    def average_item_length(self):
        return mean(self.sentence_lengths())

    @corpus_metric(
        order=7,
        description="Minimum and maximum number of tokens",
        name="Min and Max length",
        formula="\{ min(M_i), max(M_i) \}",
    )
    def extremum_item_length(self):
        return min(self.sentence_lengths()), max(self.sentence_lengths())

    @corpus_metric(
        order=8,
        description="Median number of tokens",
        name="Median length",
        formula="\\tilde{M_i}",
        decimal_round=2,
    )
    def median_item_length(self):
        return median(self.sentence_lengths())

    @corpus_metric(
        order=9,
        description="Standard deviation of the number of tokens",
        name="Std length",
        formula="s_{M_i}",
        decimal_round=2,
    )
    def std_item_length(self):
        return stdev(self.sentence_lengths())

    @corpus_metric(
        order=10,
        description="Proportion of {} that appears more than once",
        name="Duplicate proportion",
        formula="\\vert \mathcal{O} \\vert - \\vert \mathcal{O}_{unique} \\vert \over \\vert \mathcal{O} \\vert",
        decimal_round=4,
    )
    def duplicate_proportion(self):
        return (self.item_count() - self.unique_item_count()) / self.item_count()

    @corpus_metric(
        order=11,
        description="Frequency of numerical tokens",
        name="Numerical frequency",
        formula="\\vert \mathcal{T}_{numerical} \\vert \over \\vert \mathcal{T} \\vert",
        decimal_round=4,
    )
    def numerical_frequency(self):
        numerical_frequencies = [
            self.dictionary().freq(token) for token in self.numerical_tokens
        ]
        return sum(numerical_frequencies)

    @corpus_metric(
        order=12,
        description="Proportion of numerical tokens",
        name="Numerical proportion",
        formula="\\vert \mathcal{D}_{numerical} \\vert \over \\vert \mathcal{D}_{lemme} \\vert",
        decimal_round=4,
    )
    def numerical_proportion(self):
        return len(self.numerical_tokens) / self.alpha_num_dictionary_length()

    @corpus_metric(
        order=13,
        description="Proportion of tokens inside the NLTK vocabulary",
        name="In vocabulary",
        formula="\\vert \mathcal{D}_{\\alpha-lemme} \cap \mathcal{D}_{NLTK} \\vert \over \\vert \mathcal{D}_{\\alpha-lemme} \\vert",
        decimal_round=4,
    )
    def in_vocabulary_proportion(self):
        return len(self.in_vocab_tokens) / self.alpha_dictionary_length()

    @corpus_metric(
        order=14,
        description="Proportion of tokens outside the NLTK vocabulary",
        name="Out of vocabulary",
        formula="$\\vert \mathcal{D}_{\\alpha-lemme} \\vert - \\vert \mathcal{D}_{\\alpha-lemme} \cap \mathcal{D}_{NLTK} \\vert \over \\vert \mathcal{D}_{\\alpha-lemme} \\vert$",
        decimal_round=4,
    )
    def out_of_vocabulary_proportion(self):
        return len(self.out_of_vocab_tokens) / self.alpha_dictionary_length()

    @corpus_metric(
        order=15,
        description="Dictionary count over the token count",
        name="Lexical diversity",
        formula="\\vert \mathcal{D} \\vert \over \\vert \mathcal{T} \\vert",
        decimal_round=4,
    )
    def lexical_diversity(self):
        return self.dictionary_length() / self.dictionary().N()

    @corpus_metric(
        order=16,
        description="Proportion of token that occur once (hapax legomena)",
        name="Hapaxes",
        formula="\\vert \mathcal{D}_{hapax} \\vert \over \\vert \mathcal{D} \\vert",
        decimal_round=4,
    )
    def hapaxes_proportion(self):
        return len(self.dictionary().hapaxes()) / self.dictionary_length()

    @corpus_metric(
        order=17,
        description="Proportion of uppercased {}",
        name="Uppercase items",
        formula="\\vert \\mathcal{O}_{upper} \\vert \over \\vert \\mathcal{O} \\vert",
        decimal_round=4,
    )
    def uppercase_item_proportion(self):
        return len(self.uppercase_sentences()) / self.unique_item_count()

    @corpus_metric(
        order=18,
        description="Proportion of uppercased token",
        name="Uppercased token proportion",
        formula="\\vert \mathcal{T}_{uppercase} \\vert \over \\vert \mathcal{T} \\vert",
        decimal_round=4,
    )
    def uppercase_token_proportion(self):
        return len(self.uppercased_tokens()) / self.dictionary().N()

    def values(self):
        metrics = [
            getattr(self, method)
            for method in dir(self)
            if hasattr(getattr(self, method), "is_metric")
        ]

        metrics.sort(key=lambda x: x.order)

        metric_names = []
        metric_formulas = []
        metric_values = []
        metric_descriptions = []
        for metric in metrics:
            metric_names.append(metric.name)
            metric_descriptions.append(metric.description.format(self.item_name))
            metric_formulas.append(f"${metric.formula}$")
            if metric.decimal_round > 0:
                metric_values.append(round(metric(), metric.decimal_round))
            else:
                metric_values.append(metric())

        readme_df = pd.DataFrame(
            data=[metric_names, metric_formulas, metric_values, metric_descriptions],
            index=["Name", "Formula", "Value", "Description"],
        )
        return readme_df.transpose()
