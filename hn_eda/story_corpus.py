import re
from nltk.corpus.reader.api import CorpusReader
from nltk.corpus.reader.util import StreamBackedCorpusView

from hn_eda.data_preparation import load_topstories_from_zip, TOPSTORIES_PATH
from hn_eda.tokenizers import StoryTokenizer


class StoryCorpusReader(CorpusReader):
    CorpusView = StreamBackedCorpusView
    """
    The corpus view class used by this reader.
    """
    _items = None

    def __init__(self, word_tokenizer=StoryTokenizer(), encoding="utf8"):
        """
        :param file_path: The path to json.
        :param word_tokenizer: Tokenizer for breaking the text of Story into
            smaller units, including but not limited to words.
        """

        CorpusReader.__init__(
            self, str(TOPSTORIES_PATH.parent), [TOPSTORIES_PATH.name], encoding
        )

        if TOPSTORIES_PATH.stat().st_size == 0:
            raise ValueError(f"File {TOPSTORIES_PATH} is empty")
        """Check that all user-created corpus files are non-empty."""

        self._word_tokenizer = word_tokenizer

    def docs(self):
        """
        Returns the Stories objects
        :return: list of dictionaries deserialised from JSON.
        :rtype: list(dict)
        """
        return load_topstories_from_zip()

    def items(self):
        if self._items == None:
            self._items = self.titles()
        return self._items

    def titles(self):
        """
        Returns only the titles content of Stories
        """
        stories = self.docs()
        return stories["title"].tolist()

    def uppercase(self):
        regex_pattern = r"^[^a-z]*$"
        return [
            title
            for title in self.items()
            if re.match(regex_pattern, title) is not None
        ]

    def uppercase_sentences(
        self,
    ):
        tokenizer = self._word_tokenizer
        return [tuple(tokenizer.tokenize(t)) for t in self.uppercase()]

    def unique_uppercase_sentences(self):
        return set(self.uppercase_sentences())

    def sentences(self):
        """
        :return: a list of the text content of Stories as
            as a list of words.. and punctuation symbols.
        :rtype: list(list(str))
        """
        tokenizer = self._word_tokenizer
        return [tuple(tokenizer.tokenize(t)) for t in self.items()]

    def unique_sentences(self):
        """
        :return: a list of the text content of Stories as
            as a list of words.. and punctuation symbols.
        :rtype: list(list(str))
        """
        return set(self.sentences())

    def words(self):
        """
        :return: a list of the tokens of Stories.
        :rtype: list(str)
        """
        tokens = []
        for title_sentence in self.sentences():
            tokens += title_sentence
        return tokens

    def unique_words(self):
        """
        :return: a list of the tokens of Stories.
        :rtype: list(str)
        """
        tokens = []
        for title_sentence in self.unique_sentences():
            tokens += title_sentence
        return tokens
