import typing
from pathlib import Path

import re
from nltk.corpus.reader.api import CorpusReader
from nltk.corpus.reader.util import StreamBackedCorpusView
from nltk.tokenize.api import TokenizerI
from nltk.tokenize.casual import _replace_html_entities

from hn_eda.data_preparation import load_topstories_from_zip, TOPSTORIES_PATH

URLS = r"""			    # Capture 1: entire matched URL
  (?:
  https?:				# URL protocol and colon
    (?:
      /{1,3}		    # 1-3 slashes
      |					#   or
      [a-z0-9%]			# Single letter or digit or '%'
                        # (Trying not to match e.g. "URI::Escape")
    )
    |					#   or
                        # looks like domain name followed by a slash:
    [a-z0-9.\-]+[.]
    (?:[a-z]{2,13})
    /
  )
  (?:					# One or more:
    [^\s()<>{}\[\]]+	# Run of non-space, non-()<>{}[]
    |					#   or
    \([^\s()]*?\([^\s()]+\)[^\s()]*?\) # balanced parens, one level deep: (...(...)...)
    |
    \([^\s]+?\)			# balanced parens, non-recursive: (...)
  )+
  (?:					# End with:
    \([^\s()]*?\([^\s()]+\)[^\s()]*?\) # balanced parens, one level deep: (...(...)...)
    |
    \([^\s]+?\)			# balanced parens, non-recursive: (...)
    |					#   or
    [^\s`!()\[\]{};:'".,<>?«»“”‘’]	# not a space or one of these punct chars
  )
  |					    # OR, the following to match naked domains:
  (?:
  	(?<!@)			    # not preceded by a @, avoid matching foo@_gmail.com_
    [a-z0-9]+
    (?:[.\-][a-z0-9]+)*
    [.]
    (?:[a-z]{2,13})
    \b
    /?
    (?!@)			    # not succeeded by a @,
                        # avoid matching "foo.na" in "foo.na@example.com"
  )
"""

REGEXPS = (
    URLS,
    # HTML tags:
    r"""<[^>\s]+>""",
    # ASCII Arrows
    r"""[\-]+>|<[\-]+""",
    # email addresses
    r"""[\w.+-]+@[\w-]+\.(?:[\w-]\.?)+[\w-]""",
    # Remaining word types:
    r"""
    (?:[^\W\d_](?:[^\W\d_]|['\-_])+[^\W\d_]) # Words with apostrophes or dashes.
    |
    (?:[+\-]?\d+[,/.:-]\d+[+\-]?)  # Numbers, including fractions, decimals.
    |
    (?:[\w_]+)                     # Words without apostrophes or dashes.
    |
    (?:\.(?:\s*\.){1,})            # Ellipsis dots.
    |
    (?:\S)                         # Everything else that isn't whitespace.
    """,
)


class StoryTokenizer(TokenizerI):
    r"""
    Tokenizer for stories.

        >>> from nltk.tokenize import StoryTokenizer
        >>> tknzr = StoryTokenizer()
        >>> s0 = "This is a cooool #dummysmiley: :-) :-P <3 and some arrows < > -> <--"
        >>> tknzr.tokenize(s0)
        ['This', 'is', 'a', 'cooool', '#dummysmiley', ':', ':-)', ':-P', '<3'
        , 'and', 'some', 'arrows', '<', '>', '->', '<--']
    """

    # Values used to lazily compile WORD_RE
    # which are the core tokenizing regexes.
    _WORD_RE = None

    def __init__(
        self,
        preserve_case=True,
    ):
        """
        Create a `StoryTokenizer` instance with settings for use in the `tokenize` method.

        :param preserve_case: Flag indicating whether to preserve the casing (capitalisation)
            of text used in the `tokenize` method. Defaults to True.
        :type preserve_case: bool
        """
        self.preserve_case = preserve_case

    def tokenize(self, text: str) -> typing.List[str]:
        """Tokenize the input text.

        :param text: str
        :rtype: list(str)
        :return: a tokenized list of strings; joining this list returns\
        the original string if `preserve_case=False`.
        """
        # Fix HTML character entities:
        safe_text = _replace_html_entities(text)

        # Tokenize
        words = self.WORD_RE.findall(safe_text)

        # Possibly alter the case
        if not self.preserve_case:
            words = list(map((lambda x: x.lower()), words))
        return words

    @property
    def WORD_RE(self) -> "re.Pattern":
        """Core StoryTokenizer regex"""
        # Compiles the regex for this and all future instantiations of TweetTokenizer.
        if not type(self)._WORD_RE:
            type(self)._WORD_RE = re.compile(
                f"({'|'.join(REGEXPS)})",
                re.VERBOSE | re.I | re.UNICODE,
            )
        return type(self)._WORD_RE


class StoryCorpusReader(CorpusReader):
    CorpusView = StreamBackedCorpusView
    """
    The corpus view class used by this reader.
    """

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

    def titles(self):
        """
        Returns only the titles content of Stories
        """
        stories = self.docs()
        return stories["title"].tolist()

    def uppercase(self, items: list):
        regex_pattern = r"^[^a-z]*$"
        return [title for title in items if re.match(regex_pattern, title) is not None]

    def uppercase_sentences(self, items: list):
        tokenizer = self._word_tokenizer
        return [tuple(tokenizer.tokenize(t)) for t in self.uppercase(items)]

    def unique_uppercase_sentences(self, items: list):
        return set(self.uppercase_sentences(items))

    def sentences(self, items):
        """
        :return: a list of the text content of Stories as
            as a list of words.. and punctuation symbols.
        :rtype: list(list(str))
        """
        tokenizer = self._word_tokenizer
        return [tuple(tokenizer.tokenize(t)) for t in items]

    def unique_sentences(self, items: list):
        """
        :return: a list of the text content of Stories as
            as a list of words.. and punctuation symbols.
        :rtype: list(list(str))
        """
        return set(self.sentences(items))

    def words(self, items: list):
        """
        :return: a list of the tokens of Stories.
        :rtype: list(str)
        """
        tokens = []
        for title_sentence in self.sentences(items):
            tokens += title_sentence
        return tokens

    def unique_words(self, items: list):
        """
        :return: a list of the tokens of Stories.
        :rtype: list(str)
        """
        tokens = []
        for title_sentence in self.unique_sentences(items):
            tokens += title_sentence
        return tokens
