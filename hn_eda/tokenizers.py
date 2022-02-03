import typing
import re
from nltk.tokenize.api import TokenizerI
from nltk.tokenize.casual import _replace_html_entities
import string

REGEXPS = (
    # HTML tags:
    r"""<[^<>]+>""",
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
        >>> s0 = "SICP JavaScript Version 2022 pdf"
        >>> tknzr.tokenize(s0)
        ['SICP', 'JavaScript', 'Version', '2022', 'pdf']
    """

    # Values used to lazily compile WORD_RE
    # which are the core tokenizing regexes.
    _WORD_RE = None

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

        # Remove punctuation
        words = [
            word
            for word in words
            if re.match(f"[{re.escape(string.punctuation)}——–’‘“”×]", word.casefold())
            == None
        ]

        return words

    @property
    def WORD_RE(self) -> "re.Pattern":
        """StoryTokenizer regex"""
        # Compiles the regex for this and all future instantiations of TweetTokenizer.
        if not type(self)._WORD_RE:
            type(self)._WORD_RE = re.compile(
                f"({'|'.join(REGEXPS)})",
                re.VERBOSE | re.I | re.UNICODE,
            )
        return type(self)._WORD_RE
