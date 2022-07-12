from nltk.corpus.reader.api import CorpusReader
from nltk.corpus.reader.util import StreamBackedCorpusView, concat, ZipFilePathPointer

from hn_eda.data_preparation import TOPSTORIES_JSONL
from hn_eda.tokenizers import StoryTokenizer
import json
import os

from abc import abstractmethod


class CorpusReaderBase(CorpusReader):
    @abstractmethod
    def texts(self):
        pass

    @abstractmethod
    def sentences(self):
        pass

    @abstractmethod
    def words(self):
        pass


class StoryCorpusReader(CorpusReaderBase):
    corpus_view = StreamBackedCorpusView
    """
    The corpus view class used by this reader.
    """
    _titles = None

    def __init__(self, word_tokenizer=StoryTokenizer(), encoding="utf8"):
        """
        :param word_tokenizer: Tokenizer for breaking the text of Story into
            smaller units, including but not limited to words.
        """

        CorpusReader.__init__(
            self, str(TOPSTORIES_JSONL.parent), [TOPSTORIES_JSONL.name], encoding
        )

        for path in self.abspaths(self._fileids):
            if isinstance(path, ZipFilePathPointer):
                pass
            elif os.path.getsize(path) == 0:
                raise ValueError(f"File {path} is empty")
        """Check that all user-created corpus files are non-empty."""

        self._word_tokenizer = word_tokenizer

    def docs(self, fileids=None):
        """
        Returns the Stories objects
        :return: list of dictionaries deserialised from JSON.
        :rtype: list(dict)
        """
        return concat(
            [
                self.corpus_view(path, self._read_stories, encoding=enc)
                for (path, enc, fileid) in self.abspaths(fileids, True, True)
            ]
        )

    def titles(self):
        """
        Returns only the titles content of Stories
        """
        if self._titles == None:
            titles = self.docs()
            standard_titles = []
            for jsono in titles:
                text = jsono["title"]
                if isinstance(text, bytes):
                    text = text.decode(self.encoding)

                standard_titles.append(text)
            self._titles = standard_titles
        return self._titles

    def texts(self):
        return self.titles()

    def sentences(self):
        """
        :return: a list of the text content of Stories as
            as a list of words.. and punctuation symbols.
        :rtype: list(list(str))
        """
        tokenizer = self._word_tokenizer
        return [tuple(tokenizer.tokenize(t)) for t in self.titles()]

    def words(self):
        """
        :return: a list of the tokens of Stories.
        :rtype: list(str)
        """
        tokens = []
        for title_sentence in self.sentences():
            tokens += title_sentence
        return tokens

    def _read_stories(self, stream):
        """
        Assume that each line in stream is a JSON serialised object
        """
        stories = []
        for i in range(10):
            line = stream.readline()
            if not line:
                return stories
            story = json.loads(line)
            stories.append(story)
        return stories
