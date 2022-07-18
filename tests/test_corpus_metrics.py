from hn_eda.story_corpus import StoryCorpusReader
from hn_eda.corpus_metrics import CorpusMetrics


def test_load_corpus():
    story_corpus = StoryCorpusReader()
    corpus_metric = CorpusMetrics(corpus=story_corpus)
    metrics_df = corpus_metric.values()
