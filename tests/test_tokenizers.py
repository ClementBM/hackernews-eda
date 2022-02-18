from nltk.tokenize.casual import _replace_html_entities
from hn_eda.tokenizers import StoryTokenizer


def test_replace_html_entities():
    text = " &#nnnn; and &#hhhh; &nbsp;  &gt; <br/> <div>this is div</div> "
    safe_text = _replace_html_entities(text)
    assert safe_text == "  and  \xa0  > <br/> <div>this is div</div> "


def test_find_all():
    text = ""
    tokenizer = StoryTokenizer()
    words = tokenizer.WORD_RE.findall(text)
    assert words
