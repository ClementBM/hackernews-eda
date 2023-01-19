from nltk.tokenize.casual import _replace_html_entities
from hn_eda.tokenizers import StoryTokenizer
import re


def test_replace_html_entities():
    text = " &#nnnn; and &#hhhh; &nbsp;  &gt; <br/> <div>this is div</div> "
    safe_text = _replace_html_entities(text)
    assert safe_text == "  and  \xa0  > <br/> <div>this is div</div> "


def test_find_all():
    text = "Open-Sourcing our Firmware"
    tokenizer = StoryTokenizer()
    words = tokenizer.WORD_RE.findall(text)
    assert words


def test_uppercased_title_token():
    title = "This week in KDE: You wanted stability? Here\u2019s some stability"
    safe_title = _replace_html_entities(title)

    uppercased_tokens = re.findall(r"[A-Z]{2,}", safe_title)
    assert uppercased_tokens == ["KDE"]


def test_uppercased_title():
    regex_pattern = r"^[^a-z]*$"
    uppercase_title = "THIS IS A TEST TITLE 2 !!"
    is_uppercase = re.match(regex_pattern, uppercase_title) is not None
    assert is_uppercase == True
