from typing import Any, List, Tuple, Union
from numpy import ndarray
import nltk
from nltk.tokenize import RegexpTokenizer
from nltk.corpus import stopwords

nltk.download('punkt')
TOKENIZE_TEXT = RegexpTokenizer(r'\w+')
STOP_WORDS = set(stopwords.words())


class Preprocess:
    def __init__(self) -> None:
        pass

    def load_data(path: str) -> Tuple[List[str], List[str]]:
        """Loads data from file. Each except first (header) is a datapoint
        containing ID, Label, Email (content) separated by "\t".

        Args:
            path: Path to file from which to load data

        Returns:
            List of email contents and a list of lobels coresponding to each email.
        """
        data_collection = None
        with open(path) as f:
            data_collection = f.readlines()

        size = (len(data_collection) - 1)
        contents = [''] * size
        labels = [0] * size

        for i in range(1, size):
            entry = data_collection[i].split("\t")
            label = entry[1]
            trim_content = entry[2].replace('"', '').strip()
            content = trim_content.strip("\n")

            labels[i-1] = 1 if label == 'spam' else 0
            contents[i-1] = content
        return (contents, labels)

    def preprocess(self, doc: str) -> str:
        """Preprocesses text to prepare it for feature extraction.

        Args:
            doc: String comprising the unprocessed contents of some email file.

        Returns:
            String comprising the corresponding preprocessed text.
        """
        processed_txt = ""
        tokenize_doc = TOKENIZE_TEXT.tokenize(doc)
        for w in tokenize_doc:
            if w not in STOP_WORDS:
                processed_txt += ' ' + w
        processed_txt = trim_suf_s(processed_txt)
        return processed_txt

    def preprocess_multiple(self, docs: List[str]) -> List[str]:
        """Preprocesses multiple texts to prepare them for feature extraction.

        Args:
            docs: List of strings, each consisting of the unprocessed contents
                of some email file.

        Returns:
            List of strings, each comprising the corresponding preprocessed
                text.
        """
        return [self.preprocess(doc) for doc in docs]


def has_suffix(s):
    suf = list(map(str, s))
    if suf[len(suf)-1] == "s":
        suf.pop()
        return True, ''.join(suf)
    return False, None


def trim_suf_s(terms: str) -> List[str]:
    terms = terms.split()
    size = len(terms)
    no_suffix = [""] * size
    for i in range(0, size):
        has, s = has_suffix(terms[i])
        if has:
            no_suffix[i] = s
        else:
            no_suffix[i] = terms[i]

    return ' '.join(no_suffix)
