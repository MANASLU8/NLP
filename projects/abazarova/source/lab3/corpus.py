import re
from lab1.tokenizer import tok_tok, sent_tok
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords


def corpus(line):
    # Cleaing the text
    text=line[0]
    processed = text.lower()
    processed = re.sub('[^a-zA-Z]', ' ', processed)
    processed = re.sub(r'\s+', ' ', processed)

    # Preparing the dataset
    all_sentences = sent_tok(processed)
    all_words = [tok_tok(sent) for sent in all_sentences]

    # Removing Stop Words
    for i in range(len(all_words)):
        all_words[i] = [w for w in all_words[i] if w not in stopwords.words('english')]

    return all_words
