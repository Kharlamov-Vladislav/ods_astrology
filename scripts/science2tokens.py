from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.corpus import stopwords
import pandas as pd


def from_texts_to_tokens(df):
    """
    The function takes as input a pandas dataframe containing two columns ['title', 'abstract'],
    and then writes to four files abstract_tokens_ngrams_{ngram}.txt, title_tokens_ngrams_{ngram}.txt,
    all ngrams with a length of 1 or 2.
    """

    corpus_abstract, corpus_title = df['abstract'], df['title']

    for ngram in range(1, 3):
        # min_df -- Removes all words that occur in less than 0.1% of documents
        # max_df -- Removes all words that appear in more than 75% of documents
        vectorizer = TfidfVectorizer(stop_words=stopwords.words('english'), token_pattern=r'\b[a-zA-Z]+\b',
                                     min_df=0.001, max_df=0.75, ngram_range=(ngram, ngram))
        with open(f'abstract_tokens_ngrams_{ngram}.txt', 'w') as abstract_f, \
                open(f'title_tokens_ngrams_{ngram}.txt', 'w') as title_f:
            # Write tokens abstract
            vectorizer.fit(corpus_abstract)
            abstract_f.write(repr(vectorizer.get_feature_names()))
            # Write tokens title
            vectorizer.fit(corpus_title)
            title_f.write(repr(vectorizer.get_feature_names()))


NAME_DF = 'nature_computer_science.csv'
df = pd.read_csv(NAME_DF)

if __name__ == "__main__":
    from_texts_to_tokens(df)