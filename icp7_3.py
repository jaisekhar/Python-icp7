import nltk
nltk.download('punkt')
nltk.download('wordnet')
nltk.download('averaged_perceptron_tagger')
nltk.download('maxent_ne_chunker')
nltk.download('words')
from nltk.util import ngrams
from nltk.tokenize import word_tokenize
from nltk.tokenize import sent_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.stem import PorterStemmer
from nltk import ne_chunk
from collections import Counter
ps = PorterStemmer()

lemmatizer = WordNetLemmatizer()
text = open('input.txt', encoding="utf8").read()

w_tokens =word_tokenize(text)
s_tokens = sent_tokenize(text)
print("Word tokens:",w_tokens)
print("\nSentence tokens:",s_tokens)

trigrams = ngrams(w_tokens,3)
print("\nTrigrams: ",list(trigrams))

lemmatized_output = ' '.join([lemmatizer.lemmatize(w) for w in w_tokens])
print("\nLemmatization:\n",lemmatized_output)

stemmed_output = ' '.join([ps.stem(w) for w in w_tokens])
print("\nStemming:\n",stemmed_output)

n_pos = nltk.pos_tag(w_tokens)
print("\nParts of Speech :", n_pos)

noe = ne_chunk(n_pos)
print("\nNamed Entity Recognition :", noe)