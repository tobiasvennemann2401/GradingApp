from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk import word_tokenize
import nltk
import contractions
import re
import spacy

nlp = spacy.load("en_core_web_sm")

additionalStopwords = []

def preprocess_text(text):
    text = remove_new_line_and_tabs(text)
    text = lower_casing(text)
    text = expand_contractions(text)
    text = unify_end_of_sentences(text)
    text = remove_multiple_whitespaces(text)
    sentences = sentence_segmentation(text)
    text = ""
    for sentence in sentences:
        sentence = word_tokenization(sentence)
        sentence = pos_tagging(sentence)
        sentence = lemmatize_word_list(sentence)
        sentence = pos_tagging(sentence)
        sentence = remove_non_nouns_or_verbs(sentence)
        sentence = join(sentence)
        text += sentence + " "
    return text

def remove_new_line_and_tabs(string):
    # removes \n \t etc.
    return string.replace('\\n', ' ').replace('\n', ' ').replace('\t',' ').replace('\\', ' ')

def lower_casing(string):
    #change everything to lowercase
    return string.lower()

def expand_contractions(string):
    #expand "isn't" to "is not" etc.
    expanded_words = []
    for word in string.split():
        expanded_words.append(contractions.fix(word))
    return ' '.join(expanded_words)

def unify_end_of_sentences(string):
    #replaces character list below with '.'
    change_to_point = ['?','!']
    for char in change_to_point:
        string = string.replace(char, '.')
    return string

def remove_multiple_whitespaces(string):
    #removes all instances of multiple whitespaces
    return ' '.join(string.split())

def sentence_segmentation(string):
    #divide text into sentences
    return nltk.sent_tokenize(string)

def word_tokenization(string):
    return word_tokenize(string)

def lemmatize_word_list(word_list):
    #utilize pos tags to change words into base form e.g. 'companies' -> 'company'
    lemmatizer = WordNetLemmatizer()
    lemmatized_words = []
    for word in word_list:
        if word[1] == 'VERB':
            tag = 'v'
        elif word[1] == 'ADJ':
            tag = 'a'
        elif word[1] == 'ADV':
            tag = 'r'
        else:
            tag = 'n'
        lemmatized_words.append(lemmatizer.lemmatize(word[0], pos=tag))
    return lemmatized_words

def join(word_list):
    #join words to sentence
    return ' '.join(word_list)

def remove_non_nouns_or_verbs(tokens):
    #remove all non nouns from the list
    nouns = []
    for token in tokens:
        if token[1] == 'NOUN' or token[1] == 'VERB':
            nouns.append(token[0])
    return nouns

def pos_tagging(tokens):
    #tag word tokens with e.g. 'noun', 'verb' etc.
    return nltk.pos_tag(tokens, tagset = "universal")