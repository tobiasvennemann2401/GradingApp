import datastructure
import random
import spacy
import numpy as np
import re
from sklearn.cluster import KMeans

decay_factor = 0.0

# Load the spaCy English model (you need to install spaCy and download the model)
nlp = spacy.load("en_core_web_sm")

def dice_evaluation(question):
    for answer in question.student_answers:
        value = random.uniform(0, 1)
        rounded_value = round(value / 0.05) * 0.05
        rounded_value = max(0, min(1, rounded_value))
        rounded_value = round(rounded_value, 2)

        answer.suggested_score = rounded_value
        answer.display_text = answer.response

def preprocess_text(text):
    # Tokenize the text using spaCy
    doc = nlp(text)

    # Lowercase all tokens and remove filler words (stop words)
    cleaned_tokens = [token.text.lower() for token in doc if not token.is_stop]

    # Remove symbols and escaped characters
    cleaned_tokens = [re.sub(r'[.,\/#!$%\^&\*;:{}=\-_`~()\[\]\\\"\'\n\t]', '', token) for token in cleaned_tokens]

    return cleaned_tokens


def text_to_word_vector(text):
    # Preprocess the text
    cleaned_tokens = preprocess_text(text)

    # Filter out empty tokens
    cleaned_tokens = [token for token in cleaned_tokens if token.strip()]

    # Create a word vector (using a simple example of a bag-of-words vector)
    unique_tokens = list(set(cleaned_tokens))
    word_vector = np.zeros(len(unique_tokens))

    for i, token in enumerate(unique_tokens):
        word_vector[i] = cleaned_tokens.count(token)

    return unique_tokens, word_vector  # Return both the dictionary and word vector


def word_vector_for_question(text):
    # Preprocess the text
    cleaned_tokens = preprocess_text(text)

    # Filter out empty tokens
    cleaned_tokens = [token for token in cleaned_tokens if token.strip()]

    # Create a word vector (using a simple example of a bag-of-words vector)
    unique_tokens = list(set(cleaned_tokens))
    word_vector = np.zeros(len(unique_tokens))

    for i, token in enumerate(unique_tokens):
        word_vector[i] = cleaned_tokens.count(token)

    return unique_tokens, word_vector


def calculate_word_vector(text, dictionary):
    # Preprocess the text
    cleaned_tokens = preprocess_text(text)

    # Initialize a word vector with zeros
    word_vector = np.zeros(len(dictionary))

    # Count the frequency of each token in the text
    for token in cleaned_tokens:
        if token in dictionary:
            word_vector[dictionary.index(token)] += 1

    return word_vector

def euclidean_distance(vector1, vector2):
    return np.linalg.norm(vector1 - vector2)


def map_distance_to_similarity(distance):
    global decay_factor

    # Calculate the similarity score using exponential decay
    similarity = 1 / (1 + decay_factor * distance)

    return round(similarity, 2)


def colorize_words_in_text(text, dictionary):
    for word in dictionary:
        # Wrap the word in an HTML tag with a style attribute to set the text color to red
        replacement = f'<span style="color: red;">{word}</span>'
        text = text.replace(word, replacement)

    return text

def lexical_euclidean_similarity(question):
    dictionary, reference_vector = text_to_word_vector(question.reference_answers[0].text)
    for student_answer in question.student_answers:
        word_vector = calculate_word_vector(student_answer.response, reference_vector)
        student_answer.suggested_score = map_distance_to_similarity(euclidean_distance(word_vector, reference_vector))
        student_answer.display_text = colorize_words_in_text(student_answer.response, dictionary)

def kmeans_clustering(vectors, clusters):
    return KMeans(
        n_clusters=clusters,
        max_iter=100,
        n_init=500,
        random_state=24,
        algorithm="lloyd"
    ).fit(vectors)
