# organise imports
import re

import pandas as pd
import Levenshtein
from nltk import WordNetLemmatizer
from sklearn.cluster import AgglomerativeClustering
import numpy as np
from collections import Counter
import fuzzy
import nltk
from nltk.stem import PorterStemmer
from nltk.metrics import edit_distance
from nltk.corpus import stopwords, wordnet
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import CountVectorizer
from collections import Counter
from itertools import combinations

# Download the stopwords from NLTK
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('averaged_perceptron_tagger')

# Initialize the Porter stemmer
stemmer = PorterStemmer()
lemmatizer = WordNetLemmatizer()

# Get the list of stopwords
stop_words = set(stopwords.words('english')) - {'no', 'not'}

pd.options.mode.chained_assignment = None

# Define a dictionary of contractions
contraction_mapping = {
    "isn't": "is not",
    "aren't": "are not",
    "wasn't": "was not",
    "weren't": "were not",
    "haven't": "have not",
    "hasn't": "has not",
    "hadn't": "had not",
    "won't": "will not",
    "wouldn't": "would not",
    "don't": "do not",
    "doesn't": "does not",
    "didn't": "did not",
    "can't": "cannot",
    "couldn't": "could not",
    "shouldn't": "should not",
    "mightn't": "might not",
    "mustn't": "must not",
    "isnt": "is not",
    "arent": "are not",
    "wasnt": "was not",
    "werent": "were not",
    "havent": "have not",
    "hasnt": "has not",
    "hadnt": "had not",
    "wont": "will not",
    "wouldnt": "would not",
    "dont": "do not",
    "doesnt": "does not",
    "didnt": "did not",
    "cant": "cannot",
    "couldnt": "could not",
    "shouldnt": "should not",
    "mightnt": "might not",
    "mustnt": "must not",
}


def reset_preprocessing(df):
    df['answer'] = df['answer_display']
    return df


def to_expand_contractions(text):
    global contraction_mapping
    contractions_pattern = re.compile('({})'.format('|'.join(contraction_mapping.keys())),
                                      flags=re.IGNORECASE | re.DOTALL)

    def expand_match(contraction):
        match = contraction.group(0)
        first_char = match[0]
        expanded_contraction = contraction_mapping.get(match) \
            if contraction_mapping.get(match) \
            else contraction_mapping.get(match.lower())
        expanded_contraction = first_char + expanded_contraction[1:]
        return expanded_contraction

    expanded_text = contractions_pattern.sub(expand_match, text)
    expanded_text = re.sub("'", "", expanded_text)
    return expanded_text


def to_no_stopwords(value):
    try:
        # Tokenize the input string
        word_tokens = word_tokenize(value)
        # Remove stopwords
        filtered_sentence = [w for w in word_tokens if not w.lower() in stop_words]
        # Join the tokens back into a string
        return ' '.join(filtered_sentence)
    except Exception as e:
        # In case of any error, print the error and return the original value
        print(f"Error removing stopwords: {e}")
        return value


# Define a function to stem each answer
def stem_text(text):
    # Tokenize the text to words
    words = word_tokenize(text)
    # Apply stemming to each word
    stemmed_words = [stemmer.stem(word) for word in words]
    # Join the stemmed words back into a single string
    return ' '.join(stemmed_words)


def clean_text(value):
    try:
        # Convert the string to lowercase
        lowercased = value.lower()
        # Remove interpunctuation
        no_punctuation = re.sub(r'[.,;:!?\'\"]', '', lowercased)
        return no_punctuation
    except Exception as e:
        print(e)
        # Return the original value if an error occurs
        return value

# Function to convert NLTK POS tags to WordNet POS tags
def get_wordnet_pos(tag):
    if tag.startswith('J'):
        return wordnet.ADJ
    elif tag.startswith('V'):
        return wordnet.VERB
    elif tag.startswith('N'):
        return wordnet.NOUN
    elif tag.startswith('R'):
        return wordnet.ADV
    else:
        return wordnet.NOUN

# Define the lemmatization function
def lemmatize_text(text):
    # Tokenize the text to words
    words = word_tokenize(text)
    # Get POS tags for the words
    pos_tags = nltk.pos_tag(words)
    # Apply lemmatization to each word with its POS tag
    lemmatized_words = [lemmatizer.lemmatize(word, get_wordnet_pos(tag)) for word, tag in pos_tags]
    # Join the lemmatized words back into a single string
    return ' '.join(lemmatized_words)


def expand_contractions_in_df(df):
    df['answer'] = df['answer'].apply(to_expand_contractions)
    return df


# remove stopwords from answer
def remove_stopwords_from_df(df):
    # Apply the function to remove stopwords from the 'answer' column
    df['answer'] = df['answer'].apply(to_no_stopwords)
    return df


def stem_answers_in_df(df):
    # Apply the stemming function to each answer
    df['answer'] = df['answer'].apply(stem_text)

    return df


def clean_text_in_df(df):
    # Apply the function to the 'answer' column
    df['answer'] = df['answer'].apply(clean_text)
    return df

def lemmatize_answers_in_df(df):
    # Apply the stemming function to each answer
    df['answer'] = df['answer'].apply(lemmatize_text)

    return df


def calculate_levenshtein_distance_matrix(df):
    distance_matrix = {student: {} for student in df['student_id']}
    for i, row_i in df.iterrows():
        for j, row_j in df.iterrows():
            if i == j:
                # If the words are equal the distance is zero
                distance_matrix[row_i['student_id']][row_j['student_id']] = 0
            else:
                # If the words are NOT equal the distance is calculated using Levenshtein.distance
                dist = Levenshtein.distance(str(row_i['answer']), str(row_j['answer']))
                distance_matrix[row_i['student_id']][row_j['student_id']] = dist
    return distance_matrix


def calculate_token_distance_matrix(df):
    distance_matrix = {student: {} for student in df['student_id']}

    # Tokenize each answer
    tokenized_answers = {row['student_id']: word_tokenize(str(row['answer'])) for _, row in df.iterrows()}

    for student_i, tokens_i in tokenized_answers.items():
        for student_j, tokens_j in tokenized_answers.items():
            if student_i == student_j:
                distance_matrix[student_i][student_j] = 0
            else:
                dist = edit_distance(tokens_i, tokens_j)
                distance_matrix[student_i][student_j] = dist

    return distance_matrix


def extended_clustering_options(df, distance_matrix, distance_threshold, filter_negations=False,
                                non_compliance_check=False):
    def merge_clustered_dataframes(*dfs):
        noise_cluster_id = -1
        max_cluster_id = 0

        # Adjust cluster IDs for each dataframe to avoid conflicts
        for df in dfs:
            df['cluster'] = df['cluster'].apply(
                lambda x: x + max_cluster_id + 1 if x != noise_cluster_id else x)
            max_cluster_id = df['cluster'].max()

        # Concatenate all dataframes
        result_df = pd.concat(dfs, ignore_index=True)
        return result_df

    if non_compliance_check:
        top_10_words = get_top_10_bag_of_words(df)
        non_compliant_df = df[~df['answer'].str.contains('|'.join(top_10_words), case=False, na=False)]
        compliant_df = df[df['answer'].str.contains('|'.join(top_10_words), case=False, na=False)]
        non_compliant_df['cluster'] = -2
    else:
        compliant_df = df
        non_compliant_df = pd.DataFrame(columns=df.columns)

    if filter_negations:
        with_negations = compliant_df.loc[compliant_df['answer'].str.contains('not|no', case=False, na=False)]
        without_negations = compliant_df.loc[~compliant_df['answer'].str.contains('not|no', case=False, na=False)]

        with_negations, _ = agglomerative_clustering(with_negations, distance_matrix, distance_threshold, True, None)
        without_negations, _ = agglomerative_clustering(without_negations, distance_matrix, distance_threshold, True,
                                                        None)

        result_df = merge_clustered_dataframes(with_negations, without_negations, non_compliant_df)
    else:
        result_df, _ = agglomerative_clustering(compliant_df, distance_matrix, distance_threshold, True, None)
        if not non_compliant_df.empty:
            result_df = merge_clustered_dataframes(result_df, non_compliant_df)

    return result_df


def agglomerative_clustering(df, distance_matrix, distance_threshold, compact_clusters=False,
                             previous_clusters_info=None):
    students = df['student_id'].tolist()
    distances = np.array([[distance_matrix[s1][s2] for s2 in students] for s1 in students])

    clustering = AgglomerativeClustering(linkage='complete', distance_threshold=distance_threshold, n_clusters=None,
                                         metric='precomputed')
    clusters = clustering.fit_predict(distances)

    # Handling compact clusters by counting the occurrences of each cluster ID
    if compact_clusters:
        cluster_counts = {cluster_id: list(clusters).count(cluster_id) for cluster_id in set(clusters)}
        # Setting cluster ID to -1 for clusters with a size of 1
        clusters = [cluster_id if cluster_counts[cluster_id] > 1 else -1 for cluster_id in clusters]

    cluster_size_and_merges = {}
    if previous_clusters_info:
        student_to_new_cluster_id = {student: clusters[i] for i, student in enumerate(students)}
        old_cluster_id_to_size = {old_cluster_id: info[1] for old_cluster_id, info in previous_clusters_info.items()}

        for old_cluster_id, (median_student_id, _) in previous_clusters_info.items():
            if median_student_id in student_to_new_cluster_id:
                new_cluster_id = student_to_new_cluster_id[median_student_id]
                old_cluster_size = old_cluster_id_to_size[old_cluster_id]

                if new_cluster_id not in cluster_size_and_merges:
                    cluster_size_and_merges[new_cluster_id] = (set([old_cluster_id]), old_cluster_size, old_cluster_id)
                else:
                    existing_ids, existing_size, _ = cluster_size_and_merges[new_cluster_id]
                    existing_ids.add(old_cluster_id)
                    if old_cluster_size > existing_size:
                        cluster_size_and_merges[new_cluster_id] = (existing_ids, old_cluster_size, old_cluster_id)

    new_to_largest_old_cluster_id = {new_id: largest_old_id for new_id, (_, _, largest_old_id) in
                                     cluster_size_and_merges.items()}

    # Update the DataFrame with new cluster IDs or -1 for compacted clusters
    df['cluster'] = [new_to_largest_old_cluster_id[cluster] if cluster in new_to_largest_old_cluster_id else cluster for
                     cluster in clusters]

    merged_clusters = {largest_old_id: list(old_ids) for _, (old_ids, _, largest_old_id) in
                       cluster_size_and_merges.items() if len(old_ids) > 1}

    return df, merged_clusters


def get_top_10_bag_of_words(df):
    # Generate a Bag of Words model on the answers of cluster -1
    vectorizer = CountVectorizer()
    X = vectorizer.fit_transform(df['answer'])

    # Sum up the counts of each word to find the most common ones
    word_counts = X.sum(axis=0)
    word_counts_dict = {word: word_counts[0, idx] for word, idx in vectorizer.vocabulary_.items()}

    # Remove 'no' and 'not' from word_counts_dict
    excluded_words = ['no', 'not']
    for word in excluded_words:
        if word in word_counts_dict:
            del word_counts_dict[word]

    return [word for word, count in Counter(word_counts_dict).most_common(10)]
