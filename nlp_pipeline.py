# organise imports
import re

import pandas as pd
import Levenshtein
from sklearn.cluster import AgglomerativeClustering
import numpy as np
from collections import Counter
import fuzzy
import nltk
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import CountVectorizer
from collections import Counter
from itertools import combinations

# Download the stopwords from NLTK
nltk.download('punkt')
nltk.download('stopwords')

# Initialize the Porter stemmer
stemmer = PorterStemmer()

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
    # Add more contractions as needed
}


def reset_preprocessing(df):
    df['answer'] = df['answer_display']
    return df

def expand_contractions(df):
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
    df['answer'] = df['answer'].apply(to_expand_contractions)
    return df


# remove stopwords from answer
def remove_stopwords(df):
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

    # Apply the function to remove stopwords from the 'answer' column
    df['answer'] = df['answer'].apply(to_no_stopwords)
    return df


def stem_answers(df):
    # Define a function to stem each answer
    def stem_text(text):
        # Tokenize the text to words
        words = word_tokenize(text)
        # Apply stemming to each word
        stemmed_words = [stemmer.stem(word) for word in words]
        # Join the stemmed words back into a single string
        return ' '.join(stemmed_words)

    # Apply the stemming function to each answer
    df['answer'] = df['answer'].apply(stem_text)

    return df


def preprocess(df):
    def to_lowercase(value):
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

    # Apply the function to the 'answer' column
    df['answer'] = df['answer'].apply(to_lowercase)
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


def extended_clustering_options(df, distance_matrix, distance_threshold, filter_negations=False, previous_clusters_info=None):
    if filter_negations:
        with_negations = df.loc[df['answer'].str.contains('not|no', case=False, na=False)]
        without_negations = df.loc[~df['answer'].str.contains('not|no', case=False, na=False)]
        with_negations, stuff = agglomerative_clustering(with_negations, distance_matrix, distance_threshold, True, previous_clusters_info)
        without_negations, stuff = agglomerative_clustering(without_negations, distance_matrix, distance_threshold, True, previous_clusters_info)

        def merge_clustered_dataframes(df_with, df_without):
            # Handling the noise cluster by ensuring it remains -1 in both dataframes
            noise_cluster_id = -1

            # Getting the maximum cluster ID from the 'with_negations' dataframe
            max_cluster_id_with = df_with['cluster'].max()

            # Adjust cluster IDs in the 'without_negations' dataframe
            # Any cluster that isn't the noise cluster gets the offset added to its ID
            df_without['cluster'] = df_without['cluster'].apply(
                lambda x: x + max_cluster_id_with + 1 if x != noise_cluster_id else x)

            # Concatenating the two dataframes
            result_df = pd.concat([df_with, df_without], ignore_index=True)

            return result_df

        # Assuming you already have with_negations and without_negations dataframes from the clustering function
        return merge_clustered_dataframes(with_negations, without_negations)
    else:
        result, stuff = agglomerative_clustering(df, distance_matrix, distance_threshold, True, previous_clusters_info)
        return result


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


def evaluate_clusters(df, distance_matrix):
    def calculate_max_distance_answer_pair(cluster, cluster_distance_matrix):
        max_distance = 0
        answer_pair = ('', '')
        for i in student_ids:
            for j in student_ids:
                current_distance = cluster_distance_matrix[i][j]

                if current_distance > max_distance:

                    def get_answer_by_student_id(cluster, student_id):
                        filtered_df = cluster.loc[cluster['student_id'] == student_id, 'answer']
                        if not filtered_df.empty:
                            return filtered_df.iloc[0]
                        else:
                            return None

                    answer_i = cluster.loc[cluster['student_id'] == i, 'answer_display'].iloc[0]
                    answer_j = cluster.loc[cluster['student_id'] == j, 'answer_display'].iloc[0]

                    answer_pair = (answer_i, answer_j)
                    max_distance = current_distance

        return (max_distance, answer_pair)

    def calculate_median_answer(cluster, cluster_distance_matrix):
        # Calculate the average distance for each point
        keys = list(cluster_distance_matrix.keys())
        student_ids = cluster['student_id'].tolist()
        distances = np.array([[cluster_distance_matrix[s1][s2] for s2 in student_ids] for s1 in student_ids])
        average_distances = np.mean(distances, axis=1)

        # Identify the index of the point with the least average distance
        central_point_index = np.argmin(average_distances)
        central_point_key = keys[central_point_index]

        return cluster.loc[cluster['student_id'] == central_point_key, 'answer_display'].values[0]

    def calculate_median_answer_id(cluster, cluster_distance_matrix):
        # Calculate the average distance for each point
        keys = list(cluster_distance_matrix.keys())
        student_ids = cluster['student_id'].tolist()
        distances = np.array([[cluster_distance_matrix[s1][s2] for s2 in student_ids] for s1 in student_ids])
        average_distances = np.mean(distances, axis=1)

        # Identify the index of the point with the least average distance
        central_point_index = np.argmin(average_distances)
        central_point_key = keys[central_point_index]

        return central_point_key

    columns = ['Cluster_id', 'Size', 'Median Answer', 'Median Answer id', 'Max_error']
    rows = []
    for cluster_label in df['cluster'].unique().tolist():
        cluster = df[df['cluster'] == cluster_label]
        student_ids = cluster['student_id'].tolist()
        cluster_distance_matrix = {
            outer_key: {inner_key: value for inner_key, value in outer_value.items() if inner_key in student_ids}
            for outer_key, outer_value in distance_matrix.items() if outer_key in student_ids
        }
        new_row = {
            'Cluster_id': cluster_label,
            'Size': len(cluster),  # or cluster.shape[0] for the number of rows
            'Median Answer': calculate_median_answer(cluster, cluster_distance_matrix),
            'Median Answer id': calculate_median_answer_id(cluster, cluster_distance_matrix)
        }
        rows.append(new_row)

    df_results = pd.DataFrame(rows, columns=columns)
    return df_results


def calculate_maximum_error(df):
    if len(df) == 0:  # Avoid division by zero
        return 0
    grade_counts = Counter(df['grade'])
    most_common_grade, count = grade_counts.most_common(1)[0]
    return len(df) - count


def evaluate_agglomerative_clustering(question_df, list_of_result_df):
    maxerror = calculate_maximum_error(question_df)
    number_of_answers = len(question_df)
    evaluation_results = []  # List to hold dictionaries of results

    for distance_threshold in range(len(list_of_result_df)):
        result_df = list_of_result_df[distance_threshold]

        # Sum of errors
        sum_of_errors = result_df['Errors'].sum()

        # Sum of errors against maxerror
        sum_of_errors_against_maxerror = sum_of_errors / maxerror

        # Number of clusters
        number_of_clusters = len(result_df)

        # Number of clusters size 1
        # Fixed: Use result_df instead of df
        number_of_clusters_size_1 = (result_df['Size'] == 1).sum()

        # Number of clusters against number of answers
        number_of_clusters_against_number_of_answers = number_of_clusters / number_of_answers

        # Sum of errors + number of clusters
        steps_with_clustering = sum_of_errors + number_of_clusters

        # Sum of errors + number of clusters against number of answers
        steps_with_clustering_against_manual_grading = steps_with_clustering / number_of_answers

        # Create a dictionary for the current result and append it to the list
        evaluation_results.append({
            'Distance Threshold': distance_threshold,
            'Sum of Errors': sum_of_errors,
            'Sum of Errors Against MaxError': sum_of_errors_against_maxerror,
            'Number of Clusters': number_of_clusters,
            'Number of Clusters Size 1': number_of_clusters_size_1,
            'Number of Clusters Against Number of Answers': number_of_clusters_against_number_of_answers,
            'Steps with Clustering': steps_with_clustering,
            'Steps with Clustering Against Manual Grading': steps_with_clustering_against_manual_grading,
        })

    # Convert the list of dictionaries into a DataFrame
    results_df = pd.DataFrame(evaluation_results)
    return results_df


def compare_clustering(attempt1, attempt2):
    if attempt1 == None or attempt2 == None:
        return
    # Reverse the mappings to be entry -> cluster for easy comparison
    entry_to_cluster1 = {entry: cluster_id for cluster_id, entries in attempt1.items() for entry in entries}
    entry_to_cluster2 = {entry: cluster_id for cluster_id, entries in attempt2.items() for entry in entries}

    # Detect merges and splits by comparing entry mappings
    merges = {}
    splits = {}
    for entry, cluster_id1 in entry_to_cluster1.items():
        cluster_id2 = entry_to_cluster2.get(entry)
        if cluster_id1 != cluster_id2:
            if cluster_id1 not in splits:
                splits[cluster_id1] = set()
            splits[cluster_id1].add(cluster_id2)
            if cluster_id2 not in merges:
                merges[cluster_id2] = set()
            merges[cluster_id2].add(cluster_id1)

    # Detect unchanged clusters
    # unchanged = {cluster_id: entries for cluster_id, entries in attempt1.items() if attempt2.get(cluster_id) == entries}

    # Detect new and disappeared clusters
    disappeared_clusters = set(attempt1) - set(attempt2)

    result = {"merges": merges, "splits": splits, "disappeared_clusters": disappeared_clusters}
    print(result)


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


def find_clusters_by_max_edit_distance(df, distance_matrix, max_distance):
    clusters_with_desired_distance = []

    # Generate all possible combinations of two different clusters, excluding clusters -1 and -2
    valid_clusters = df[df['Cluster_id'].apply(lambda x: x not in [-1, -2])]
    for (id1, answer_id1), (id2, answer_id2) in combinations(
            zip(valid_clusters['Cluster_id'], valid_clusters['Median Answer id']), 2):
        # Check if the edit distance between their median answers is less than or equal to max_distance
        if distance_matrix[answer_id1].get(answer_id2, float('inf')) <= max_distance:
            clusters_with_desired_distance.append((int(id1), int(id2)))

    return clusters_with_desired_distance


def find_clusters_with_and_without_not_no(df):
    # Exclude clusters with IDs -1 and -2
    df_filtered = df[~df['cluster'].isin([-1, -2])]

    # Filter the DataFrame for rows where 'answer' contains 'not' or 'no'
    contains_filter = df_filtered['answer'].str.contains('not|no', case=False, na=False)
    not_contains_filter = ~df_filtered['answer'].str.contains('not|no', case=False, na=False)

    # Get the unique cluster IDs from both filters
    clusters_with = df_filtered.loc[contains_filter, 'cluster'].unique().tolist()
    clusters_without = df_filtered.loc[not_contains_filter, 'cluster'].unique().tolist()

    # Convert numpy.int64 to int
    clusters_with = [int(cluster) for cluster in clusters_with]
    clusters_without = [int(cluster) for cluster in clusters_without]

    # Find the intersection of clusters containing both conditions
    common_clusters = set(clusters_with).intersection(clusters_without)

    return list(common_clusters)
