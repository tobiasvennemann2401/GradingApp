import numpy as np
from matplotlib import pyplot as plt
from sklearn.manifold import MDS

import nlp_pipeline
import datastructure


session=None

def get_session():
    global session
    return session

def cluster(expand_contractions, remove_stopwords, stem_answers, distance_threshold):
    global session
    if expand_contractions:
        session.student_answers = nlp_pipeline.expand_contractions(session.student_answers)
    if remove_stopwords:
        session.student_answers = nlp_pipeline.remove_stopwords(session.student_answers)
    student_answers = nlp_pipeline.preprocess(session.student_answers)
    if stem_answers:
        session.student_answers = nlp_pipeline.stem_answers(session.student_answers)
    session.distance_matrix = nlp_pipeline.calculate_levenshtein_distance_matrix(session.student_answers)
    previous_cluster_info = session.previous_cluster_info
    session.student_answers, stuff = nlp_pipeline.agglomerative_clustering(student_answers,
                                                                           session.distance_matrix,
                                                                           distance_threshold, True,
                                                                           previous_cluster_info)


def preprocess():
    global session
    student_answers = session.student_answers
    student_answers = nlp_pipeline.expand_contractions(student_answers)
    student_answers = nlp_pipeline.remove_stopwords(student_answers)
    student_answers = nlp_pipeline.preprocess(student_answers)
    student_answers = nlp_pipeline.stem_answers(student_answers)
    session.distance_matrix = nlp_pipeline.calculate_levenshtein_distance_matrix(student_answers)

def create_session():
    global session
    session = datastructure.create_session(15)

def get_progress():
    if 'grade' in session.student_answers.columns:
        total_count = len(session.student_answers['grade'])
        if total_count == 0:
            return 0  # Avoid division by zero if the column is empty

        # Count non -1 values
        non_negative_one_count = session.student_answers['grade'][session.student_answers['grade'] != -1].count()

        return non_negative_one_count / total_count
    else:
        return 0  # Return 0 if there is no 'grade' column

def set_grade_for_cluster(cluster_value, new_grade):
    if 'cluster' in session.student_answers.columns and 'grade' in session.student_answers.columns:
        session.student_answers.loc[session.student_answers['cluster'] == cluster_value, 'grade'] = new_grade
def set_grade_for_student(cluster_value, new_grade):
    if 'student_id' in session.student_answers.columns and 'grade' in session.student_answers.columns:
        session.student_answers.loc[session.student_answers['student_id'] == cluster_value, 'grade'] = new_grade

def remove_student_from_cluster(student_id):
    if 'student_id' in session.student_answers.columns and 'cluster' in session.student_answers.columns:
        session.student_answers.loc[session.student_answers['student_id'] == student_id, 'cluster'] = -1

def plot_student_clusters():
    global session
    # Convert the distance dictionary to a square distance matrix
    students = session.student_answers['student_id'].tolist()
    distance_matrix = np.array([[session.distance_matrix[s1][s2] for s2 in students] for s1 in students])

    # Apply MDS to reduce dimensions to 2D
    mds = MDS(n_components=2, dissimilarity='precomputed', random_state=42)
    coordinates = mds.fit_transform(distance_matrix)

    # Create a scatter plot colored by cluster
    plt.figure(figsize=(10, 8))
    clusters = session.student_answers['cluster'].unique()
    colors = plt.cm.get_cmap('tab10', len(clusters))  # Colormap with different colors for each cluster

    for i, cluster in enumerate(clusters):
        idx = session.student_answers['cluster'] == cluster
        plt.scatter(coordinates[idx, 0], coordinates[idx, 1], color=colors(i), label=f'Cluster {cluster}')

    plt.title('2D Scatterplot of Answers by Cluster')
    plt.legend(title='Cluster')
    plt.grid(True)
    return plt
