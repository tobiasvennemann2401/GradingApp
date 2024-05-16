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
    print(session.student_answers["cluster"])


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

    plt.title('2D Scatterplot of Students by Cluster')
    plt.xlabel('MDS1')
    plt.ylabel('MDS2')
    plt.legend(title='Cluster')
    plt.grid(True)
    return plt
