from datetime import datetime

import pandas as pd
import nlp_pipeline
import datastructure

session = None
image = None


def get_session():
    global session
    return session


def preprocess(expand_contractions, remove_stopwords, prepro_method):
    global session
    session.student_answers = nlp_pipeline.reset_preprocessing(session.student_answers)
    session.student_answers = nlp_pipeline.clean_text_in_df(session.student_answers)
    if expand_contractions:
        session.student_answers = nlp_pipeline.expand_contractions_in_df(session.student_answers)
    if remove_stopwords:
        session.student_answers = nlp_pipeline.remove_stopwords_from_df(session.student_answers)
    if prepro_method == "Lemmatization":
        session.student_answers = nlp_pipeline.lemmatize_answers_in_df(session.student_answers)
    if prepro_method == "Stemming":
        session.student_answers = nlp_pipeline.stem_answers_in_df(session.student_answers)


def cluster(filter_negations, token_based_clustering, non_compliance, distance_threshold):
    global session
    if token_based_clustering:
        session.distance_matrix = nlp_pipeline.calculate_token_distance_matrix(session.student_answers)
    else:
        session.distance_matrix = nlp_pipeline.calculate_levenshtein_distance_matrix(session.student_answers)
    session.student_answers = nlp_pipeline.extended_clustering_options(session.student_answers,
                                                                       session.distance_matrix,
                                                                       distance_threshold,
                                                                       filter_negations,
                                                                       non_compliance)


def create_session(participant_id, question):
    global session
    global image
    session = datastructure.create_session(participant_id, question)
    #image = f"circuit_images/circuit_{question_number}.jpg"
    image = f"circuit_images/circuit_3.jpg"


def get_progress():
    if 'grade' in session.student_answers.columns:
        total_count = len(session.student_answers['grade']) + len(session.student_grades)
        complete = len(session.student_grades)
        return complete / total_count
    else:
        return 0  # Return 0 if there is no 'grade' column


def set_grade_for_cluster(cluster_value, new_grade):
    global session
    session.student_answers.loc[session.student_answers['cluster'] == cluster_value, 'grade'] = new_grade
    selected_rows = session.student_answers[session.student_answers['cluster'] == cluster_value]
    session.student_grades = pd.concat([session.student_grades, selected_rows], ignore_index=True)
    session.student_answers = session.student_answers.drop(selected_rows.index).reset_index(drop=True)


def set_grade_for_student(student_id, new_grade):
    global session
    session.student_answers.loc[session.student_answers['student_id'] == student_id, 'grade'] = new_grade
    selected_rows = session.student_answers[session.student_answers['student_id'] == student_id]
    session.student_grades = pd.concat([session.student_grades, selected_rows], ignore_index=True)
    session.student_answers = session.student_answers.drop(selected_rows.index).reset_index(drop=True)


def remove_student_from_cluster(student_id):
    if 'student_id' in session.student_answers.columns and 'cluster' in session.student_answers.columns:
        session.student_answers.loc[session.student_answers['student_id'] == student_id, 'cluster'] = -1


def get_top_10_words():
    global session
    if len(session.student_answers) != 0:
        return nlp_pipeline.get_top_10_bag_of_words(session.student_answers)
    else:
        return []


def log_button(button, parameter=""):
    global session
    session.button_log.loc[len(session.button_log.index)] = [button, parameter, datetime.now()]


def get_cluster_header(cluster):
    global session
    clusterdf = session.student_answers[session.student_answers['cluster'] == cluster]
    top_ten = nlp_pipeline.get_top_10_bag_of_words(clusterdf)
    first_two_entries = top_ten[:2]
    key_words = " ".join(first_two_entries)
    size = len(clusterdf)
    return f"{key_words}: Size ({size})"

def revoke_grade_of_student(student_id):
    global session
    session.student_grades.loc[session.student_grades['cluster'] == student_id, 'grade'] = -1
    selected_rows = session.student_grades[session.student_grades['student_id'] == student_id]
    session.student_answers = pd.concat([session.student_answers, selected_rows], ignore_index=True)
    session.student_grades = session.student_grades.drop(selected_rows.index).reset_index(drop=True)


def get_biggest_cluster_id():
    cluster_counts = session.student_answers["cluster"].value_counts()
    if len(cluster_counts) > 1:
        cluster_counts_no_minus_one = cluster_counts[cluster_counts.index != -1]
        if not cluster_counts_no_minus_one.empty:
            return cluster_counts_no_minus_one.idxmax()

    return -1