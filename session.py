import pandas as pd
import nlp_pipeline
import datastructure

session = None
image = None


def get_session():
    global session
    return session


def cluster(expand_contractions, remove_stopwords, stem_answers, filter_negations, token_based_clustering, non_compliance, distance_threshold):
    global session
    session.student_answers = nlp_pipeline.reset_preprocessing(session.student_answers)
    student_answers = nlp_pipeline.clean_text_in_df(session.student_answers)
    if expand_contractions:
        session.student_answers = nlp_pipeline.expand_contractions_in_df(session.student_answers)
    if remove_stopwords:
        session.student_answers = nlp_pipeline.remove_stopwords_from_df(session.student_answers)
    if stem_answers:
        session.student_answers = nlp_pipeline.stem_answers_in_df(session.student_answers)
    session.distance_matrix = nlp_pipeline.calculate_levenshtein_distance_matrix(session.student_answers)
    previous_cluster_info = session.previous_cluster_info
    session.student_answers, stuff = nlp_pipeline.agglomerative_clustering(student_answers,
                                                                           session.distance_matrix,
                                                                           distance_threshold,
                                                                           True,
                                                                           previous_cluster_info)


def create_session(question_number):
    global session
    global image
    session = datastructure.create_session(question_number)
    image = f"circuit_images/circuit_{question_number}.jpg"


def get_progress():
    if 'grade' in session.student_answers.columns:
        total_count = len(session.student_answers['grade']) + len(session.student_grades)
        complete = len(session.student_grades)
        return complete / total_count
    else:
        return 0  # Return 0 if there is no 'grade' column


def set_grade_for_cluster(cluster_value, new_grade):
    global session
    selected_rows = session.student_answers[session.student_answers['cluster'] == cluster_value]
    session.student_answers.loc[session.student_answers['cluster'] == cluster_value, 'grade'] = new_grade
    session.student_grades = pd.concat([session.student_grades, selected_rows], ignore_index=True)
    session.student_answers = session.student_answers.drop(selected_rows.index).reset_index(drop=True)


def set_grade_for_student(cluster_value, new_grade):
    global session
    selected_rows = session.student_answers[session.student_answers['student_id'] == cluster_value]
    session.student_answers.loc[session.student_answers['cluster'] == cluster_value, 'grade'] = new_grade
    session.student_grades = pd.concat([session.student_grades, selected_rows], ignore_index=True)
    session.student_answers = session.student_answers.drop(selected_rows.index).reset_index(drop=True)


def remove_student_from_cluster(student_id):
    if 'student_id' in session.student_answers.columns and 'cluster' in session.student_answers.columns:
        session.student_answers.loc[session.student_answers['student_id'] == student_id, 'cluster'] = -1
