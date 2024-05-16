from flask import Flask, jsonify, request, abort
from datetime import datetime
from flask_cors import CORS
import pandas as pd

import nlp_pipeline
import plots
import datastructure
from dateutil import parser
from pandas import to_timedelta
from sklearn.feature_extraction.text import CountVectorizer
from collections import Counter

app = Flask(__name__)
sessions = {}

# Disable CORS for all routes
CORS(app, resources={r"/*": {"origins": "*"}})


# Define Flask endpoints
@app.route('/session', methods=['Post'])
def create_session():
    global sessions
    condition = request.json.get('condition')
    question_number = request.json.get('question_number')
    session = create_session_object(question_number, condition)
    return jsonify({'session_id': session.id})


@app.route('/session/<string:session_id>', methods=['Get'])
def get_session(session_id):
    global sessions
    return jsonify({'session': sessions[session_id].to_json_serializable_dict()})


@app.route('/session/<string:session_id>', methods=['POST'])
def update_session(session_id):
    global sessions
    # Check if the session exists
    if session_id not in sessions:
        return abort(404, description="Session not found")

    # Get the JSON data sent with the request
    data = request.json

    # Retrieve the session object
    session = sessions[session_id]

    # Update session attributes from the data, checking for each key
    if 'study_condition' in data:
        session.study_condition = data['study_condition']
    if 'question_text' in data:
        session.question_text = data['question_text']
    if 'reference_answer' in data:
        session.reference_answer = data['reference_answer']
    if 'start_time' in data and data['start_time']:
        session.start_time = datetime.fromisoformat(data['start_time'])
    if 'end_time' in data and data['end_time']:
        session.end_time = datetime.fromisoformat(data['end_time'])

    # For updating 'student_answers', ensure it's provided and is a list
    if 'student_answers' in data and isinstance(data['student_answers'], list):
        # Convert the list of dictionaries to a DataFrame
        session.student_answers = pd.DataFrame(data['student_answers'])

    return jsonify({"message": "Session updated successfully"}), 200


@app.route('/session', methods=['Get'])
def get_sessions():
    global sessions
    return jsonify({'session': list(sessions.keys())})


@app.route('/session/<session_id>/setStartTime', methods=['POST'])
def set_start_time(session_id):
    # Ensure the session exists
    if session_id not in sessions:
        return jsonify({"error": "Session not found"}), 404

    # Get the start time from the request body
    data = request.get_json()
    start_time_str = data.get('start_time', None)

    if start_time_str:
        try:
            # Convert the start time to a datetime object
            start_time = parser.parse(start_time_str)
        except ValueError:
            # Handle the case where the date format is incorrect
            return jsonify({"error": "Invalid start_time format"}), 400

        # Set the start time for the session
        sessions[session_id].start_time = start_time
        return jsonify({"message": "Start time set successfully"}), 200
    else:
        # If no start time was provided in the request
        return jsonify({"error": "No start_time provided"}), 400


@app.route('/session/<session_id>/setEndTime', methods=['POST'])
def set_end_time(session_id):
    # Ensure the session exists
    if session_id not in sessions:
        return jsonify({"error": "Session not found"}), 404

    # Get the start time from the request body
    data = request.get_json()
    end_time_str = data.get('end_time', None)

    if end_time_str:
        try:
            # Convert the start time to a datetime object
            end_time = parser.parse(end_time_str)
        except ValueError:
            # Handle the case where the date format is incorrect
            return jsonify({"error": "Invalid start_time format"}), 400

        # Set the start time for the session
        sessions[session_id].end_time = end_time
        return jsonify({"message": "Start time set successfully"}), 200
    else:
        # If no start time was provided in the request
        return jsonify({"error": "No start_time provided"}), 400


@app.route('/session/<session_id>/student/<int:student_id>/setGrade', methods=['POST'])
def set_grade(session_id, student_id):
    # Parse the incoming JSON data for the grade
    data = request.json
    grade = data.get('grade')

    # Check if the session ID exists in the sessions dictionary
    if session_id in sessions:
        session = sessions[session_id]
        # Directly access the student_answers DataFrame
        df = session.student_answers
        # Check if the student ID exists in the DataFrame
        if student_id in df['student_id'].values:
            # Update the grade for the student ID
            df.loc[df['student_id'] == student_id, 'grade'] = grade
            return jsonify({'message': 'Grade updated successfully'}), 200
        else:
            return jsonify({'error': 'Student ID not found in session'}), 404
    else:
        # Session ID not found
        return jsonify({'error': 'Session not found'}), 404


@app.route('/session/<session_id>/student/<int:student_id>/addTimeDelta', methods=['POST'])
def add_time_delta(session_id, student_id):
    # Extract the time delta from the request body as a float of seconds
    data = request.json
    time_delta_seconds = data.get('time_delta')  # This is now expected to be a float representing seconds

    # Convert the seconds to a pandas Timedelta object
    time_delta = to_timedelta(time_delta_seconds, unit='s')

    # Check if the session and student exist
    if session_id in sessions:
        session = sessions[session_id]
        df = session.student_answers
        if student_id in df['student_id'].values:
            # Update the time delta for the student
            # If the 'time_delta' column does not exist, it initializes with 0 duration Timedelta
            if 'time_delta' not in df.columns:
                df['time_delta'] = to_timedelta(0, unit='s')

            # Locate the specific student and add the new time delta to the existing one
            df.loc[df['student_id'] == student_id, 'time_delta'] += time_delta

            return jsonify({'message': 'Time delta updated successfully'}), 200
        else:
            return jsonify({'error': 'Student ID not found in session'}), 404
    else:
        return jsonify({'error': 'Session not found'}), 404


@app.route('/session/<session_id>/scatterPlot', methods=['GET'])
def get_scatter_plot(session_id):
    # Assuming the client sends the cluster as a query parameter instead of JSON body because it's a GET request
    clusterA = request.args.get('clusterA', None)  # Get 'cluster' from query parameters, default to None if not present
    clusterB = request.args.get('clusterB', None)  # Get 'cluster' from query parameters, default to None if not present
    if session_id in sessions:
        df = sessions[session_id].student_answers
        distance_matrix = sessions[session_id].distance_matrix
        # Pass the cluster to the function, which can be None
        scatter_plot_base64 = plots.get_base64_scatter_plot(df, distance_matrix, clusterA, clusterB)
        return jsonify({'scatterplot': f"data:image/png;base64,{scatter_plot_base64}"})
    else:
        return jsonify({'error': 'Session not found'}), 404


@app.route('/session/<session_id>/pieChart', methods=['GET'])
def pie_chart(session_id):
    # Assuming the client sends the cluster as a query parameter instead of JSON body because it's a GET request
    if session_id in sessions:
        previous_cluster_info = sessions[session_id].previous_cluster_info
        scatter_plot_base64 = plots.get_pie_chart(previous_cluster_info)
        return jsonify({'scatterplot': f"data:image/png;base64,{scatter_plot_base64}"})
    else:
        return jsonify({'error': 'Session not found'}), 404


@app.route('/session/<session_id>/progressPieChart', methods=['GET'])
def progress_pie_chart(session_id):
    # Assuming the client sends the cluster as a query parameter instead of JSON body because it's a GET request
    if session_id in sessions:
        scatter_plot_base64 = plots.get_progress_pie_chart(sessions[session_id].student_answers)
        return jsonify({'scatterplot': f"data:image/png;base64,{scatter_plot_base64}"})
    else:
        return jsonify({'error': 'Session not found'}), 404


@app.route('/session/<session_id>/cluster', methods=['POST'])
def cluster(session_id):
    global sessions
    data = request.get_json()
    distance_threshold = float(data.get('distance_threshold'))
    student_answers = sessions[session_id].student_answers
    distance_matrix = sessions[session_id].distance_matrix
    previous_cluster_info = sessions[session_id].previous_cluster_info
    sessions[session_id].student_answers, stuff = nlp_pipeline.agglomerative_clustering(student_answers,
                                                                                        distance_matrix,
                                                                                        distance_threshold, True,
                                                                                        previous_cluster_info)
    # Assuming 'df' is your DataFrame
    # Filter the DataFrame to get only cluster -1
    cluster_minus_1_df = sessions[session_id].student_answers[sessions[session_id].student_answers['cluster'] < 0]

    most_common_words = nlp_pipeline.get_top_10_bag_of_words(sessions[session_id].student_answers)
    # Function to check if any of the most common words are in the answer
    def check_compliance(answer):
        return any(word in answer for word in most_common_words)

    # Apply the compliance check and update the cluster_id for non-compliant entries
    cluster_minus_1_df['compliance'] = cluster_minus_1_df['answer'].apply(check_compliance)
    sessions[session_id].student_answers.loc[sessions[session_id].student_answers['cluster'] == -1, 'cluster'] = cluster_minus_1_df.apply(
        lambda row: -2 if not row['compliance'] else -1, axis=1)

    sessions[session_id].cluster_evaluation = nlp_pipeline.evaluate_clusters(sessions[session_id].student_answers,
                                                                             sessions[session_id].distance_matrix)

    sessions[session_id].previous_cluster_info = sessions[session_id].cluster_evaluation.set_index('Cluster_id').apply(
        lambda row: (row['Median Answer id'], row['Size']), axis=1).to_dict()

    nonot = nlp_pipeline.find_clusters_with_and_without_not_no(sessions[session_id].student_answers)
    merge = nlp_pipeline.find_clusters_by_max_edit_distance(sessions[session_id].cluster_evaluation, sessions[session_id].distance_matrix, distance_threshold+1)
    response = {
        "nonot": nonot,
        "merge": merge
    }
    return response


@app.route('/session/<session_id>/cluster/simple', methods=['POST'])
def simple_cluster(session_id):
    try:
        # Retrieve JSON data from request
        data = request.get_json()
        distance_threshold = float(data.get('distance_threshold'))
        centroid_id = int(data.get('centroid_id'))
        ignored_ids = set(data.get('ignored_ids', []))  # Retrieve ignored student IDs; default to empty list

        # Retrieve student answers and distance matrix from session
        student_answers = sessions[session_id].student_answers
        distance_matrix = sessions[session_id].distance_matrix

        # Filter answers based on distance to centroid and not in ignored list
        close_students = []
        for index, row in student_answers.iterrows():
            student_id = row['student_id']
            if student_id not in ignored_ids:  # Check if student_id is not in the ignored list
                if student_id in distance_matrix[centroid_id] and distance_matrix[centroid_id][student_id] <= distance_threshold:
                    close_students.append(row.to_dict())

        # Return the list of student answers that are within the distance threshold
        return jsonify(close_students)

    except Exception as e:
        return str(e), 400


@app.route('/session/<session_id>/getTop10Words', methods=['Get'])
def get_top_10_words(session_id):
    global sessions
    return jsonify({'words': list(nlp_pipeline.get_top_10_bag_of_words(sessions[session_id].student_answers))})


@app.route('/session/<session_id>/preProcess', methods=['GET'])
def preprocess(session_id):
    student_answers = sessions[session_id].student_answers
    student_answers = nlp_pipeline.expand_contractions(student_answers)
    student_answers = nlp_pipeline.remove_stopwords(student_answers)
    student_answers = nlp_pipeline.preprocess(student_answers)
    student_answers = nlp_pipeline.stem_answers(student_answers)
    sessions[session_id].distance_matrix = nlp_pipeline.calculate_levenshtein_distance_matrix(student_answers)
    return "200"


@app.route('/session/<session_id>/preProcessWithCluster', methods=['POST'])
def preprocessWithCluster(session_id):
    global sessions
    data = request.get_json()
    preprocess = bool(data.get('preprocess'))
    expand_contractions = bool(data.get('expand_contractions'))
    filter_negations = bool(data.get('filter_negations'))
    remove_stopwords = bool(data.get('remove_stopwords'))
    stemming = bool(data.get('stemming'))
    distance_threshold = int(data.get('distance_threshold'))
    student_answers = sessions[session_id].student_answers
    student_answers = nlp_pipeline.reset_preprocessing(student_answers)
    if preprocess:
        student_answers = nlp_pipeline.preprocess(student_answers)
    if expand_contractions:
        student_answers = nlp_pipeline.expand_contractions(student_answers)
    if remove_stopwords:
        student_answers = nlp_pipeline.remove_stopwords(student_answers)
    if stemming:
        student_answers = nlp_pipeline.stem_answers(student_answers)
    sessions[session_id].distance_matrix = nlp_pipeline.calculate_levenshtein_distance_matrix(student_answers)
    sessions[session_id].student_answers = nlp_pipeline.extended_clustering_options(student_answers, sessions[session_id].distance_matrix, distance_threshold, filter_negations=filter_negations, previous_clusters_info=sessions[session_id].previous_cluster_info)
    sessions[session_id].previous_cluster_info = sessions[session_id].cluster_evaluation.set_index('Cluster_id').apply(lambda row: (row['Median Answer id'], row['Size']), axis=1).to_dict()
    sessions[session_id].cluster_evaluation = nlp_pipeline.evaluate_clusters(sessions[session_id].student_answers, sessions[session_id].distance_matrix)
    return "200"


@app.route('/session/<session_id>/cluster/<cluster_id>', methods=['GET'])
def get_cluster(session_id, cluster_id):
    global sessions
    # Check if the session exists
    if session_id not in sessions:
        return abort(404, description="Session not found")

    # Retrieve the session object
    session = sessions[session_id]

    # Ensure the session has a 'cluster_evaluation' attribute that's a DataFrame
    if not hasattr(session, 'cluster_evaluation') or not isinstance(session.cluster_evaluation, pd.DataFrame):
        return abort(404, description="Cluster evaluation data not found")

    # Attempt to find the row in 'cluster_evaluation' with the specified 'cluster_id'
    cluster_data = session.cluster_evaluation[session.cluster_evaluation['Cluster_id'] == int(cluster_id)]

    # If no row is found, return an error
    if cluster_data.empty:
        return abort(404, description="Cluster ID not found")

    # Convert the row to a JSON-serializable format and return it
    # Assuming you want to return the first match if there are multiple
    cluster_data_dict = cluster_data.iloc[0].to_dict()
    return jsonify(cluster_data_dict), 200


def create_initial_session_object():
    global sessions
    #session = datastructure.create_session(15)
    session = datastructure.create_session(15)
    print(len(session.student_answers))
    session.id = "8dbbd01a-62db-43bc-ad96-d7308d2486d0"
    sessions[session.id] = session


def create_session_object(question_number, condition):
    global sessions
    session = datastructure.create_session(question_number)
    session.study_condition = condition
    sessions[session.id] = session
    return session


if __name__ == '__main__':
    debug = True
    if debug:
        create_initial_session_object()
    app.run(debug=True)
