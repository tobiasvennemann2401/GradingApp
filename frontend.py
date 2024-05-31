import numpy as np
import streamlit as st

import nlp_pipeline
import session

if 'initialized' not in st.session_state:
    session.create_session(3)
    #session.cluster(True, True, True, False, False, False,4)
    st.session_state['initialized'] = True
    st.session_state['selected_cluster_index'] = 0

if 'selected_cluster' not in st.session_state:
    st.session_state['selected_cluster'] = -1

# Get unique clusters
clusters = session.get_session().student_answers['cluster'].unique()

with st.sidebar.expander("Preprocessing"):
    expand_contractions = st.checkbox('Expand Contractions', help='This checkbox enables the changes from: don´t -> do not')
    remove_stopwords = st.checkbox('Remove Stopwords', help='This checkbox removes certain words as: do, and, is, if')
    preproc_method = st.radio(
        "Word Normalization Method",
        ["Nothing", "Lemmatization", "Stemming"],
        captions=["Words arent changed", "Words are transformed into there dictionary form", "Words are cut after the word stem"])
    st.text("\n\n")
    st.text("Example Sentence:")
    with st.container(border=20):
        sentence = "This sentence is an example of a sentence but hasn't got any other function"
        st.write(sentence)
        st.text("\n\n")
        sentence = nlp_pipeline.clean_text(sentence)
        if expand_contractions:
            sentence = nlp_pipeline.to_expand_contractions(sentence)
        if remove_stopwords:
            sentence = nlp_pipeline.to_no_stopwords(sentence)
        if preproc_method == "Stemming":
            sentence = nlp_pipeline.stem_text(sentence)
        if preproc_method == "Lemmatization":
            sentence = nlp_pipeline.lemmatize_text(sentence)
        st.text("After Preprocessing")
        st.write(sentence)

with st.sidebar.expander("Non Compliance Check"):
    non_compliance = st.checkbox('Create Non Compliance Cluster')
    st.write("Answers that do not contain any of the top 10 most used words are considered non compliant and put in a separate cluster.")
    st.text("Top 10 words")
    st.table(session.get_top_10_words())
with st.sidebar.expander("Clustering"):
    distance_calculation_method = st.radio(
        "Distance Calculation Method",
        ["Token Based", "Character Based"],
        captions=["Distance is calculated based on how many words are different", "Distance is calculated based on how many characters are different"])
    distance_threshold = st.number_input('Distance Threshold', min_value=0, max_value=10, step=1, help='This value determines the maximum distance two answers in one cluster can be apart')
    filter_negations = st.checkbox('Filter Negations', help='Prevents sentences with opposite meanings (e.g., "I like it" vs. "I dont like it") from being grouped together.')

if 'update' not in st.session_state:
    st.session_state['update'] = True  # Initialize state

if st.sidebar.button('Cluster'):
    session.cluster(expand_contractions, remove_stopwords, preproc_method, filter_negations, distance_calculation_method == "Token Based", non_compliance, distance_threshold)
    st.session_state['update'] = True  # Update state on clustering
    st.session_state['selected_cluster'] = -1

col1, col2 = st.columns(2)

with col1:
    st.image(session.image)
    st.text("Question:")
    st.write(session.get_session().question_text)
    st.text("\n\n")
    st.text("Reference Answer:")
    st.write(session.get_session().reference_answer)
    st.text("\n\n")

    st.progress(session.get_progress(), text=f"Progress: {round(session.get_progress()*100, 2)}%")
with col2:
    if 'update' in st.session_state and st.session_state['update']:
        clusters = session.get_session().student_answers['cluster'].unique()
        st.session_state['update'] = False

    cluster_choice = st.selectbox("Choose a Cluster", options=sorted(clusters),
                                  index=st.session_state['selected_cluster_index'],
                                  key='selected_cluster',
                                  format_func=lambda x: "Unclustered" if x == -1 else "Non Compliance" if x == -2 else f"Cluster {x}")
    show_preprocess = st.checkbox('Show Preprocessing Result')

    if cluster_choice is not None:
        filtered_data = session.get_session().student_answers[
            (session.get_session().student_answers['cluster'] == cluster_choice) & (
                        session.get_session().student_answers['grade'] == -1)]
        filtered_data = filtered_data.drop(['grade', 'cluster', 'time_delta'], axis=1)

        if show_preprocess:
            filtered_data = filtered_data.drop(['answer_display'], axis=1)
        else:
            filtered_data = filtered_data.drop(['answer'], axis=1)

        for index, row in filtered_data.iterrows():
            if cluster_choice != -1:
                cols = st.columns([4, 1])
                cols[0].write(row.values[1])
                if cols[1].button("🗑️", key=f"{index}_btn_{index}", help="Remove item from cluster"):
                    session.remove_student_from_cluster(row.values[0])
                    st.session_state['update'] = True
                    st.experimental_rerun()
            else:
                cols = st.columns([3, 1, 1])
                cols[0].write(row.values[1])
                if cols[1].button("✔️", key=f"{index}_btn_{index}_1"):
                    session.set_grade_for_student(row.values[0], 1)
                    st.session_state['update'] = True
                    st.experimental_rerun()
                if cols[2].button("❌", key=f"{index}_btn_{index}_2"):
                    session.set_grade_for_student(row.values[0], 0)
                    st.session_state['update'] = True
                    st.experimental_rerun()
        if cluster_choice != -1:
            bot_cols = st.columns([1, 1])
            if bot_cols[0].button('Complete Cluster ✔️', disabled=st.session_state['selected_cluster'] == -1,
                                  key=f"{index}_btun_{index}"):
                session.set_grade_for_cluster(st.session_state['selected_cluster'], 1)
                st.session_state['update'] = True
                st.session_state['selected_cluster_index'] = 0
                st.experimental_rerun()
            if bot_cols[1].button('Complete Cluster ❌', disabled=st.session_state['selected_cluster'] == -1,
                                  key=f"{index}_btun_2_{index}"):
                session.set_grade_for_cluster(st.session_state['selected_cluster'], 0)
                st.session_state['update'] = True
                st.session_state['selected_cluster_index'] = 0
                st.experimental_rerun()
