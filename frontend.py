import numpy as np
import pandas as pd
import streamlit as st
import session

if 'initialized' not in st.session_state:
    session.create_session(3)
    session.cluster(True, True, True, 4)
    st.session_state['initialized'] = True
    st.session_state['selected_cluster_index'] = 0

if 'selected_cluster' not in st.session_state:
    st.session_state['selected_cluster'] = -1

# Get unique clusters
clusters = session.get_session().student_answers['cluster'].unique()

with st.sidebar.expander("Preprocessing"):
    expand_contractions = st.checkbox('Expand Contractions', help='This checkbox enables the changes from: don´t -> do not')
    stemming = st.checkbox('Stemming', help='This checkbox enables stemming, which shortens words: walking -> walk')
    remove_stopwords = st.checkbox('Remove Stopwords', help='This checkbox removes certain words as: do, and, is, if')
    st.text("\n\n")
    st.write("This sentence is an example of a sentence but hasn't got any other function")
    st.text("\n\n")
    st.write("Preprocessed sentence")

with st.sidebar.expander("Clustering"):
    filter_negations = st.checkbox('Filter Negations', help='This checkbox ensures that no answers with and without negations are in the same cluster')
    token_based_clustering = st.checkbox('Token Based Clustering', help='The edit distance is based on tokens instead of characters')
    distance_threshold = st.number_input('Distance Threshold', min_value=0, max_value=10, step=1, help='This value determines the maximum distance two answers in one cluster can be apart')

if 'update' not in st.session_state:
    st.session_state['update'] = True  # Initialize state

if st.sidebar.button('Cluster'):
    session.cluster(expand_contractions, remove_stopwords, stemming, distance_threshold)
    st.session_state['update'] = True  # Update state on clustering
    st.session_state['selected_cluster'] = -1

col1, col2 = st.columns(2)

with col1:
    st.image(session.image)
    st.text(session.get_session().question_text)
    st.text("\n\n")
    st.text(session.get_session().reference_answer)
    st.text("\n\n")
    if st.button('All Answers In Cluster As Correct', disabled=st.session_state['selected_cluster'] == -1):
        session.set_grade_for_cluster(st.session_state['selected_cluster'], 1)
        st.session_state['update'] = True
        st.session_state['selected_cluster_index'] = 0
        st.experimental_rerun()
    if st.button('All Answers In Cluster As False', disabled=st.session_state['selected_cluster'] == -1):
        session.set_grade_for_cluster(st.session_state['selected_cluster'], 0)
        st.session_state['update'] = True
        st.session_state['selected_cluster_index'] = 0
        st.experimental_rerun()
    st.progress(session.get_progress(), text="Progress")
with col2:
    if 'update' in st.session_state and st.session_state['update']:
        clusters = session.get_session().student_answers['cluster'].unique()
        st.session_state['update'] = False

    cluster_choice = st.selectbox("Choose a Cluster", options=sorted(clusters),
                                  index=st.session_state['selected_cluster_index'],
                                  key='selected_cluster',
                                  format_func=lambda x: "Unclustered" if x == -1 else f"Cluster {x}")
    show_preprocess = st.checkbox('Show Preprocessed Data')

    if cluster_choice is not None:
        print(np.where(clusters == cluster_choice))
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
                if cols[1].button("🗑️", key=f"{index}_btn_{index}"):
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
