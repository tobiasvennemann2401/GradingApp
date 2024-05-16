import pandas as pd
import streamlit as st
import plots
import session



if 'initialized' not in st.session_state:
    session.create_session()
    st.session_state['initialized'] = True

# Get unique clusters
clusters = session.get_session().student_answers['cluster'].unique()

expand_contractions = st.sidebar.checkbox('Expand Contractions')
filter_negations = st.sidebar.checkbox('Filter Negations')
stemming = st.sidebar.checkbox('Stemming')
remove_stopwords = st.sidebar.checkbox('Remove Stopwords')
distance_threshold = st.sidebar.number_input('Distance Threshold', min_value=0, max_value=10, step=1)

if 'update' not in st.session_state:
    st.session_state['update'] = False  # Initialize state

if st.sidebar.button('Cluster'):
    session.cluster(expand_contractions, remove_stopwords, stemming, distance_threshold)
    cluster_choice = 1
    st.session_state['update'] = True  # Update state on clustering


col1, col2 = st.columns(2)

with col1:
    st.pyplot(session.plot_student_clusters())
with col2:
    if st.session_state['update']:
        clusters = session.get_session().student_answers['cluster'].unique()  # Refresh cluster data
        st.session_state['update'] = False  # Reset the update indicator
    cluster_choice = st.selectbox("Choose a Cluster", options=sorted(clusters))
    show_preprocess = st.checkbox('Show Preprocessed Data')
    if cluster_choice is not None:
        filtered_data = session.get_session().student_answers[
            session.get_session().student_answers['cluster'] == cluster_choice]
        filtered_data = filtered_data.drop(['student_id', 'grade', 'cluster', 'time_delta'], axis=1)
        if show_preprocess:
            filtered_data = filtered_data.drop(['answer_display'], axis=1)
        else:
            filtered_data = filtered_data.drop(['answer'], axis=1)
        st.dataframe(filtered_data, height=400)