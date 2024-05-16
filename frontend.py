import pandas as pd
import streamlit as st
import plots
import session



if 'initialized' not in st.session_state:
    session.create_session()
    session.cluster(True, True, True, 4)
    st.session_state['initialized'] = True
    st.session_state['selected_cluster'] = -1

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
    fix_cluster = st.checkbox('View Cluster')
    if st.button('All Answers In Cluster As Correct'):
        session.set_grade_for_cluster(st.session_state['selected_cluster'], 1)
    if st.button('All Answers In Cluster As False'):
        session.set_grade_for_cluster(st.session_state['selected_cluster'], 0)
    st.progress(session.get_progress(), text="Progress")
with col2:
    if 'update' in st.session_state and st.session_state['update']:
        clusters = session.get_session().student_answers['cluster'].unique()
        st.session_state['update'] = False

    cluster_choice = st.selectbox("Choose a Cluster", options=sorted(clusters))
    show_preprocess = st.checkbox('Show Preprocessed Data')

    if cluster_choice is not None:
        st.session_state['selected_cluster'] = cluster_choice
        filtered_data = session.get_session().student_answers[
            session.get_session().student_answers['cluster'] == cluster_choice]
        filtered_data = filtered_data.drop(['grade', 'cluster', 'time_delta'], axis=1)

        if show_preprocess:
            filtered_data = filtered_data.drop(['answer_display'], axis=1)
        else:
            filtered_data = filtered_data.drop(['answer'], axis=1)

        # Display data and buttons
        for index, row in filtered_data.iterrows():
            if cluster_choice != -1:
                cols = st.columns([4, 1])  # Adjust sizes if needed
                cols[0].write(row.values[1])
                button_key = f"btn_{index}"  # Unique key for each button
                if cols[1].button("🗑️", key=f"{index}_{button_key}"):
                    session.remove_student_from_cluster(row.values[0])
            else:
                cols = st.columns([3, 1, 1])  # Adjust sizes if needed
                cols[0].write(row.values[1])
                button_key = f"btn_{index}"  # Unique key for each button
                if cols[1].button("✔️", key=f"{index}_{button_key}_1"):
                    session.set_grade_for_student(row.values[0], 0)
                if cols[2].button("❌", key=f"{index}_{button_key}_2"):
                    session.set_grade_for_student(row.values[0], 0)


        if 'last_clicked' in st.session_state:
            st.write(st.session_state['last_clicked'])