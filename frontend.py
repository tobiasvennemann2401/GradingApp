import numpy as np
import streamlit as st

import datastructure
import nlp_pipeline
import session


@st.experimental_dialog("Enter Study Data")
def start_dialog():
    st.write(f"Please enter you participant data")
    participant_id = st.number_input('Participant Id', min_value=0, max_value=50, step=1)
    prompt_choice = st.selectbox("Prompt", options=datastructure.prompts)
    condition = st.selectbox("Condition", options=["Single Grading", "Cluster Grading"])
    st.session_state['condition'] = condition
    if st.button('Start'):
        session.create_session(participant_id, prompt_choice)
        st.rerun()


if 'initialized' not in st.session_state:
    st.session_state['initialized'] = True
    st.session_state['update'] = True
    st.session_state['expand_contractions'] = False
    st.session_state['remove_stopwords'] = False
    st.session_state['preproc_method'] = "Nothing"
    st.session_state['non_compliance'] = False
    st.session_state['distance_calculation_method'] = "Character Based"
    st.session_state['distance_threshold'] = 0
    st.session_state['filter_negations'] = False
    st.session_state['show_preprocess'] = False
    st.session_state['clusters'] = [-1]
    st.session_state.cluster_choice = -1
    st.session_state['last_choice'] = -1
    st.session_state['condition'] = 'Single Grading'
    start_dialog()
else:
    if st.session_state['condition'] != 'Single Grading':
        st.sidebar.header('Clustering Options')

        # Sidebar
        with st.sidebar.expander("Preprocessing"):
            expand_contractions = st.checkbox('Expand Contractions',
                                              help='This checkbox enables the changes from: don´t -> do not')
            if expand_contractions != st.session_state['expand_contractions']:
                st.session_state['expand_contractions'] = expand_contractions
                session.log_button("expand_contractions", expand_contractions)
            remove_stopwords = st.checkbox('Remove Stopwords',
                                           help='This checkbox removes certain words as: do, and, is, if')
            if remove_stopwords != st.session_state['remove_stopwords']:
                st.session_state['remove_stopwords'] = remove_stopwords
                session.log_button("remove_stopwords", remove_stopwords)
            preproc_method = st.radio(
                "Word Normalization Method",
                ["Nothing", "Lemmatization", "Stemming"],
                captions=["Words arent changed", "Words are transformed into there dictionary form",
                          "Words are cut after the word stem"])
            if preproc_method != st.session_state['preproc_method']:
                st.session_state['preproc_method'] = preproc_method
                session.log_button("preproc_method", preproc_method)
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
                session.preprocess(expand_contractions, remove_stopwords, preproc_method)
                st.text("After Preprocessing")
                st.write(sentence)

        with st.sidebar.expander("Non Compliance Check"):
            non_compliance = st.checkbox('Create Non Compliance Cluster')
            if non_compliance != st.session_state['non_compliance']:
                st.session_state['non_compliance'] = non_compliance
                session.log_button("non_compliance", non_compliance)
            st.write(
                "Answers that do not contain any of the top 10 most used words are considered non compliant and put in a separate cluster.")
            st.text("Top 10 words")
            st.table(session.get_top_10_words())

        with st.sidebar.expander("Clustering"):
            distance_calculation_method = st.radio(
                "Distance Calculation Method",
                ["Character Based", "Token Based"],
                captions=["Distance is calculated based on how many characters are different",
                          "Distance is calculated based on how many words are different"])
            if distance_calculation_method != st.session_state['distance_calculation_method']:
                st.session_state['distance_calculation_method'] = distance_calculation_method
                session.log_button("distance_calculation_method", distance_calculation_method)
            distance_threshold = st.number_input('Distance Threshold', min_value=0, max_value=10, step=1, value=5 if distance_calculation_method == "Character Based" else 2,
                                                 help='This value determines the maximum distance two answers in one cluster can be apart')
            if distance_threshold != st.session_state['distance_threshold']:
                st.session_state['distance_threshold'] = distance_threshold
                session.log_button("distance_threshold", distance_threshold)
            filter_negations = st.checkbox('Filter Negations',
                                           help='Prevents sentences with opposite meanings (e.g., "I like it" vs. "I dont like it") from being grouped together.')
            if filter_negations != st.session_state['filter_negations']:
                st.session_state['filter_negations'] = filter_negations
                session.log_button("filter_negations", filter_negations)

        if st.sidebar.button('Cluster'):
            session.cluster(filter_negations, distance_calculation_method == "Token Based", non_compliance,
                            distance_threshold)
            st.session_state['clusters'] = session.sort_clusters()
            st.session_state.cluster_choice = st.session_state['clusters'][0]
            st.session_state['last_choice'] = st.session_state.cluster_choice
            session.log_button("cluster")







    @st.experimental_dialog("Show Graded Answers")
    def show_graded_answers():
        graded_answers = session.get_session().student_grades.drop(['cluster'], axis=1)
        show_preproc = st.checkbox('Show Preprocessing Result', key="show_preproc_dialog")
        if show_preproc:
            graded_answers = graded_answers.drop(['answer_display'], axis=1)
        else:
            graded_answers = graded_answers.drop(['answer'], axis=1)
        header_cols = st.columns([4, 1, 1])
        header_cols[0].write("Answer")
        header_cols[1].write("Grade")
        header_cols[2].write("Revoke Grade")
        for index, row in graded_answers.iloc[::-1].iterrows():
            dialog_cols = st.columns([4, 1, 1])
            dialog_cols[0].write(row.values[2])
            dialog_cols[1].write("❌" if row.values[1] == 0 else "✔️")
            if dialog_cols[2].button("🗑️", key=f"{index}_btn_{index}", help="Revoke Grade"):
                session.revoke_grade_of_student(row.values[0])
                st.session_state['update'] = True
                session.log_button("revoke_grade", row.values[0])
                st.rerun()

    # MainPage
    top_col1, top_col2 = st.columns([3, 1])
    with top_col1:
        st.progress(session.get_progress(), text=f"Progress: {round(session.get_progress() * 100, 2)}%")
    with top_col2:
        if st.button('Graded Answers', disabled=not (session.get_progress() > 0)):
            show_graded_answers()

    col1, col2 = st.columns(2)

    with col1:
        try:
            st.image(session.image)
        except:
            print("Ich habe heute leider kein Foto für dich")
        with st.container(border=20):
            st.text("Question:")
            st.write(session.get_session().question_text)
        st.text("\n\n")
        with st.container(border=20):
            st.text("Examples for Correct Answer:")
            for answer in session.get_session().reference_answer:
                st.write(f"• {answer}")
        st.text("\n\n")


    if session.get_progress() < 1:
        with col2:
            if 'update' in st.session_state and st.session_state['update']:
                st.session_state['clusters'] = session.sort_clusters()
                st.session_state['update'] = False

            if st.session_state.cluster_choice not in st.session_state['clusters']:
                st.session_state.cluster_choice = st.session_state['clusters'][0]

            st.session_state.cluster_choice = st.selectbox("Choose a Cluster", options=st.session_state['clusters'],
                                                           index=st.session_state['clusters'].index(st.session_state.cluster_choice),
                                                           format_func=lambda
                                                               x: "Unclustered" if x == -1 else "Non Compliance" if x == -2 else session.get_cluster_header(
                                                               x))
            if st.session_state.cluster_choice != st.session_state['last_choice']:
                st.session_state['last_choice'] = st.session_state.cluster_choice
                session.log_button("cluster_choice", st.session_state.cluster_choice)

            if st.session_state['condition'] != 'Single Grading':
                show_preprocess = st.checkbox('Show Preprocessing Result', key="show_preproc")
                if show_preprocess != st.session_state['show_preprocess']:
                    st.session_state['show_preprocess'] = show_preprocess
                    session.log_button("show_preprocess", show_preprocess)

            if st.session_state.cluster_choice is not None:
                filtered_data = session.get_session().student_answers[
                    (session.get_session().student_answers['cluster'] == st.session_state.cluster_choice) & (
                            session.get_session().student_answers['grade'] == -1)]
                filtered_data = filtered_data.drop(['grade', 'cluster'], axis=1).iloc[::-1]

                if st.session_state['show_preprocess']:
                    filtered_data = filtered_data.drop(['answer_display'], axis=1)
                else:
                    filtered_data = filtered_data.drop(['answer'], axis=1)

                for index, row in filtered_data.iterrows():
                    cols = st.columns([3, 1, 1])
                    cols[0].write(row.values[1])
                    if cols[1].button("✔️", key=f"{index}_btn_{index}_1"):
                        session.remove_student_from_cluster(row.values[0])
                        session.set_grade_for_student(row.values[0], 1)
                        st.session_state['update'] = True
                        session.log_button("grade_single_correct", row.values[0])
                        st.rerun()
                    if cols[2].button("❌", key=f"{index}_btn_{index}_2"):
                        session.remove_student_from_cluster(row.values[0])
                        session.set_grade_for_student(row.values[0], 0)
                        st.session_state['update'] = True
                        session.log_button("grade_single_false", row.values[0])
                        st.rerun()
                if st.session_state.cluster_choice != -1:
                    bot_cols = st.columns([1, 1])
                    if bot_cols[0].button('Grade Complete Cluster as Correct ✔️'):
                        session.set_grade_for_cluster(st.session_state.cluster_choice, 1)
                        st.session_state['update'] = True
                        session.log_button("grade_cluster_correct", st.session_state.cluster_choice)
                        st.rerun()
                    if bot_cols[1].button('Grade Complete Cluster as Incorrect ❌'):
                        session.set_grade_for_cluster(st.session_state.cluster_choice, 0)
                        st.session_state['update'] = True
                        session.log_button("grade_cluster_false", st.session_state.cluster_choice)
                        st.rerun()


    def convert_df(df):
        return df.to_csv(index=False).encode('utf-8')


    @st.experimental_dialog("Export your Data")
    def export():
        st.write(f"You can now export your data as CSV")
        grade_csv = convert_df(session.get_session().student_grades)
        st.download_button(
            "Press to Download Grades CSV",
            grade_csv,
            f"grades_{session.get_session().id}_{st.session_state['condition'].strip()}.csv",
            "text/csv",
            key='download-grade-csv'
        )
        log_csv = convert_df(session.get_session().button_log)
        st.download_button(
            "Press to Download Log",
            log_csv,
            f"button_log_{session.get_session().id}_{st.session_state['condition'].strip()}.csv",
            "text/csv",
            key='download-log-csv'
        )


    if session.get_progress() >= 1:
        export()

    if st.sidebar.button('Open Emergency Dialog'):
        export()
