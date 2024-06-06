import csv
import uuid
import pandas as pd
import nlp_pipeline


class Session:
    def __init__(self, participant_id, question_text, reference_answer, student_answers):
        self.id = participant_id
        self.study_condition = 1
        self.question_text = question_text
        self.reference_answer = reference_answer
        self.start_time = None
        self.end_time = None
        self.previous_cluster_info = None
        # as pandas dataframe
        self.student_answers = student_answers
        self.student_grades = pd.DataFrame()
        self.button_log = pd.DataFrame({
            "Button": pd.Series(dtype='str'),
            "Parameter": pd.Series(dtype='str'),
            "Timestamp": pd.Series(dtype='datetime64[ns]')
        })
        self.distance_matrix = nlp_pipeline.calculate_levenshtein_distance_matrix(student_answers)


def create_session(participant_id, question):
    df = pd.read_csv('beetle_questions.csv')
    question_text = df.loc[df['id'] == question, 'q_text'].values[0]
    reference_answer = "That there is a short circuit"
    session = Session(participant_id=participant_id, question_text=question_text, reference_answer=reference_answer,
                      student_answers=get_SRA_numeric_df_for_question(question))
    return session


prompts = [
    "SHORT_CIRCUIT_EXPLAIN_Q_2",
    "SHORT_CIRCUIT_EXPLAIN_Q_4",
    "SHORT_CIRCUIT_EXPLAIN_Q_5",
    "SHORT_CIRCUIT_X_Q",
    "SWITCH_OPEN_EXPLAIN_Q",
    "SWITCH_TABLE_EXPLAIN_Q1",
    "SWITCH_TABLE_EXPLAIN_Q2",
    "SWITCH_TABLE_EXPLAIN_Q3",
    "PARALLEL_SWITCH_EXPLAIN_Q1",
    "PARALLEL_SWITCH_EXPLAIN_Q2",
    "BULB_C_VOLTAGE_EXPLAIN_WHY2",
    "BULB_C_VOLTAGE_EXPLAIN_WHY6"
]


def get_SRA_numeric_df_for_question(question):
    # setup dataframe
    df = pd.read_csv(f'SRA_numeric/SRA_numeric_allAnswers_prompt{question}.tsv', sep="\t")
    df = df.reset_index(drop=True)
    df['student_id'] = range(1, len(df) + 1)
    df['grade'] = -1
    df['cluster'] = -1
    df['answer'] = df['AnswerText']
    df['answer_display'] = df['answer']
    df['time_delta'] = pd.Timedelta(seconds=0)
    df = df.drop('AnswerId', axis=1)
    df = df.drop('AnswerText', axis=1)
    df = df.drop('PromptId', axis=1)
    df = df.drop('Score1', axis=1)
    return df
