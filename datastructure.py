import csv
import uuid
import pandas as pd
import nlp_pipeline


class Session:
    def __init__(self, question_text, reference_answer, student_answers):
        self.id = str(uuid.uuid4())
        self.study_condition = 1
        self.question_text = question_text
        self.reference_answer = reference_answer
        self.start_time = None
        self.end_time = None
        self.previous_cluster_info = None
        # as pandas dataframe
        self.student_answers = student_answers
        self.student_grades = pd.DataFrame()
        self.distance_matrix = nlp_pipeline.calculate_levenshtein_distance_matrix(student_answers)


def create_session(question_number):
    df = pd.read_csv('beetle_questions.csv')
    print(prompts[question_number])
    question_text = df.loc[df['id'] == prompts[question_number], 'q_text'].values[0]
    reference_answer = "That there is a short circuit"
    session = Session(question_text=question_text, reference_answer=reference_answer, student_answers=get_SRA_numeric_df_for_question(question_number))
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
    "TERMINAL_STATE_EXPLAIN_Q",
    "VOLTAGE_AND_GAP_DISCUSS_Q",
    "VOLTAGE_DEFINE_Q",
    "VOLTAGE_DIFF_DISCUSS_1_Q",
    "VOLTAGE_DIFF_DISCUSS_2_Q",
    "VOLTAGE_ELECTRICAL_STATE_DISCUSS_Q",
    "VOLTAGE_GAP_EXPLAIN_WHY1",
    "VOLTAGE_GAP_EXPLAIN_WHY2",
    "VOLTAGE_GAP_EXPLAIN_WHY3",
    "VOLTAGE_GAP_EXPLAIN_WHY4",
    "VOLTAGE_GAP_EXPLAIN_WHY5",
    "VOLTAGE_GAP_EXPLAIN_WHY6",
    "VOLTAGE_INCOMPLETE_CIRCUIT_2_Q",
    "OPT1_EXPLAIN_Q2",
    "OPT2_EXPLAIN_Q",
    "OTHER_TERMINAL_STATE_EXPLAIN_Q",
    "PARALLEL_SWITCH_EXPLAIN_Q1",
    "PARALLEL_SWITCH_EXPLAIN_Q2",
    "PARALLEL_SWITCH_EXPLAIN_Q3",
    "GIVE_CIRCUIT_TYPE_HYBRID_EXPLAIN_Q2",
    "GIVE_CIRCUIT_TYPE_HYBRID_EXPLAIN_Q3",
    "GIVE_CIRCUIT_TYPE_PARALLEL_EXPLAIN_Q2",
    "GIVE_CIRCUIT_TYPE_SERIES_EXPLAIN_Q",
    "HYBRID_BURNED_OUT_EXPLAIN_Q1",
    "HYBRID_BURNED_OUT_EXPLAIN_Q2",
    "HYBRID_BURNED_OUT_EXPLAIN_Q3",
    "HYBRID_BURNED_OUT_WHY_Q1",
    "HYBRID_BURNED_OUT_WHY_Q2",
    "HYBRID_BURNED_OUT_WHY_Q3",
    "BULB_C_VOLTAGE_EXPLAIN_WHY1",
    "BULB_C_VOLTAGE_EXPLAIN_WHY2",
    "BULB_C_VOLTAGE_EXPLAIN_WHY4",
    "BULB_C_VOLTAGE_EXPLAIN_WHY6",
    "BULB_ONLY_EXPLAIN_WHY2",
    "BULB_ONLY_EXPLAIN_WHY4",
    "BULB_ONLY_EXPLAIN_WHY6",
    "BURNED_BULB_LOCATE_EXPLAIN_Q",
    "BURNED_BULB_PARALLEL_EXPLAIN_Q1",
    "BURNED_BULB_PARALLEL_EXPLAIN_Q2",
    "BURNED_BULB_PARALLEL_EXPLAIN_Q3",
    "BURNED_BULB_PARALLEL_WHY_Q",
    "BURNED_BULB_SERIES_Q2",
    "CLOSED_PATH_EXPLAIN",
    "CONDITIONS_FOR_BULB_TO_LIGHT",
    "DAMAGED_BUILD_EXPLAIN_Q",
    "DAMAGED_BULB_EXPLAIN_2_Q",
    "DAMAGED_BULB_SWITCH_Q",
    "DESCRIBE_GAP_LOCATE_PROCEDURE_Q"
]


def get_SRA_numeric_df_for_question(question):
    # setup dataframe
    df = pd.read_csv(f'SRA_numeric/SRA_numeric_allAnswers_prompt{prompts[question]}.tsv', sep="\t")
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
