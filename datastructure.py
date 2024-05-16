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
        self.distance_matrix = nlp_pipeline.calculate_levenshtein_distance_matrix(student_answers)
        self.cluster_evaluation = nlp_pipeline.evaluate_clusters(student_answers, self.distance_matrix)

    def to_json_serializable_dict(self):
        # Assuming the rest of the session attributes are handled here
        session_dict = {
            "id": self.id,
            "study_condition": self.study_condition,
            "question_text": self.question_text,
            "reference_answer": self.reference_answer,
            "start_time": self.start_time.isoformat() if self.start_time else None,
            "end_time": self.end_time.isoformat() if self.end_time else None,
        }

        # Convert student_answers DataFrame, including Timedelta conversion
        if not self.student_answers.empty:
            session_dict['student_answers'] = self.student_answers.copy()

            # Convert 'time_delta' from Timedelta to total seconds for serialization
            session_dict['student_answers']['time_delta'] = session_dict['student_answers']['time_delta'].apply(
                lambda td: td.total_seconds())

            # Convert DataFrame to a list of dictionaries
            session_dict['student_answers'] = session_dict['student_answers'].to_dict(orient='records')

        return session_dict


def create_session(question_number):
    question_text = "WHY?"
    reference_answer = "BECAUSE!"
    session = Session(question_text=question_text, reference_answer=reference_answer, student_answers=get_SRA_numeric_df_for_question(question_number))
    return session


prompts = [
    "SRA_numeric_allAnswers_promptSE_22b.tsv",
    "SRA_numeric_allAnswers_promptSE_22c.tsv",
    "SRA_numeric_allAnswers_promptSE_24a.tsv",
    "SRA_numeric_allAnswers_promptSE_25a.tsv",
    "SRA_numeric_allAnswers_promptSE_27b.tsv",
    "SRA_numeric_allAnswers_promptSE_31b.tsv",
    "SRA_numeric_allAnswers_promptSE_3c.tsv",
    "SRA_numeric_allAnswers_promptSE_44.tsv",
    "SRA_numeric_allAnswers_promptSE_45.tsv",
    "SRA_numeric_allAnswers_promptSE_46.tsv",
    "SRA_numeric_allAnswers_promptSE_47b.tsv",
    "SRA_numeric_allAnswers_promptSE_48.tsv",
    "SRA_numeric_allAnswers_promptSE_4a.tsv",
    "SRA_numeric_allAnswers_promptSE_51b.tsv",
    "SRA_numeric_allAnswers_promptSHORT_CIRCUIT_EXPLAIN_Q_2.tsv",
    "SRA_numeric_allAnswers_promptSHORT_CIRCUIT_EXPLAIN_Q_4.tsv",
    "SRA_numeric_allAnswers_promptSHORT_CIRCUIT_EXPLAIN_Q_5.tsv",
    "SRA_numeric_allAnswers_promptSHORT_CIRCUIT_X_Q.tsv",
    "SRA_numeric_allAnswers_promptST_25b1.tsv",
    "SRA_numeric_allAnswers_promptST_25b2.tsv",
    "SRA_numeric_allAnswers_promptST_31b.tsv",
    "SRA_numeric_allAnswers_promptST_52a.tsv",
    "SRA_numeric_allAnswers_promptST_54b2.tsv",
    "SRA_numeric_allAnswers_promptST_54b3.tsv",
    "SRA_numeric_allAnswers_promptST_58.tsv",
    "SRA_numeric_allAnswers_promptST_59.tsv",
    "SRA_numeric_allAnswers_promptSWITCH_OPEN_EXPLAIN_Q.tsv",
    "SRA_numeric_allAnswers_promptSWITCH_TABLE_EXPLAIN_Q1.tsv",
    "SRA_numeric_allAnswers_promptSWITCH_TABLE_EXPLAIN_Q2.tsv",
    "SRA_numeric_allAnswers_promptSWITCH_TABLE_EXPLAIN_Q3.tsv",
    "SRA_numeric_allAnswers_promptTERMINAL_STATE_EXPLAIN_Q.tsv",
    "SRA_numeric_allAnswers_promptVB_1.tsv",
    "SRA_numeric_allAnswers_promptVB_12d.tsv",
    "SRA_numeric_allAnswers_promptVB_15a.tsv",
    "SRA_numeric_allAnswers_promptVB_15b.tsv",
    "SRA_numeric_allAnswers_promptVB_15c.tsv",
    "SRA_numeric_allAnswers_promptVB_22c.tsv",
    "SRA_numeric_allAnswers_promptVB_29.tsv",
    "SRA_numeric_allAnswers_promptVB_40a.tsv",
    "SRA_numeric_allAnswers_promptVB_42.tsv",
    "SRA_numeric_allAnswers_promptVB_5a.tsv",
    "SRA_numeric_allAnswers_promptVB_5b.tsv",
    "SRA_numeric_allAnswers_promptVB_5c.tsv",
    "SRA_numeric_allAnswers_promptVOLTAGE_AND_GAP_DISCUSS_Q.tsv",
    "SRA_numeric_allAnswers_promptVOLTAGE_DEFINE_Q.tsv",
    "SRA_numeric_allAnswers_promptVOLTAGE_DIFF_DISCUSS_1_Q.tsv",
    "SRA_numeric_allAnswers_promptVOLTAGE_DIFF_DISCUSS_2_Q.tsv",
    "SRA_numeric_allAnswers_promptVOLTAGE_ELECTRICAL_STATE_DISCUSS_Q.tsv",
    "SRA_numeric_allAnswers_promptVOLTAGE_GAP_EXPLAIN_WHY1.tsv",
    "SRA_numeric_allAnswers_promptVOLTAGE_GAP_EXPLAIN_WHY2.tsv",
    "SRA_numeric_allAnswers_promptVOLTAGE_GAP_EXPLAIN_WHY3.tsv",
    "SRA_numeric_allAnswers_promptVOLTAGE_GAP_EXPLAIN_WHY4.tsv",
    "SRA_numeric_allAnswers_promptVOLTAGE_GAP_EXPLAIN_WHY5.tsv",
    "SRA_numeric_allAnswers_promptVOLTAGE_GAP_EXPLAIN_WHY6.tsv",
    "SRA_numeric_allAnswers_promptVOLTAGE_INCOMPLETE_CIRCUIT_2_Q.tsv",
    "SRA_numeric_allAnswers_promptME_69b.tsv",
    "SRA_numeric_allAnswers_promptME_6b.tsv",
    "SRA_numeric_allAnswers_promptME_72.tsv",
    "SRA_numeric_allAnswers_promptME_73.tsv",
    "SRA_numeric_allAnswers_promptME_74b.tsv",
    "SRA_numeric_allAnswers_promptME_78b.tsv",
    "SRA_numeric_allAnswers_promptME_79.tsv",
    "SRA_numeric_allAnswers_promptME_7a.tsv",
    "SRA_numeric_allAnswers_promptME_7b.tsv",
    "SRA_numeric_allAnswers_promptMS_14b.tsv",
    "SRA_numeric_allAnswers_promptMS_30b.tsv",
    "SRA_numeric_allAnswers_promptMS_39.tsv",
    "SRA_numeric_allAnswers_promptMS_43a.tsv",
    "SRA_numeric_allAnswers_promptMS_43b.tsv",
    "SRA_numeric_allAnswers_promptMS_50a.tsv",
    "SRA_numeric_allAnswers_promptMS_50b.tsv",
    "SRA_numeric_allAnswers_promptMS_64a.tsv",
    "SRA_numeric_allAnswers_promptMX_1.tsv",
    "SRA_numeric_allAnswers_promptMX_10.tsv",
    "SRA_numeric_allAnswers_promptMX_11a.tsv",
    "SRA_numeric_allAnswers_promptMX_11b.tsv",
    "SRA_numeric_allAnswers_promptMX_11c.tsv",
    "SRA_numeric_allAnswers_promptMX_11e.tsv",
    "SRA_numeric_allAnswers_promptMX_11f.tsv",
    "SRA_numeric_allAnswers_promptMX_16a.tsv",
    "SRA_numeric_allAnswers_promptMX_18.tsv",
    "SRA_numeric_allAnswers_promptMX_19.tsv",
    "SRA_numeric_allAnswers_promptMX_22a.tsv",
    "SRA_numeric_allAnswers_promptMX_24.tsv",
    "SRA_numeric_allAnswers_promptMX_36a.tsv",
    "SRA_numeric_allAnswers_promptMX_36b.tsv",
    "SRA_numeric_allAnswers_promptMX_41.tsv",
    "SRA_numeric_allAnswers_promptMX_42a.tsv",
    "SRA_numeric_allAnswers_promptMX_46b.tsv",
    "SRA_numeric_allAnswers_promptMX_47b.tsv",
    "SRA_numeric_allAnswers_promptMX_49.tsv",
    "SRA_numeric_allAnswers_promptMX_52b.tsv",
    "SRA_numeric_allAnswers_promptMX_53.tsv",
    "SRA_numeric_allAnswers_promptOPT1_EXPLAIN_Q2.tsv",
    "SRA_numeric_allAnswers_promptOPT2_EXPLAIN_Q.tsv",
    "SRA_numeric_allAnswers_promptOTHER_TERMINAL_STATE_EXPLAIN_Q.tsv",
    "SRA_numeric_allAnswers_promptPARALLEL_SWITCH_EXPLAIN_Q1.tsv",
    "SRA_numeric_allAnswers_promptPARALLEL_SWITCH_EXPLAIN_Q2.tsv",
    "SRA_numeric_allAnswers_promptPARALLEL_SWITCH_EXPLAIN_Q3.tsv",
    "SRA_numeric_allAnswers_promptPS_12.tsv",
    "SRA_numeric_allAnswers_promptPS_15bp.tsv",
    "SRA_numeric_allAnswers_promptPS_24a.tsv",
    "SRA_numeric_allAnswers_promptPS_26p.tsv",
    "SRA_numeric_allAnswers_promptPS_2a.tsv",
    "SRA_numeric_allAnswers_promptPS_2b.tsv",
    "SRA_numeric_allAnswers_promptPS_44.tsv",
    "SRA_numeric_allAnswers_promptPS_45b.tsv",
    "SRA_numeric_allAnswers_promptPS_46b.tsv",
    "SRA_numeric_allAnswers_promptPS_4ap.tsv",
    "SRA_numeric_allAnswers_promptPS_4bp.tsv",
    "SRA_numeric_allAnswers_promptPS_51a.tsv",
    "SRA_numeric_allAnswers_promptPS_51b.tsv",
    "SRA_numeric_allAnswers_promptSE_10.tsv",
    "SRA_numeric_allAnswers_promptSE_16b2.tsv",
    "SRA_numeric_allAnswers_promptSE_22a.tsv",
    "SRA_numeric_allAnswers_promptEM_47.tsv",
    "SRA_numeric_allAnswers_promptEM_48b.tsv",
    "SRA_numeric_allAnswers_promptFN_17a.tsv",
    "SRA_numeric_allAnswers_promptFN_17c.tsv",
    "SRA_numeric_allAnswers_promptFN_19b.tsv",
    "SRA_numeric_allAnswers_promptFN_20a.tsv",
    "SRA_numeric_allAnswers_promptFN_20b.tsv",
    "SRA_numeric_allAnswers_promptFN_24b.tsv",
    "SRA_numeric_allAnswers_promptFN_24c.tsv",
    "SRA_numeric_allAnswers_promptFN_26b.tsv",
    "SRA_numeric_allAnswers_promptFN_27a.tsv",
    "SRA_numeric_allAnswers_promptFN_27b.tsv",
    "SRA_numeric_allAnswers_promptGIVE_CIRCUIT_TYPE_HYBRID_EXPLAIN_Q2.tsv",
    "SRA_numeric_allAnswers_promptGIVE_CIRCUIT_TYPE_HYBRID_EXPLAIN_Q3.tsv",
    "SRA_numeric_allAnswers_promptGIVE_CIRCUIT_TYPE_PARALLEL_EXPLAIN_Q2.tsv",
    "SRA_numeric_allAnswers_promptGIVE_CIRCUIT_TYPE_SERIES_EXPLAIN_Q.tsv",
    "SRA_numeric_allAnswers_promptHYBRID_BURNED_OUT_EXPLAIN_Q1.tsv",
    "SRA_numeric_allAnswers_promptHYBRID_BURNED_OUT_EXPLAIN_Q2.tsv",
    "SRA_numeric_allAnswers_promptHYBRID_BURNED_OUT_EXPLAIN_Q3.tsv",
    "SRA_numeric_allAnswers_promptHYBRID_BURNED_OUT_WHY_Q1.tsv",
    "SRA_numeric_allAnswers_promptHYBRID_BURNED_OUT_WHY_Q2.tsv",
    "SRA_numeric_allAnswers_promptHYBRID_BURNED_OUT_WHY_Q3.tsv",
    "SRA_numeric_allAnswers_promptII_12b.tsv",
    "SRA_numeric_allAnswers_promptII_13a.tsv",
    "SRA_numeric_allAnswers_promptII_13b.tsv",
    "SRA_numeric_allAnswers_promptII_20b.tsv",
    "SRA_numeric_allAnswers_promptII_24b.tsv",
    "SRA_numeric_allAnswers_promptII_26.tsv",
    "SRA_numeric_allAnswers_promptII_38.tsv",
    "SRA_numeric_allAnswers_promptLF_13a.tsv",
    "SRA_numeric_allAnswers_promptLF_13b.tsv",
    "SRA_numeric_allAnswers_promptLF_18a.tsv",
    "SRA_numeric_allAnswers_promptLF_26a2.tsv",
    "SRA_numeric_allAnswers_promptLF_26b2.tsv",
    "SRA_numeric_allAnswers_promptLF_27a.tsv",
    "SRA_numeric_allAnswers_promptLF_28a2.tsv",
    "SRA_numeric_allAnswers_promptLF_31b.tsv",
    "SRA_numeric_allAnswers_promptLF_33b.tsv",
    "SRA_numeric_allAnswers_promptLF_34b.tsv",
    "SRA_numeric_allAnswers_promptLF_39.tsv",
    "SRA_numeric_allAnswers_promptLF_6b.tsv",
    "SRA_numeric_allAnswers_promptLP_15c.tsv",
    "SRA_numeric_allAnswers_promptLP_16c.tsv",
    "SRA_numeric_allAnswers_promptLP_16d.tsv",
    "SRA_numeric_allAnswers_promptME_10.tsv",
    "SRA_numeric_allAnswers_promptME_17a.tsv",
    "SRA_numeric_allAnswers_promptME_17b.tsv",
    "SRA_numeric_allAnswers_promptME_17c.tsv",
    "SRA_numeric_allAnswers_promptME_17d.tsv",
    "SRA_numeric_allAnswers_promptME_17e.tsv",
    "SRA_numeric_allAnswers_promptME_27b.tsv",
    "SRA_numeric_allAnswers_promptME_28b.tsv",
    "SRA_numeric_allAnswers_promptME_30.tsv",
    "SRA_numeric_allAnswers_promptME_36.tsv",
    "SRA_numeric_allAnswers_promptME_38a.tsv",
    "SRA_numeric_allAnswers_promptME_5b.tsv",
    "SRA_numeric_allAnswers_promptME_65a.tsv",
    "SRA_numeric_allAnswers_promptME_65b.tsv",
    "SRA_numeric_allAnswers_promptME_66a.tsv",
    "SRA_numeric_allAnswers_promptME_66b.tsv",
    "SRA_numeric_allAnswers_promptBULB_C_VOLTAGE_EXPLAIN_WHY1.tsv",
    "SRA_numeric_allAnswers_promptBULB_C_VOLTAGE_EXPLAIN_WHY2.tsv",
    "SRA_numeric_allAnswers_promptBULB_C_VOLTAGE_EXPLAIN_WHY4.tsv",
    "SRA_numeric_allAnswers_promptBULB_C_VOLTAGE_EXPLAIN_WHY6.tsv",
    "SRA_numeric_allAnswers_promptBULB_ONLY_EXPLAIN_WHY2.tsv",
    "SRA_numeric_allAnswers_promptBULB_ONLY_EXPLAIN_WHY4.tsv",
    "SRA_numeric_allAnswers_promptBULB_ONLY_EXPLAIN_WHY6.tsv",
    "SRA_numeric_allAnswers_promptBURNED_BULB_LOCATE_EXPLAIN_Q.tsv",
    "SRA_numeric_allAnswers_promptBURNED_BULB_PARALLEL_EXPLAIN_Q1.tsv",
    "SRA_numeric_allAnswers_promptBURNED_BULB_PARALLEL_EXPLAIN_Q2.tsv",
    "SRA_numeric_allAnswers_promptBURNED_BULB_PARALLEL_EXPLAIN_Q3.tsv",
    "SRA_numeric_allAnswers_promptBURNED_BULB_PARALLEL_WHY_Q.tsv",
    "SRA_numeric_allAnswers_promptBURNED_BULB_SERIES_Q2.tsv",
    "SRA_numeric_allAnswers_promptCLOSED_PATH_EXPLAIN.tsv",
    "SRA_numeric_allAnswers_promptCONDITIONS_FOR_BULB_TO_LIGHT.tsv",
    "SRA_numeric_allAnswers_promptDAMAGED_BUILD_EXPLAIN_Q.tsv",
    "SRA_numeric_allAnswers_promptDAMAGED_BULB_EXPLAIN_2_Q.tsv",
    "SRA_numeric_allAnswers_promptDAMAGED_BULB_SWITCH_Q.tsv",
    "SRA_numeric_allAnswers_promptDESCRIBE_GAP_LOCATE_PROCEDURE_Q.tsv",
    "SRA_numeric_allAnswers_promptEM_13.tsv",
    "SRA_numeric_allAnswers_promptEM_16b.tsv",
    "SRA_numeric_allAnswers_promptEM_21a.tsv",
    "SRA_numeric_allAnswers_promptEM_21b.tsv",
    "SRA_numeric_allAnswers_promptEM_26.tsv",
    "SRA_numeric_allAnswers_promptEM_27b.tsv",
    "SRA_numeric_allAnswers_promptEM_33b.tsv",
    "SRA_numeric_allAnswers_promptEM_35.tsv",
    "SRA_numeric_allAnswers_promptEM_43b.tsv",
    "SRA_numeric_allAnswers_promptEM_45b.tsv",
    "SRA_numeric_allAnswers_promptEM_45c.tsv",
    "SRA_numeric_allAnswers_promptEM_46.tsv"
]


def get_SRA_numeric_df_for_question(question):
    # setup dataframe
    df = pd.read_csv(f'SRA_numeric/{prompts[question]}', sep="\t")
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
