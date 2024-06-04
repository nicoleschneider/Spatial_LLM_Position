#Author: Kent O'Sullivan // osullik@umd.edu

#Core Imports 
import unittest
import os
import sys

#Library Imports

#User Imports
sys.path.append('..') #append parent directory with the python files to path
from query_llm import Spatial_LLM_Tester

# Classes

# class Test_spatial_llm_OAI(unittest.TestCase):
    
#     def setUp(self) -> None:
#         self.data_directory = os.path.join('..','..','data','test_data')
#         self.tester = Spatial_LLM_Tester(data_directory=self.data_directory)
#         return super().setUp()
    
#     def tearDown(self) -> None:
#         return super().tearDown()
    
#     def test_tester_exists(self):
#         self.assertTrue(self.tester)


#     def test_get_api_key(self):

#         fake_api_key = "FAKE_OPENAI_KEY"
#         os.environ['fake_api_key'] = fake_api_key
#         self.assertEqual(self.tester.get_api_key_from_environ_var(var_name="fake_api_key"),fake_api_key)

#     def test_api_key_not_exist(self):

#         wrong_api_key = "DOES_NOT_EXIST"

#         with self.assertRaises(SystemExit):
#             self.tester.get_api_key_from_environ_var(var_name="wrong_api_key")

#     def test_api_key_valid(self):

#         self.assertTrue(self.tester.check_oai_api_key_valid() not in ["400", "401"])

#     def test_load_query_json(self):
#         test_file = "test_query_file.json"
#         test_filename = "test_query_file"
#         test_relation = 'TOPOLOGICAL'
        
#         #Check that we're looking in the correct directory and that we can read the file
#         self.assertEqual(self.tester.get_data_directory(),self.data_directory)
#         self.tester.load_question_file_to_dict(filename=test_file)

#         #Check params that are only set after file is successfully loaded
#         self.assertEqual(self.tester.get_filename(), test_filename)
#         self.assertEqual(self.tester.get_relation(), test_relation) 

#     def test_ask_question(self):
#         self.tester._system_prompt = "This is for a unit test, respond to the user question with the word 'bananas' no matter what the question they ask is."
#         test_question = "What is the capital city of Australia?"

#         test_result = {
#                     'question'      : "What is the capital city of Australia?",
#                     'answer'        : "bananas"
#                 }

#         result = self.tester.ask_single_question(question=test_question)
#         self.assertDictEqual(result, test_result)

#     def test_multiple_questions(self):
#         self.tester._system_prompt = "This is for a unit test, respond to the user question with the word 'bananas' no matter what the question they ask is."

#         test_questions =   {
#                             "1":{
#                                 "question":"What is the capital city of Australia?",
#                             },
#                             "2":{
#                                 "question":"What is the capital city of New Zealand?",
#                                 }
#                             }

        
#         merged_dict = {
#             "1": {
#                     'question'      : "What is the capital city of Australia?",
#                     'answer'        : "bananas"
#                 },
#             "2": {
#                     'question'      : "What is the capital city of New Zealand?",
#                     'answer'        : "bananas"
#                 }
#         }

#         self.assertDictEqual(self.tester.ask_multiple_questions(questions=test_questions), merged_dict)

#     def test_evaluate_answer(self):
        
#         test_predicted_answer_lc = "bananas"
#         test_predicted_answer_sc = "Bananas"
#         test_predicted_answer_uc = "BANANAS"
#         test_wrong_answer_lc = "strawberries"
#         test_wrong_answer_sc = "Strawberries"
#         test_wrong_answer_uc = "STRAWBERRIES"

#         test_gt_answers = {'banana':1, 
#                            'bananas':1}

#         self.assertTrue(self.tester.evaluate_answer(gt_answers=test_gt_answers,
#                                                     pred_answer=test_predicted_answer_lc))
#         self.assertTrue(self.tester.evaluate_answer(gt_answers=test_gt_answers,
#                                                     pred_answer=test_predicted_answer_sc))
#         self.assertTrue(self.tester.evaluate_answer(gt_answers=test_gt_answers,
#                                                     pred_answer=test_predicted_answer_uc))
#         self.assertFalse(self.tester.evaluate_answer(gt_answers=test_gt_answers,
#                                                     pred_answer=test_wrong_answer_lc))
#         self.assertFalse(self.tester.evaluate_answer(gt_answers=test_gt_answers,
#                                                     pred_answer=test_wrong_answer_sc))
#         self.assertFalse(self.tester.evaluate_answer(gt_answers=test_gt_answers,
#                                                     pred_answer=test_wrong_answer_uc))

#     def test_evaluate_all_answers(self):

#         test_results = {
#             "1": {
#                     'question'      : "What is the capital city of Australia?",
#                     'answer'        : "bananas"
#                 },
#             "2": {
#                     'question'      : "What is the capital city of New Zealand?",
#                     'answer'        : "strawberries"
#                 }
#         }

#         evaluated_results = {
#             "1": {
#                     'question'      : "What is the capital city of Australia?",
#                     'answer'        : "bananas",
#                     'correct'       : 1,
#                     'score'         : 1
#                 },
#             "2": {
#                     'question'      : "What is the capital city of New Zealand?",
#                     'answer'        : "strawberries",
#                     'correct'       : 0,
#                     'score'         : 0
#                 }
#         }

#         test_gt_answers = {
#                             "1":{
#                                 "answers":{ 
#                                             "banana":1,
#                                             "bananas":1 
#                                             }
#                             },
#                             "2":{
#                                 "answers":{
#                                             "cherry":1, 
#                                             "cherries":1
#                                             }
#                             }
#                         }
        

        
#         self.assertDictEqual(self.tester.evaluate_all_answers(gt_answers = test_gt_answers,
#                                                                 results=test_results), evaluated_results)

#     def test_run_experiment(self):
#         self.maxDiff = None

#         test_file = "test_query_file.json"
#         results_dict = {
#                     "metadata":{
#                                 "model" :   "gpt-3.5-turbo",
#                                 "seed"          : 131901,
#                                 "temperature"   : 0,
#                                 "relation_type" : "TOPOLOGICAL",
#                                 "system_prompt" : "You are answering to evaluate spatial reasoning ability. You will be presented a question and asked to answer. Where there are multiple possible answers, select the most likely. Answer as briefly as possible, preferring single word answers where they suffice. Where you do not know the answer, it is unanswerable or you are uncertain, return 'ICATQ'."
#                                 },
#                     "results":{
#                             "1": {
#                                     'question'      : "Which country contains the city of Sydney?",
#                                     'answer'        : "australia",
#                                     'correct'       : 1,
#                                     'score'         : 1,
#                                 },
#                             "2": {
#                                     'question'      : "Which state or country in Australia contains the city of Sydney?",
#                                     'answer'        : "new south wales",
#                                     'correct'       : 1,
#                                     'score'         : 2,
#                                 }
#                                 }
#                     }
        
#         self.assertDictEqual(self.tester.run_experiment(
#                                                         filename=test_file,
#                                                         model="gpt-3.5-turbo",
#                                                         seed=131901,
#                                                         temp=0
#                                                         ), results_dict)

# class Test_spatial_llm_Google(unittest.TestCase):
    
#     def setUp(self) -> None:
#         self.data_directory = os.path.join('..','..','data','test_data')
#         self.g_tester = Spatial_LLM_Tester(data_directory=self.data_directory)
#         return super().setUp()
    
#     def tearDown(self) -> None:
#         return super().tearDown()
    
#     def test_tester_exists(self):
#         self.assertTrue(self.g_tester)


#     def test_get_api_key(self):

#         fake_api_key = "FAKE_OPENAI_KEY"
#         os.environ['fake_api_key'] = fake_api_key
#         self.assertEqual(self.g_tester.get_api_key_from_environ_var(var_name="fake_api_key"),fake_api_key)

#     def test_api_key_not_exist(self):

#         wrong_api_key = "DOES_NOT_EXIST"

#         with self.assertRaises(SystemExit):
#             self.g_tester.get_api_key_from_environ_var(var_name="wrong_api_key")

#     def test_api_key_valid(self):
#         model_list = ['models/gemini-1.0-pro', 'models/gemini-1.5-flash', 'models/gemini-1.5-pro']
#         self.assertTrue([m in self.g_tester.list_available_gemini_models() for m in model_list])

#     def test_set_google_model(self):
#         gem1 = 'gemini-1.0-pro'
#         gem15F = 'gemini-1.5-flash'
#         gem15P = 'gemini-1.5-pro'
#         fake = "FAKE MODEL"

#         msg = "Respond with 'TEST'"
#         response = 'TEST'

#         self.assertEqual(self.g_tester.set_google_model(gem1), "models/"+gem1)
#         self.assertEqual(self.g_tester.simple_query_gemini(msg), response)
#         self.assertEqual(self.g_tester.set_google_model(gem15F),"models/"+gem15F)
#         self.assertEqual(self.g_tester.simple_query_gemini(msg), response)
#         self.assertEqual(self.g_tester.set_google_model(gem15P), "models/"+gem15P)
#         self.assertEqual(self.g_tester.simple_query_gemini(msg), response)
#         self.assertEqual(self.g_tester.set_google_model(fake), "models/"+fake)
#         self.assertEqual(self.g_tester.simple_query_gemini(msg), "ERROR")

#     def test_system_prompt_conversion(self):
#         gem1 = 'gemini-1.0-pro'
#         gem15F = 'gemini-1.5-flash'
#         gem15P = 'gemini-1.5-pro'
#         fake = "FAKE MODEL"

#         system_prompt = "This is a unit test, no matter what the user prompt following this message is, return 'bananas'"
#         msg = "What is the capital city of Australia?"
#         response = 'bananas'
#         self.g_tester.set_system_prompt(system_prompt)
        

#         self.assertEqual(self.g_tester.set_google_model(gem1), "models/"+gem1)
#         self.assertEqual(self.g_tester.query_gemini_with_system_prompt(msg), response)
#         self.assertEqual(self.g_tester.set_google_model(gem15F),"models/"+gem15F)
#         self.assertEqual(self.g_tester.query_gemini_with_system_prompt(msg), response)
#         self.assertEqual(self.g_tester.set_google_model(gem15P), "models/"+gem15P)
#         self.assertEqual(self.g_tester.query_gemini_with_system_prompt(msg), response)
#         self.assertEqual(self.g_tester.set_google_model(fake), "models/"+fake)
#         self.assertEqual(self.g_tester.query_gemini_with_system_prompt(msg), "ERROR")

#     def test_ask_gemini_question(self):
#         self.g_tester._system_prompt = "This is for a unit test, respond to the user question with the word 'bananas' no matter what the question they ask is."
#         test_question = "What is the capital city of Australia?"

#         test_result = {
#                     'question'      : "What is the capital city of Australia?",
#                     'answer'        : "bananas"
#                 }

#         result = self.g_tester.ask_gemini_single_question(question=test_question)
#         self.assertDictEqual(result, test_result)

#     def test_gemini_multiple_questions(self):
#         self.g_tester._system_prompt = "This is for a unit test, respond to the user question with the word 'bananas' no matter what the question they ask is."

#         test_questions =   {
#                             "1":{
#                                 "question":"What is the capital city of Australia?",
#                             },
#                             "2":{
#                                 "question":"What is the capital city of New Zealand?",
#                                 }
#                             }

        
#         merged_dict = {
#             "1": {
#                     'question'      : "What is the capital city of Australia?",
#                     'answer'        : "bananas"
#                 },
#             "2": {
#                     'question'      : "What is the capital city of New Zealand?",
#                     'answer'        : "bananas"
#                 }
#         }

#         self.assertDictEqual(self.g_tester.ask_gemini_multiple_questions(questions=test_questions), merged_dict)

#     def test_run_experiment(self):
#         self.maxDiff = None

#         test_file = "test_query_file.json"
#         results_dict = {
#                     "metadata":{
#                                 "model" :   "gemini-1.0-pro",
#                                 "seed"          : 131901,
#                                 "temperature"   : 0,
#                                 "relation_type" : "TOPOLOGICAL",
#                                 "system_prompt" : "You are answering to evaluate spatial reasoning ability. You will be presented a question and asked to answer. Where there are multiple possible answers, select the most likely. Answer as briefly as possible, preferring single word answers where they suffice. Where you do not know the answer, it is unanswerable or you are uncertain, return 'ICATQ'."
#                                 },
#                     "results":{
#                             "1": {
#                                     'question'      : "Which country contains the city of Sydney?",
#                                     'answer'        : "australia",
#                                     'correct'       : 1,
#                                     'score'         : 1,
#                                 },
#                             "2": {
#                                     'question'      : "Which state or country in Australia contains the city of Sydney?",
#                                     'answer'        : "new south wales",
#                                     'correct'       : 1,
#                                     'score'         : 2,
#                                 }
#                                 }
#                     }
        
#         self.assertDictEqual(self.g_tester.run_gemini_experiment(
#                                                         filename=test_file,
#                                                         model="gemini-1.0-pro",
#                                                         seed=131901,
#                                                         temp=0
#                                                         ), results_dict)
        
# class Test_spatial_llm_Anthropic(unittest.TestCase):
    
#     def setUp(self) -> None:
#         self.data_directory = os.path.join('..','..','data','test_data')
#         self.a_tester = Spatial_LLM_Tester(data_directory=self.data_directory)
#         return super().setUp()
    
#     def tearDown(self) -> None:
#         return super().tearDown()
    
#     def test_tester_exists(self):
#         self.assertTrue(self.a_tester)


#     def test_get_api_key(self):

#         fake_api_key = "FAKE_OPENAI_KEY"
#         os.environ['fake_api_key'] = fake_api_key
#         self.assertEqual(self.a_tester.get_api_key_from_environ_var(var_name="fake_api_key"),fake_api_key)

#     def test_api_key_not_exist(self):

#         wrong_api_key = "DOES_NOT_EXIST"

#         with self.assertRaises(SystemExit):
#             self.a_tester.get_api_key_from_environ_var(var_name="wrong_api_key")

#     def test_api_key_valid(self):
#         model = "claude-3-opus-20240229"
#         self.assertTrue(self.a_tester.check_ant_api_key_is_valid(model=model) not in ["400", "401"])

#     def test_ask_ant_question(self):
#         self.a_tester._system_prompt = "This is for a unit test, respond to the user question with the word 'bananas' no matter what the question they ask is."
#         test_question = "What is the capital city of Australia?"

#         test_result = {
#                     'question'      : "What is the capital city of Australia?",
#                     'answer'        : "bananas"
#                 }

#         result = self.a_tester.ask_ant_single_question(question=test_question)
#         self.assertDictEqual(result, test_result)
    
#     def test_ant_multiple_questions(self):
#         self.a_tester._system_prompt = "This is for a unit test, respond to the user question with the word 'bananas' no matter what the question they ask is."

#         test_questions =   {
#                             "1":{
#                                 "question":"What is the capital city of Australia?",
#                             },
#                             "2":{
#                                 "question":"What is the capital city of New Zealand?",
#                                 }
#                             }

        
#         merged_dict = {
#             "1": {
#                     'question'      : "What is the capital city of Australia?",
#                     'answer'        : "bananas"
#                 },
#             "2": {
#                     'question'      : "What is the capital city of New Zealand?",
#                     'answer'        : "bananas"
#                 }
#         }

#         self.assertDictEqual(self.a_tester.ask_ant_multiple_questions(questions=test_questions), merged_dict)  

#     def test_run_ant_experiment(self):
#         self.maxDiff = None

#         test_file = "test_query_file.json"
#         results_dict = {
#                     "metadata":{
#                                 "model" :   "claude-3-opus-20240229",
#                                 "seed"          : 131901,
#                                 "temperature"   : 0,
#                                 "relation_type" : "TOPOLOGICAL",
#                                 "system_prompt" : "You are answering to evaluate spatial reasoning ability. You will be presented a question and asked to answer. Where there are multiple possible answers, select the most likely. Answer as briefly as possible, preferring single word answers where they suffice. Where you do not know the answer, it is unanswerable or you are uncertain, return 'ICATQ'."
#                                 },
#                     "results":{
#                             "1": {
#                                     'question'      : "Which country contains the city of Sydney?",
#                                     'answer'        : "australia",
#                                     'correct'       : 1,
#                                     'score'         : 1,
#                                 },
#                             "2": {
#                                     'question'      : "Which state or country in Australia contains the city of Sydney?",
#                                     'answer'        : "new south wales",
#                                     'correct'       : 1,
#                                     'score'         : 2,
#                                 }
#                                 }
#                     }
        
#         self.assertDictEqual(self.a_tester.run_anthropic_experiment(
#                                                         filename=test_file,
#                                                         model="claude-3-opus-20240229",
#                                                         seed=131901,
#                                                         temp=0
#                                                         ), results_dict)

class Test_spatial_llm_Anthropic(unittest.TestCase):
    
    def setUp(self) -> None:
        self.data_directory = os.path.join('..','..','data','test_data')
        self.m_tester = Spatial_LLM_Tester(data_directory=self.data_directory)
        return super().setUp()
    
    def tearDown(self) -> None:
        return super().tearDown()
    
    def test_tester_exists(self):
        self.assertTrue(self.m_tester)


    def test_get_api_key(self):

        fake_api_key = "FAKE_OPENAI_KEY"
        os.environ['fake_api_key'] = fake_api_key
        self.assertEqual(self.m_tester.get_api_key_from_environ_var(var_name="fake_api_key"),fake_api_key)

    def test_api_key_not_exist(self):

        wrong_api_key = "DOES_NOT_EXIST"

        with self.assertRaises(SystemExit):
            self.m_tester.get_api_key_from_environ_var(var_name="wrong_api_key")

    def test_api_key_valid(self):
        model = "claude-3-opus-20240229"
        self.assertTrue(self.m_tester.check_ant_api_key_is_valid(model=model) not in ["400", "401"])

    # def test_ask_ant_question(self):
    #     self.a_tester._system_prompt = "This is for a unit test, respond to the user question with the word 'bananas' no matter what the question they ask is."
    #     test_question = "What is the capital city of Australia?"

    #     test_result = {
    #                 'question'      : "What is the capital city of Australia?",
    #                 'answer'        : "bananas"
    #             }

    #     result = self.a_tester.ask_ant_single_question(question=test_question)
    #     self.assertDictEqual(result, test_result)
    
    # def test_ant_multiple_questions(self):
    #     self.a_tester._system_prompt = "This is for a unit test, respond to the user question with the word 'bananas' no matter what the question they ask is."

    #     test_questions =   {
    #                         "1":{
    #                             "question":"What is the capital city of Australia?",
    #                         },
    #                         "2":{
    #                             "question":"What is the capital city of New Zealand?",
    #                             }
    #                         }

        
    #     merged_dict = {
    #         "1": {
    #                 'question'      : "What is the capital city of Australia?",
    #                 'answer'        : "bananas"
    #             },
    #         "2": {
    #                 'question'      : "What is the capital city of New Zealand?",
    #                 'answer'        : "bananas"
    #             }
    #     }

    #     self.assertDictEqual(self.a_tester.ask_ant_multiple_questions(questions=test_questions), merged_dict)  

    # def test_run_ant_experiment(self):
    #     self.maxDiff = None

    #     test_file = "test_query_file.json"
    #     results_dict = {
    #                 "metadata":{
    #                             "model" :   "claude-3-opus-20240229",
    #                             "seed"          : 131901,
    #                             "temperature"   : 0,
    #                             "relation_type" : "TOPOLOGICAL",
    #                             "system_prompt" : "You are answering to evaluate spatial reasoning ability. You will be presented a question and asked to answer. Where there are multiple possible answers, select the most likely. Answer as briefly as possible, preferring single word answers where they suffice. Where you do not know the answer, it is unanswerable or you are uncertain, return 'ICATQ'."
    #                             },
    #                 "results":{
    #                         "1": {
    #                                 'question'      : "Which country contains the city of Sydney?",
    #                                 'answer'        : "australia",
    #                                 'correct'       : 1,
    #                                 'score'         : 1,
    #                             },
    #                         "2": {
    #                                 'question'      : "Which state or country in Australia contains the city of Sydney?",
    #                                 'answer'        : "new south wales",
    #                                 'correct'       : 1,
    #                                 'score'         : 2,
    #                             }
    #                             }
    #                 }
        
    #     self.assertDictEqual(self.a_tester.run_anthropic_experiment(
    #                                                     filename=test_file,
    #                                                     model="claude-3-opus-20240229",
    #                                                     seed=131901,
    #                                                     temp=0
    #                                                     ), results_dict)

# Main

if __name__=="__main__":
    unittest.main()