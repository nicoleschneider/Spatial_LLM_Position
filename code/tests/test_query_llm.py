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

class Test_spatial_llm(unittest.TestCase):
    
    def setUp(self) -> None:
        self.data_directory = os.path.join('..','..','data','test_data')
        self.tester = Spatial_LLM_Tester(data_directory=self.data_directory)
        return super().setUp()
    
    def tearDown(self) -> None:
        return super().tearDown()
    
    def test_tester_exists(self):
        self.assertTrue(self.tester)


    def test_get_api_key(self):

        fake_api_key = "FAKE_OPENAI_KEY"
        os.environ['fake_api_key'] = fake_api_key
        self.assertEqual(self.tester.get_api_key_from_environ_var(var_name="fake_api_key"),fake_api_key)

    def test_api_key_not_exist(self):

        wrong_api_key = "DOES_NOT_EXIST"

        with self.assertRaises(SystemExit):
            self.tester.get_api_key_from_environ_var(var_name="wrong_api_key")

    def test_api_key_valid(self):

        self.assertTrue(self.tester.check_oai_api_key_valid() not in ["400", "401"])

    def test_load_query_json(self):
        test_file = "test_query_file.json"
        test_filename = "test_query_file"
        test_relation = 'TOPOLOGICAL'
        
        #Check that we're looking in the correct directory and that we can read the file
        self.assertEqual(self.tester.get_data_directory(),self.data_directory)
        self.tester.load_question_file_to_dict(filename=test_file)

        #Check params that are only set after file is successfully loaded
        self.assertEqual(self.tester.get_filename(), test_filename)
        self.assertEqual(self.tester.get_relation(), test_relation) 

    def test_ask_question(self):
        self.tester._system_prompt = "This is for a unit test, respond to the user question with the word 'bananas' no matter what the question they ask is."
        test_question = "What is the capital city of Australia?"

        test_result = {
                    'model'         : "gpt-3.5-turbo",
                    'seed'          : 131901,
                    'temperature'   : 0,
                    'question'      : "What is the capital city of Australia?",
                    'answer'        : "bananas"
                }

        result = self.tester.ask_single_question(question=test_question)
        self.assertDictEqual(result, test_result)

    def test_multiple_questions(self):
        self.tester._system_prompt = "This is for a unit test, respond to the user question with the word 'bananas' no matter what the question they ask is."

        test_questions =   {
                            "1":{
                                "question":"What is the capital city of Australia?",
                            },
                            "2":{
                                "question":"What is the capital city of New Zealand?",
                                }
                            }

        
        merged_dict = {
            "1": {
                    'model'         : "gpt-3.5-turbo",
                    'seed'          : 131901,
                    'temperature'   : 0,
                    'question'      : "What is the capital city of Australia?",
                    'answer'        : "bananas"
                },
            "2": {
                    'model'         : "gpt-3.5-turbo",
                    'seed'          : 131901,
                    'temperature'   : 0,
                    'question'      : "What is the capital city of New Zealand?",
                    'answer'        : "bananas"
                }
        }

        self.assertDictEqual(self.tester.ask_multiple_questions(questions=test_questions), merged_dict)

    def test_evaluate_answer(self):
        
        test_predicted_answer_lc = "bananas"
        test_predicted_answer_sc = "Bananas"
        test_predicted_answer_uc = "BANANAS"
        test_wrong_answer_lc = "strawberries"
        test_wrong_answer_sc = "Strawberries"
        test_wrong_answer_uc = "STRAWBERRIES"

        test_gt_answers = ['banana', 'bananas']

        self.assertTrue(self.tester.evaluate_answer(gt_answers=test_gt_answers,
                                                    pred_answer=test_predicted_answer_lc))
        self.assertTrue(self.tester.evaluate_answer(gt_answers=test_gt_answers,
                                                    pred_answer=test_predicted_answer_sc))
        self.assertTrue(self.tester.evaluate_answer(gt_answers=test_gt_answers,
                                                    pred_answer=test_predicted_answer_uc))
        self.assertFalse(self.tester.evaluate_answer(gt_answers=test_gt_answers,
                                                    pred_answer=test_wrong_answer_lc))
        self.assertFalse(self.tester.evaluate_answer(gt_answers=test_gt_answers,
                                                    pred_answer=test_wrong_answer_sc))
        self.assertFalse(self.tester.evaluate_answer(gt_answers=test_gt_answers,
                                                    pred_answer=test_wrong_answer_uc))

    def test_evaluate_all_answers(self):

        test_results = {
            "1": {
                    'model'         : "gpt-3.5-turbo",
                    'seed'          : 131901,
                    'temperature'   : 0,
                    'question'      : "What is the capital city of Australia?",
                    'answer'        : "bananas"
                },
            "2": {
                    'model'         : "gpt-3.5-turbo",
                    'seed'          : 131901,
                    'temperature'   : 0,
                    'question'      : "What is the capital city of New Zealand?",
                    'answer'        : "strawberries"
                }
        }

        evaluated_results = {
            "1": {
                    'model'         : "gpt-3.5-turbo",
                    'seed'          : 131901,
                    'temperature'   : 0,
                    'question'      : "What is the capital city of Australia?",
                    'answer'        : "bananas",
                    'correct'       : 1
                },
            "2": {
                    'model'         : "gpt-3.5-turbo",
                    'seed'          : 131901,
                    'temperature'   : 0,
                    'question'      : "What is the capital city of New Zealand?",
                    'answer'        : "strawberries",
                    'correct'       : 0
                }
        }

        test_gt_answers = {
                            "1":{
                                "answers":["banana", "bananas"]
                            },
                            "2":{
                                "answers":["cherry", "cherries"]
                            }
                        }
        

        
        self.assertDictEqual(self.tester.evaluate_all_answers(gt_answers = test_gt_answers,
                                                                results=test_results), evaluated_results)


# Main

if __name__=="__main__":
    unittest.main()