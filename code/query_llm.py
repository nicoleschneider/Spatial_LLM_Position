#Author: Kent O'Sulivan // osullik@umd.edu

# Core Imports
import os
import json

# Library Imports
import openai

# User Imports

# Classes

class Spatial_LLM_Tester():
    def __init__(self, 
                 api_var_name:str = "OAI_API",
                 data_directory:os.path = os.path.join("..","..","data")):
        
        self._oai_api_key = self.get_api_key_from_environ_var(var_name=api_var_name)
        self.oai_client = openai.OpenAI(api_key=self._oai_api_key)
        self._data_directory = self.set_data_directory(data_directory=data_directory)
        self.experiment_file = {}
        self._system_prompt = ""

        
    
    def get_api_key_from_environ_var(self, var_name:str):
        try: 
            key= os.environ[var_name]
        except KeyError:
            exit("API Key not found, ensure you run 'echo OAI_API=<your_api_key>' in your shell and try again")

        return key
        
    def check_oai_api_key_valid(self):
        try:
            chat_completion = self.oai_client.chat.completions.create(
            messages=[
                {
                    "role": "user",
                    "content": "Say this is a test",
                }
            ],
            model="gpt-3.5-turbo",
            seed=131901,            #Setting seed makes result sampling consistent
            temperature=0           #Setting temprature to 0 minimizes randomness in result.
            )
        except openai.APIConnectionError as e:
            return str(e.status_code)
        
        return chat_completion
    
    def set_data_directory(self, data_directory:os.path)->None:
        if not os.path.exists(data_directory):
            print(f'{data_directory} not found, creating...')
            os.makedirs(data_directory)
        self._data_directory = data_directory
        return data_directory
    
    def get_data_directory(self)->os.path:
        return self._data_directory
    
    def load_question_file_to_dict(self, filename:str)->dict:

        file_path = os.path.join(self.get_data_directory(), filename)
        with open(file_path, 'r') as f:
            experiment_dict = json.load(f)
        
        self.experiment_file = experiment_dict 
        return experiment_dict
    
    def get_filename(self)->str:
        return(self.experiment_file['metadata']['file_name'])
    
    def get_relation(self)->str:
        return(self.experiment_file['metadata']['realation_type'])
    
    def ask_single_question(self, question:str, 
                            model="gpt-3.5-turbo", 
                            seed:int=131901, 
                            temp:int=0)->str:
        
        chat_completion = self.oai_client.chat.completions.create(
            messages=[
                        {
                            "role": "system", 
                            "content": f"{self._system_prompt}"},
                        {
                            "role": "user", 
                            "content": f"{question}"}
                    ],
            model=model,          #Set the model to use
            seed=seed,            #Setting seed makes result sampling consistent
            temperature=temp      #Setting temprature to 0 minimizes randomness in result.
            )
        
        result = {
                    'model'         : model,
                    'seed'          : seed,
                    'temperature'   : temp,
                    'question'      : question,
                    'answer'        : chat_completion.choices[0].message.content.casefold()
                }
        
        return(result)
    
    def ask_multiple_questions(self,    questions:dict, 
                                        model:str="gpt-3.5-turbo",
                                        seed:int=131901,
                                        temp:int=0)->dict:
        results = {}
        for question in questions.keys():
            q = questions[question]['question']
            a = self.ask_single_question(question=q, 
                                         model=model, 
                                         seed=seed,
                                         temp=temp)
            results[question] = a

        return results
    
    def evaluate_answer(self, gt_answers:list[str], pred_answer:str):

        lc_answers = [a.casefold() for a in gt_answers]

        return(pred_answer.casefold() in lc_answers)

    def evaluate_all_answers(self, gt_answers:dict, results:dict)->dict:

        for result in results.keys():
            if self.evaluate_answer(gt_answers=gt_answers[result]['answers'], 
                                    pred_answer=results[result]['answer']):
                results[result]['correct'] = 1
            else:
                results[result]['correct'] = 0
        
        return results

# Main