#Author: Kent O'Sulivan // osullik@umd.edu

# Core Imports
import os
import json
import argparse
from tqdm import tqdm

# Library Imports
import openai

# User Imports

# Classes

class Spatial_LLM_Tester():
    def __init__(self, 
                 api_var_name:str = "OAI_API",
                 data_directory:os.path = os.path.join("..","data"),
                 results_directory:os.path = os.path.join("..","results")):
        
        self._oai_api_key = self.get_api_key_from_environ_var(var_name=api_var_name)
        self.oai_client = openai.OpenAI(api_key=self._oai_api_key)
        self._data_directory = self.set_data_directory(data_directory=data_directory)
        self.experiment_file = {}
        self._system_prompt = ""
        self._results_directory = self.set_results_directory(results_directory=results_directory)

        
    
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
    
    def set_results_directory(self, results_directory:os.path)->None:
        if not os.path.exists(results_directory):
            print(f'{results_directory} not found, creating...')
            os.makedirs(results_directory)
        self._results_directory = results_directory
        return results_directory
    
    def get_results_directory(self)->os.path:
        return self._results_directory
    
    def set_system_prompt(self, system_prompt:str)->None:
        self._system_prompt = system_prompt

    def get_system_prompt(self)->str:
        return(self._system_prompt)
    
    def load_question_file_to_dict(self, filename:str)->dict:
        file_path = os.path.join(self.get_data_directory(), filename)
        print(f"Trying to load from {file_path}...")
        try:
            with open(file_path, 'r') as f:
                experiment_dict = json.load(f)
                print("Success!")
        except:
            exit(f"Unable to Load {file_path}")
        
        self.experiment_file = experiment_dict 
        return experiment_dict
    
    def get_filename(self)->str:
        return(self.experiment_file['metadata']['file_name'])
    
    def get_relation(self)->str:
        return(self.experiment_file['metadata']['relation_type'])
    
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
                    'question'      : question,
                    'answer'        : chat_completion.choices[0].message.content.casefold()
                }
        
        return(result)
    
    def ask_multiple_questions(self,    questions:dict, 
                                        model:str="gpt-3.5-turbo",
                                        seed:int=131901,
                                        temp:int=0)->dict:
        results = {}
        print(f"Running Experiment querying {model} API")
        for question in tqdm(questions.keys()):
            q = questions[question]['question']
            a = self.ask_single_question(question=q, 
                                         model=model, 
                                         seed=seed,
                                         temp=temp)
            results[question] = a

        return results
    
    def evaluate_answer(self, gt_answers:list[str], pred_answer:str):

        lc_dict = {}
        for k,v in gt_answers.items():
            lc_dict[k.casefold()] = v

        if pred_answer.casefold() in lc_dict:
            return(lc_dict[pred_answer.casefold()])
        else:
            return 0

    def evaluate_all_answers(self, gt_answers:dict, results:dict)->dict:

        print(f"Evaluating the answers...")
        for result in tqdm(results.keys()):
            score = self.evaluate_answer(gt_answers=gt_answers[result]['answers'], 
                                    pred_answer=results[result]['answer'])
            if score > 0:
                results[result]['correct'] = 1
            else:
                results[result]['correct'] = 0
            
            results[result]['score'] = score
        
        return results
    
    def run_experiment(self, filename:os.path, model:str="gpt-3.5-turbo", seed:int=131901, temp:int=0)->dict:

        experiment_dict = self.load_question_file_to_dict(filename=os.path.join(self.get_data_directory(), filename))
        
        self.set_system_prompt(experiment_dict['metadata']['system_prompt'])

        results = self.ask_multiple_questions(questions=experiment_dict['questions'],model=model,seed=seed,temp=temp)

        evaluated = self.evaluate_all_answers(gt_answers=experiment_dict['questions'], results=results)

        to_return = {
                        "metadata":{
                            "model": model, 
                            "seed":seed,
                            "temperature":temp,
                            "relation_type" : experiment_dict['metadata']['relation_type'],
                            "system_prompt" : experiment_dict['metadata']['system_prompt']
                            },
                        "results": evaluated
                    }

        return to_return
    
    def save_results_to_file(self, results):

        filename = f"results_{results['metadata']['model']}_{results['metadata']['relation_type']}.json"

        filepath = os.path.join(self.get_results_directory(),filename)

        print(f"Trying to save to {filepath}")

        try:
            with open(filepath, "w") as f:
                print(f'Saved to {filepath}')
                json.dump(results,f,indent=3)
        except Exception as e:
            print("Unable to save to File -- check and try again", e)

# Main

if __name__ == '__main__':

    argparser = argparse.ArgumentParser()

    argparser.add_argument("--data_directory", 
                           help="path to directory that question files are located", 
                           type=str,
                           required=False,
                           default="../data")
    argparser.add_argument("--results_directory",
                           help="directory for the results to be saved", 
                           type=str, 
                           required=False, 
                           default="../results")
    argparser.add_argument("--quiz_file",
                           help="File name of the quiz file to test, including extension",
                           type=str,
                           required=True)
    argparser.add_argument("--model", 
                           help="Model to use from ['gpt-4o','gpt-4-turbo', 'gpt-4', 'gpt-3.5-turbo']",
                           type=str, 
                           required=True)
    argparser.add_argument("--seed",
                           help='Seed to use to improve reproducibility of results. Integer', 
                           type=int, 
                           required=False, 
                           default=131901)
    argparser.add_argument("--temp", 
                           help="set the temperature of the model. 0 is low, 2 is high", 
                           type=int, 
                           required=False, 
                           default=0)
    
    flags = argparser.parse_args()

    tester = Spatial_LLM_Tester(data_directory=flags.data_directory,
                                results_directory=flags.results_directory)
    
    results = tester.run_experiment(filename=flags.quiz_file,
                                    model=flags.model,
                                    seed=flags.seed,
                                    temp=flags.temp)
    
    tester.save_results_to_file(results=results)
    
# Run a test on topological contains using gpt-3.5-turbo:
# python query_llm.py --quiz_file topological_contains.json --model gpt-3.5-turbo