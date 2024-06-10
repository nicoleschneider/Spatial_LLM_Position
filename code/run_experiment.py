#Author Kent O'Sullivan

import subprocess
import os

experiments = [ 'toponym.json',
                'topological.json',
                'metric_near-far.json',
                'directional.json',
                'order.json'
               ]


models = {'OPENAI':['gpt-4o','gpt-4-turbo', 'gpt-4', 'gpt-3.5-turbo'], 
        'GOOGLE': ['gemini-1.0-pro','gemini-1.5-flash','gemini-1.5-pro'],
        'ANTHROPIC': ['claude-3-opus-20240229', 'claude-3-sonnet-20240229', 'claude-3-haiku-20240307'],
        'META': ['llama3-70b','llama3-8b', 'mixtral-8x22b-instruct', 'mistral-7b-instruct']}

for experiment in experiments:
    for family in models.keys():
        for model in models[family]:
            command = [
                'python',
                'query_llm.py',
                '--data_directory', os.path.join('..','data'),
                '--results_directory', os.path.join('..','results'),
                '--quiz_file', experiment,
                '--model_family', family,
                '--model', model,
                '--seed',"131901",
                '--temp',"0"
            ]
            print("Running:", command)
            result = subprocess.run(command)