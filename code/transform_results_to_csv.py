#Author: Kent O'Sullivan

# Core Imports
import os
import json
from tqdm import tqdm

# Library Imports
import pandas as pd

# User Imports

# Functions

def get_list_of_files(results_directory:os.path)->list[str]:
    return os.listdir(results_directory)
 

def load_json_from_file(file:os.path)->dict:
    with open(file, 'r') as f:
        data = json.load(f)
    return(data)

def transform_data_to_dict_of_lists(data:dict)->dict:

    relation_type = []
    relation_subtype = []
    model = []
    q_num = []
    answer = []
    correct = []
    unanswered = []
    score = []

    has_line = []
    has_point = []
    has_region = []
    has_indigenous = []
    population = []

    #Metric Only
    example_dist = []
    predicted_dist = []
    ratio = []

    #Load Prompts file

    prompts_file = os.path.join("..","data",data['metadata']['relation_type'.casefold()]+".json")
    prompts = load_json_from_file(prompts_file)

    #Readability
    r = data['results']
    m = data['metadata']
    p = prompts['questions']

    for rr in tqdm(r.keys()):
        #Metadata
        relation_type.append(m['relation_type'])
        model.append(m['model'])
        
        q_num.append(rr)
        correct.append(r[rr]['correct'])
        try:
            score.append(r[rr]['score'])
        except KeyError:
            score.append(0)

        # For metric the predicted answer is stored in 'prediction' rather than 'answer'
        try: 
            if 'icatq' in r[rr]['prediction'].casefold():
                unanswered.append(1)
            else:
                unanswered.append(0)
        except KeyError:
            if 'icatq' in r[rr]['answer'].casefold():
                unanswered.append(1)
            else:
                unanswered.append(0)

        #Handle Metric Only Results
        if 'metric' in m['relation_type'].casefold():
            answer.append(r[rr]['prediction'])
        else:
            answer.append(r[rr]['answer'])
        try:
            example_dist.append(r[rr]['example_dist'])
        except KeyError:
            example_dist.append(None) 
        try:
            predicted_dist.append(r[rr]['predicted_dist'])
        except KeyError:
            predicted_dist.append(None)
        try:
            ratio.append(r[rr]['ratio'])
        except KeyError:
            ratio.append(None)

        if "line" in p[rr]['entity_type'] or "Line" in p[rr]['entity_type']:
            has_line.append(1)
        else:
            has_line.append(0)

        if "point" in p[rr]['entity_type'] or "Point" in p[rr]['entity_type']:
            has_point.append(1)
        else:
            has_point.append(0)

        if "region" in p[rr]['entity_type'] or "Region" in p[rr]['entity_type']:
            has_region.append(1)
        else:
            has_region.append(0)

        has_indigenous.append(p[rr]['indigenous'])

        try:
            if p[rr]['population'] > 0:
                population.append(p[rr]['population'])
            else:
                population.append(None)
        except KeyError:
            population.append(None)

        try:
            relation_subtype.append(p[rr]['sub_relation'])
        except KeyError:
            relation_subtype.append(m['relation_type'])


    results = {
        "relation_type" : relation_type, 
        "relation_subtype" : relation_subtype, 
        "model" : model, 
        "q_num" : q_num, 
        "answer" : answer, 
        "correct" : correct, 
        "unanswered" : unanswered,
        "score" : score, 
        "has_line" : has_line, 
        "has_point" : has_point, 
        "has_region" : has_region, 
        "has_indigenous" : has_indigenous, 
        "population" : population, 
        "example_dist" : example_dist, 
        "predicted_dist" : predicted_dist, 
        "ratio" : ratio, 
    }

    return results

def ingest_all_results(file_list:list[str]):

    all_results = {}

    for file in tqdm(file_list):
        if not file.endswith('.json'):
            continue
        print('loading', file)
        file_to_load=os.path.join(results_directory,file)
        raw_data = load_json_from_file(file_to_load)
        data = transform_data_to_dict_of_lists(raw_data)

        for k in data.keys():
            try:
                all_results[k].extend(data[k])
            except KeyError:
                all_results[k] = data[k]
    
    return all_results

def convert_to_dataframe(data:dict)->pd.DataFrame:
    df = pd.DataFrame(data=data)

    return df

def export_df_to_csv(df:pd.DataFrame, output_dir=os.path.join("..","results"), file_name ="geospatial_reasoning_llm.csv")->None:
    
    try:
        df.to_csv(os.path.join(output_dir,file_name))
        print(f"Saved dataframe as {os.path.join(output_dir,file_name)}")
    except Exception as e:
        print("Error!", e)

    


# Main

if __name__ == "__main__":

    results_directory = os.path.join('..','results')

    file_list = get_list_of_files(results_directory=results_directory)

    formatted = ingest_all_results(file_list=file_list)

    df = convert_to_dataframe(data=formatted)

    export_df_to_csv(df=df)