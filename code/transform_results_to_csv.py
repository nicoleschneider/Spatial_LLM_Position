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
    bias_term = []

    #Toponym Only
    country_found = []
    state_found = []

    #Topological Only
    entity_type = []

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

        if (r[rr]['correct']) == 0 and r[rr]['answer'].replace(".","").casefold() == "icatq":
            correct.append("abstain")
        elif (r[rr]['correct']) == 0 and r[rr]['answer'].replace(".","").casefold() != "icatq":
            correct.append("incorrect")
        elif r[rr]['correct'] == 1:
            correct.append('correct')

        
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
        
        if 'metric' in m['relation_type'].casefold():
            if 'near' in p[rr]['question'].casefold():
                bias_term.append('near')
            elif 'far' in p[rr]['question'].casefold():
                bias_term.append('far')
            elif 'similar' in p[rr]['question'].casefold():
                bias_term.append('neutral')
        else:
            bias_term.append(None)
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


        #Toponym:
        if m['relation_type'] == "TOPONYM":
            if r[rr]['answer'].replace(".","") == "icatq" or r[rr]['answer'].replace(".","") == "ICATQ":
                country_found.append(2)
                state_found.append(2)
            else:
                try:
                    if 'country' in r[rr]['admin_levels']:
                        country_found.append(1)
                    else:
                        country_found.append(0)
                except KeyError:
                    country_found.append(0)
                try:
                    if 'state' in r[rr]['admin_levels']:
                        state_found.append(1)
                    else:
                        state_found.append(0)
                except KeyError:
                    state_found.append(0)
        else:
            country_found.append(0)
            state_found.append(0)
            
        # Topological

        if m['relation_type'] == "TOPOLOGICAL":
            entity_string = None
            lc = []
            for x in p[rr]['entity_type']:
                lc.append(x.casefold())

            #Get in consistent order, and assign relations based on composite types
            lc.sort()
            if lc is None:
                entity_string = None
            elif len(lc) == 1:
                entity_string=f"{lc[0]}-{lc[0]}"
            elif len(lc) == 2:
                entity_string=f"{lc[0]}-{lc[1]}"
            elif len(lc) == 3:
                entity_string=f"{lc[0]}-{lc[1]}-{lc[2]}"
            else:
                entity_string = None

        else:
            entity_string = None       
        
        print("ES:",entity_string)
        entity_type.append(entity_string) 

                

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
            "bias_term": bias_term,
            "country_found":country_found,
            "state_found":state_found,
            "entity_type":entity_type
    }
        
    print(len(model))
    print(len(entity_type))

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