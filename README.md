# Towards Geospatial Reasoning in Large Language Models

Submitted to SIGSPATIAL 2024

## Abstract

Spatial reasoning is a particularly challenging form of reasoning that requires inferring implicit information about objects based on their relative positions in space.
Traditionally, spatial reasoning is addressed using formal methods that rely on pre-computed indices and data structures, which limit the scope of questions that can be answered.
As the research community moves towards developing general purpose geo-foundation models that can perform a variety of spatial reasoning tasks, recent research has explored what kinds of world knowledge and spatial reasoning capabilities Large Language Models (LLMs) naturally inherit from their training data.
In this paper we assess the spatial reasoning ability of LLMs through a set of experiments designed to cover a broad range of spatial tasks, including toponym resolution, and reasoning about four fundamental spatial relations: metric, directional, topological, and order relationships.
While previous work has demonstrated that LLMs have some basic level of spatial awareness in the form of knowledge about geocoordinates, directional relationships between major cities, and distances between cities, we find that increasing the complexity of spatial tasks to include more than two entities and expanding the coverage to address all four major forms of spatial relations reveals significant gaps in the spatial reasoning abilities of the LLMs tested.
Given these findings, we propose several avenues of opportunity to improve the spatial reasoning ability of LLMs.

## Setting up the environment: 

*The following instructions assume a linux-like system*

Clone the repo:

    git clone https://github.com/nicoleschneider/Spatial_LLM_Position.git

Change into the repo: 

    cd Spatial_LLM_Position/

Create a virtual environment

    python -m venv spatial_llm_env

Activate the environment:

    source spatial_llm_env/bin/activate

Install required libraries

    pip install -r requirements.txt


## Replicating Experiments

The replication is designed to flow from least work to most work. LLMs are inherently non-deterministic, so replication `from scratch' will likely yield slightly different results. 

### 1. Recreating Figures

To use our existing data to regenerate the figures used in the picture from [the CSV](https://github.com/nicoleschneider/Spatial_LLM_Position/blob/main/results/geospatial_reasoning_llm.csv) we generate from our raw results:

Change into the code directory

    cd code

Run the visualization script: 

    python generate_viz.py

*note: if you change the location or name of the csv, you will need to update lines 275. If you want to output to a custom location, you will need to update line 276 (first two lines of the main function):*

        **275:** source_file = os.path.join("..","results","geospatial_reasoning_llm.csv")
        **276:** output_directory = os.path.join("..","paper","figures")

## 2. Analyzing Data

To regenerate [the CSV](https://github.com/nicoleschneider/Spatial_LLM_Position/blob/main/results/geospatial_reasoning_llm.csv) from our [raw results](https://github.com/nicoleschneider/Spatial_LLM_Position/tree/main/results):

Change into the code directory if not already in it: 

    cd ~/Spatial_LLM_Position/code

Run the analysis script: 

    python transform_results_to_csv.py

*note: If you have re-run the data collection and outputted it to a different location you will need to update the main function (line 286) to point to the directory with the results json files*

    **286:** results_directory = os.path.join('..','results')

## Collecting Data

To reproduce the experiments from scratch you must: 

Gain access to the [OpenAI](https://openai.com/index/openai-api/), [Google](https://ai.google.dev/), [Anthropic](https://www.anthropic.com/api) and [LLama AI (for Llama and Mistral)](https://www.llama-api.com/) APIs and note your API keys. 

Set the relevant environment variables

    export OAI_API="<your openAI API Key>"
    export GEM_API="<your Google API Key>"
    export ANT_API="<your Anthropic API Key>"
    export MET_API="<your LLamaAI API Key>"

Run the same experiments: 

    python run_experiment.py

*note: if you don't want to use the default [results directory](https://github.com/nicoleschneider/Spatial_LLM_Position/tree/main/results) you will need to modify the parameters in the main experiment loop*


## Running a single experiment: 

If you would just like to run a single experiment use: 

    python query_llm.py --quiz_file <quiz_file.json> --model_family <choose from: `OPENAI', `GOOGLE', `ANTHROPIC', or `META'> --model <chosen_model_here>

For example, running the toponym test on gpt3 using the default directories is: 

    python query_llm.py --quiz_file topological_contains.json --model_family OPENAI --model gpt-3.5-turbo

you can also specify the parameters: 

    --data_directory

    --resuts_directory

    --seed

    --temp


## Recompile the Paper: 

To recompile the LaTeX for the paper:

simply run the following from the root of the papers directory:

    make main.pdf

You'll notice a lot of random intermediary files get made. To get rid of them and leave your PDF use:

    make publish

Sometimes everything goes wrong and we need to start again, when that happens use: 

    make clean

Detain on how to configure a PDFLatex compilation environment are [available here](https://github.com/osullik/project_template)