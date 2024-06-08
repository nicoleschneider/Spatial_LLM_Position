#Author: Kent O'Sullivan // osullik@umd.edu

#Core Imports
import os

#Library Imports
import pandas as pd
import plotnine as p9
from plotnine import ggplot, aes, theme_classic, labs, xlim, ylim, theme, scale_fill_manual,  scale_x_discrete, position_dodge, facet_wrap, after_stat
from plotnine import geom_point, geom_abline, geom_bar, element_text, geom_text, geom_label

#User Imports

#Functions

def load_df_from_csv(source_file:os.path)->pd.DataFrame:
    return (pd.read_csv(source_file))

def generate_metric_scatterplot(df_in:pd.DataFrame, bias=None, remove_outliers=False):

    df_metric = df_in[df_in['relation_type'] == 'METRIC_NEAR-FAR']

    OUTLIER_THRESHOLD = 10000

    if bias:
        df = df_metric[df_metric['bias_term'] == bias]
    else:
        df = df_metric

    #Count the Total, Null and Number of outliers
    num_examples = len(df)
    na_df = df[df['predicted_dist'].isna()]
    na_counts = len(na_df)

    if remove_outliers:
        outlier_df = df[df['predicted_dist'] > OUTLIER_THRESHOLD]
        outlier_counts = len(outlier_df)
        df = df[df['predicted_dist'] <= OUTLIER_THRESHOLD]
    else:
        outlier_counts = 0

    #Create the plot
    scatterplot = (
    ggplot(df, 
           aes(x='example_dist',
                   y='predicted_dist', 
                   color='model')) +
                   geom_point() +
                   theme_classic () + 
                   xlim(0,5000) +
                   ylim(0,5000)+
                   labs (
                       x="Target Distance in KM", 
                       y= 'Predicted Distance in KM') +

    geom_abline(slope=1, intercept=0, linetype='dotted', color='black')
)

    return scatterplot

def generate_toponym_barplot(df_in:pd.DataFrame, admin_area:str):

    df_toponym = df_in[df_in['relation_type'] == 'TOPONYM']

    df_toponym[f'{admin_area}'] = pd.Categorical(df_toponym[f'{admin_area}'], categories=[2,1,0], ordered=True)

    # plot = (ggplot(df_toponym, aes('model', f'factor({admin_area})', fill=f'{admin_area}'))
    # + geom_col())
    # )

    plot = (ggplot(df_toponym, aes(x='model',
                                   fill=f'factor({admin_area})')) +
            geom_bar(position='stack')+
            theme_classic()+
            theme(axis_text_x=element_text(angle=45, hjust=1))+
            labs ( 
                x=f"{admin_area}".replace("_found","").upper(), 
                y= 'Count of Responses',
                fill='Answer' ) +
            scale_fill_manual(values=["#7E4794", "#59A89C","#E25759"], labels=["Abstain", "Correct", "Incorrect"])+
            scale_x_discrete(labels=['Western', 'Indigenous'])
    )
    
    return plot

def generate_topological_bar_plot(df_in:pd.DataFrame,entity_type:str, relation:str):

    print(entity_type,relation)

    df_topological = df_in[df_in['relation_type'] == 'TOPOLOGICAL']
    df_topological = df_topological[df_topological['entity_type'] == entity_type]
    df_topological = df_topological[df_topological['relation_subtype'] == relation]


    df_topological['correct'] = pd.Categorical(df_topological[f'correct'], categories=["incorrect","correct",'abstain'], ordered=True)
    # df_topological['model'] = pd.Categorical(df_topological[f'model'], ordered=True)

    # bar_plot = (ggplot(df_topological, aes('entity_type','factor(correct)',
    #                                        fill='factor(correct)')) +
    #                                        geom_bar(stat='identity', position='stack')
    #                         )
    bar_plot = (ggplot(df_topological, aes(x='model',
                                   fill=f'factor(correct)')) +
            geom_bar(position='stack')+
            theme_classic()+
            theme(axis_text_x=element_text(angle=45, hjust=1))+
            labs ( 
                x="model", 
                y= 'Count of Responses',
                fill='Answer') +
            scale_fill_manual(values=["#7E4794", "#59A89C","#E25759"], labels=["Abstain", "Correct", "Incorrect"])
    )

    return bar_plot
    
def generate_directional_barplot(df_in:pd.DataFrame, n_way:str):

    df_directional = df_in[df_in['relation_type'] == 'DIRECTIONAL']
    df_directional = df_directional[df_directional['n_way'] == n_way]
    
    bar_plot = (ggplot(df_directional, aes(x='model',
                                   fill=f'factor(correct)')) +
            geom_bar(position='stack')+
            theme_classic()+
            theme(axis_text_x=element_text(angle=45, hjust=1))+
            labs ( 
                x="model", 
                y= 'Count of Responses',
                fill='Answer') +
            scale_fill_manual(values=["#7E4794", "#59A89C","#E25759"], labels=["Abstain", "Correct", "Incorrect"])
    )

    return(bar_plot)

def generate_order_barplot(df_in:pd.DataFrame)->ggplot:
    df_order = df_in[df_in['relation_type'] == 'ORDER']
    
    bar_plot = (ggplot(df_order, aes(x='model',
                                   fill=f'factor(correct)')) +
            geom_bar(position='stack')+
            theme_classic()+
            theme(axis_text_x=element_text(angle=45, hjust=1))+
            labs ( 
                x="model", 
                y= 'Count of Responses',
                fill='Answer') +
            scale_fill_manual(values=["#7E4794", "#59A89C","#E25759"], labels=["Abstain", "Correct", "Incorrect"])
    )
    return bar_plot

def generate_bar_plot_of_scores(df_in:pd.DataFrame)->ggplot:

    df = df_in.groupby(['model']).score.sum().reset_index()

    print(df)


    bar_plot = (ggplot(df, aes(x='model',y='score')) +
            geom_bar(stat='identity')+
            theme_classic()+
            theme(axis_text_x=element_text(angle=45, hjust=1))+
            labs ( 
                x="model", 
                y= 'TOTAL SCORE') +
            scale_fill_manual(values=["#7E4794", "#59A89C","#E25759"], labels=["Abstain", "Correct", "Incorrect"])
    )
    return bar_plot

def generate_bar_plot_of_correct_counts(df_in:pd.DataFrame)->ggplot:

    df = df_in

    bar_plot = (ggplot(df, aes(x='model',
                                   fill=f'factor(correct)')) +
            geom_bar(position='stack')+
            theme_classic()+
            theme(axis_text_x=element_text(angle=45, hjust=1))+
            labs ( 
                x="model", 
                y= 'Count of Responses',
                fill='Answer') +
            scale_fill_manual(values=["#7E4794", "#59A89C","#E25759"], labels=["Abstain", "Correct", "Incorrect"])
    )
    return bar_plot

def generate_bar_plot_indigenous(df_in:pd.DataFrame, toponym_only=False)->ggplot:

    df_indigenous = df_in.copy()
    if toponym_only:
        title = "Response Accuracy for Indigenous vs Western Toponym Queries"
        df_indigenous = df_in[df_in['relation_type'] == 'TOPONYM']   
    else:
        title = "Response Accuracy for Indigenous vs Western Non-Toponym Queries"
        df_indigenous = df_in[df_in['relation_type'] != 'TOPONYM']

    df_indigenous['has_indigenous'] = df_indigenous['has_indigenous'].map({0: 'Western', 1: 'Indigenous'})

    df_western = df_indigenous[df_in['has_indigenous'] == 0]
    num_western = len(df_western)
    num_indigenous = len(df_indigenous)-num_western

    print("West", num_western, "Ind", num_indigenous)

    scaling_factor = num_indigenous/num_western

    bar_plot = (ggplot(df_indigenous, aes(x='model', fill='correct')) +
                geom_bar(position='stack', stat='count') +
                theme_classic() +
                theme(axis_text_x=element_text(angle=45, hjust=1)) +
                labs(
                    x="Model",
                    y='Proportion of Responses',
                    fill='Answer'
                ) +
                scale_fill_manual(values=["#7E4794", "#59A89C", "#E25759"], labels=["Abstain", "Correct", "Incorrect"]) +
                facet_wrap('~has_indigenous', scales='free_x')
                
                )
    
    return bar_plot

def generate_scatter_plot_of_cost_per_score(df_in:pd.DataFrame, in_costs:dict, out_costs, output_cost=True)->ggplot:

    df = df_in

    scores = df.groupby('model')['score'].sum().reset_index()

    scores['in_cost'] = scores['model'].map(in_costs)
    scores['out_cost'] = scores['model'].map(out_costs)

    print(scores)

    if not output_cost:
        scatterplot = (
        ggplot(scores, 
            aes(x='score',
                    y='in_cost', 
                    color='model')) +
                    geom_point() +
                    theme_classic () + 
                    labs (
                        x="Score", 
                        y= 'Cost in USD per million input tokens') 
        )
    else:
        scatterplot = (
        ggplot(scores, 
            aes(x='score',
                    y='out_cost', 
                    color='model')) +
                    geom_point() +
                    theme_classic () + 
                    labs (
                        x="Score", 
                        y= 'Cost in USD per million output tokens') 
        )


    return scatterplot



   
def save_plots_as(plot, plot_directory:os.path, filename:str, filetype:str)->None:
    
    try:
        plot.save(os.path.join(plot_directory,f"{filename}.{filetype}"))
        print("Saved to:", os.path.join(plot_directory,f"{filename}.{filetype}"))
    except Exception as e:
        print(f"Unable to save {filename}", e)

#Main

if __name__ == "__main__":
    source_file = os.path.join("..","results","geospatial_reasoning_llm.csv")
    output_directory = os.path.join("..","paper","figures")
    save_format = "svg"

    model_input_costs = {
        'gpt-3.5-turbo':0.50,
        'gpt-4':30.00,
        'gpt-4-turbo':10.00,
        'gpt-4o':5.00,
        'gemini-1.0-pro':0.50,
        'gemini-1.5-flash':0.35,
        'gemini-1.5-pro':3.50,
        'claude-3-opus-20240229':15.00,
        'claude-3-sonnet-20240229':3.00,
        'claude-3-haiku-20240307':0.25,
        'llama-3-70b':3.20,
        'llama-3-8b':1.60,
        'llama-3-70b':3.20,
        'mixtral-8x22b-instruct':3.20,
        'mistral-7b-instruct':1.60
        }

    model_output_costs = {
                        'gpt-3.5-turbo':1.50,
                        'gpt-4':60.00,
                        'gpt-4-turbo':30.00,
                        'gpt-4o':15.00,
                        'gemini-1.0-pro':1.50,
                        'gemini-1.5-flash':1.05,
                        'gemini-1.5-pro':10.50,
                        'claude-3-opus-20240229':75.00,
                        'claude-3-sonnet-20240229':15.00,
                        'claude-3-haiku-20240307':1.25,
                        'llama-3-70b':3.20,
                        'llama-3-8b':1.60,
                        'llama-3-70b':3.20,
                        'mixtral-8x22b-instruct':3.20,
                        'mistral-7b-instruct':1.60
                        }

    df = load_df_from_csv(source_file=source_file)

    plot_list = []

    plot_list.append(("metric_scatter_neutral", generate_metric_scatterplot(df_in=df,
                                                                             bias='neutral',
                                                                             remove_outliers=True)))
    plot_list.append(("metric_scatter_near", generate_metric_scatterplot(df_in=df,
                                                                          bias='near',
                                                                          remove_outliers=True)))
    plot_list.append(("metric_scatter_far", generate_metric_scatterplot(df_in=df,
                                                                         bias='far',
                                                                         remove_outliers=True)))
    plot_list.append(("toponym_bar_country", generate_toponym_barplot(df_in=df,
                                                                      admin_area="country_found")))
    plot_list.append(("toponym_bar_state", generate_toponym_barplot(df_in=df,
                                                                    admin_area="state_found")))

    entities = ['region-region', 'line-line', 'line-region']
    relations = ['Equals','Disjoint','Intersect','Touch', 'Partially Overlap','Within','Contain']

    for relation in relations:
        for entity in entities:
            plot_list.append((f"topological_bar_{relation}_{entity}", generate_topological_bar_plot(df_in=df, entity_type=entity, relation=relation)))

    for way in ['2-way','3-way']:
        plot_list.append((f"directional_bar_{way}", generate_directional_barplot(df_in=df, n_way=way)))


    plot_list.append(("order_bar", generate_order_barplot(df_in=df)))

    plot_list.append(("total_scores", generate_bar_plot_of_scores(df_in=df)))
    plot_list.append(("total_correct", generate_bar_plot_of_correct_counts(df_in=df)))

    plot_list.append(('indigenous_names_toponym_only', generate_bar_plot_indigenous(df_in=df, toponym_only=True)))
    plot_list.append(('indigenous_names_not_toponym', generate_bar_plot_indigenous(df_in=df, toponym_only=False)))

    plot_list.append(('scatter_model_cost_per_score_input', generate_scatter_plot_of_cost_per_score(df_in=df, in_costs=model_input_costs,
                                                                                              out_costs=model_output_costs, output_cost=False)))
    plot_list.append(('scatter_model_cost_per_score_output', generate_scatter_plot_of_cost_per_score(df_in=df, in_costs=model_input_costs,
                                                                                              out_costs=model_output_costs, output_cost=True)))

    for plot in plot_list:
        save_plots_as(plot=plot[1], plot_directory=output_directory,filename=plot[0],filetype=save_format)