#Author: Kent O'Sullivan // osullik@umd.edu

#Core Imports
import os

#Library Imports
import pandas as pd
import plotnine as p9
from plotnine import ggplot, aes, theme_classic, labs, xlim, ylim, theme, scale_fill_manual, position_dodge, facet_wrap
from plotnine import geom_point, geom_abline, geom_bar, element_text, geom_col

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
                       title=f'Bias Term: {bias}, {na_counts}/{num_examples} excluded as Null, {outlier_counts} outliers excluded', 
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
            scale_fill_manual(values=["#7E4794", "#59A89C","#E25759"], labels=["Abstain", "Correct", "Incorrect"])
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
                title=f"{entity_type.upper()}_{relation.upper()}",
                x="model", 
                y= 'Count of Responses',
                fill='Answer') +
            scale_fill_manual(values=["#7E4794", "#59A89C","#E25759"], labels=["Abstain", "Correct", "Incorrect"])
    )

    return bar_plot
    
    
    

   
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


    for plot in plot_list:
        save_plots_as(plot=plot[1], plot_directory=output_directory,filename=plot[0],filetype=save_format)