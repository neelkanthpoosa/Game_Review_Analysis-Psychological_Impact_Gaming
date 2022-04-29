from distutils.command.config import config
import plotly.express as px
import dash_core_components as dcc
import dash_html_components as html
import plotly.graph_objs as go
import pandas as pd
import numpy as np
import dash
import dash_table
from dash_table.Format import Format, Group, Scheme
import dash_table.FormatTemplate as FormatTemplate
from datetime import datetime as dt
from app import app
import os
import re
import json
####################################################################################################
# 000 - FORMATTING INFO
####################################################################################################

####################### diva css formatting
diva_colors = {
    'dark-blue-grey' : 'rgb(62, 64, 76)',
    'medium-blue-grey' : 'rgb(77, 79, 91)',
    'superdark-green' : 'rgb(41, 56, 55)',
    'dark-green' : 'rgb(57, 81, 85)',
    'medium-green' : 'rgb(93, 113, 120)',
    'light-green' : 'rgb(224, 158, 2)',
    'pink-red' : 'rgb(251, 251, 252)',
    'dark-pink-red' : 'rgb(224, 72, 2)',
    'white' : 'rgb(251, 251, 252)',
    'light-grey' : 'rgb(208, 206, 206)'
}

externalgraph_rowstyling = {
    'margin-left' : '15px',
    'margin-right' : '15px'
}

externalgraph_colstyling = {
    'border-radius' : '10px',
    'border-style' : 'solid',
    'border-width' : '1px',
    'border-color' : diva_colors['superdark-green'],
    'background-color' : diva_colors['superdark-green'],
    'box-shadow' : '0px 0px 17px 0px rgba(186, 218, 212, .5)',
    'padding-top' : '10px'
}

filterdiv_borderstyling = {
    'border-radius' : '0px 0px 10px 10px',
    'border-style' : 'solid',
    'border-width' : '1px',
    'border-color' : diva_colors['light-green'],
    'background-color' : diva_colors['light-green'],
    'box-shadow' : '2px 5px 5px 1px rgba(255, 101, 131, .5)'
    }

navbarcurrentpage = {
    'text-decoration' : 'bold',
    'text-decoration-color' : diva_colors['pink-red'],
    'text-shadow': '0px 0px 1px rgb(251, 251, 252)'
    }

recapdiv = {
    'border-radius' : '10px',
    'border-style' : 'solid',
    'border-width' : '1px',
    'border-color' : 'rgb(251, 251, 252, 0.1)',
    'margin-left' : '15px',
    'margin-right' : '15px',
    'margin-top' : '15px',
    'margin-bottom' : '15px',
    'padding-top' : '5px',
    'padding-bottom' : '5px',
    'background-color' : 'rgb(251, 251, 252, 0.1)'
    }

recapdiv_text = {
    'text-align' : 'left',
    'font-weight' : '350',
    'color' : diva_colors['white'],
    'font-size' : '1.5rem',
    'letter-spacing' : '0.04em'
    }

####################### diva chart formatting

diva_title = {
    'font' : {
        'size' : 16,
        'color' : diva_colors['white']}
}

diva_xaxis = {
    'showgrid' : False,
    # 'gridwidth':100,
    'linecolor' : diva_colors['light-grey'],
    'color' : diva_colors['light-grey'],
    'tickangle' : 315,
    'titlefont' : {
        'size' : 12,
        'color' : diva_colors['light-grey']},
    'tickfont' : {
        'size' : 11,
        'color' : diva_colors['light-grey']},
    'zeroline': False
}

diva_yaxis = {
    'showgrid' : True,
    'color' : diva_colors['light-grey'],
    'gridwidth' : 0.5,
    'gridcolor' : diva_colors['dark-green'],
    'linecolor' : diva_colors['light-grey'],
    'titlefont' : {
        'size' : 12,
        'color' : diva_colors['light-grey']},
    'tickfont' : {
        'size' : 11,
        'color' : diva_colors['light-grey']},
    'zeroline': False
}

diva_font_family = 'Dosis'

diva_legend = {
    'orientation' : 'h',
    'yanchor' : 'bottom',
    'y' : 1.01,
    'xanchor' : 'right',
    'x' : 1.05,
	'font' : {'size' : 9, 'color' : diva_colors['light-grey']}
} # Legend will be on the top right, above the graph, horizontally

diva_margins = {'l' : 5, 'r' : 5, 't' : 45, 'b' : 15}  # Set top margin to in case there is a legend

diva_layout = go.Layout(
    font = {'family' : diva_font_family},
    title = diva_title,
    title_x = 0.5, # Align chart title to center
    paper_bgcolor = 'rgba(0,0,0,0)',
    plot_bgcolor = 'rgba(0,0,0,0)',
    xaxis = diva_xaxis,
    yaxis = diva_yaxis,
    height = 270,
    legend = diva_legend,
    margin = diva_margins
    )


####################################################################################################
# 000 - IMPORT DATA
####################################################################################################

####################################################################################################
# 000 - REVIEW ANALYSIS
####################################################################################################


@app.callback(
    dash.dependencies.Output('rv-analysis', 'figure'),
	[dash.dependencies.Input('game-list', 'value')])
def update_chart(value):

    # Filter df according to selection
    df=pd.read_csv(value)
    # Build graph
    data = go.Bar(
        x = df["Aspects"],
        y = df['SentimentScore'],
        marker = {'color': '#ffa343',
                'opacity' : 0.9},
        hovertemplate = (
        "<i style='color:'>Feature Discussed</i>: <b style='color:'>%{x:|%a}</b><br>"+
        "<i style='color:'>Sentiment Score</i>: <b style='color:'>%{y}</b>"+
        "<extra></extra>")
                )
    fig = go.Figure(data=data, layout=diva_layout)
    fig.update_layout(
        title={'text' : "Most Discussed Aspects"},
        xaxis = {'title' : "Aspects", 'tickangle' : 0},
        yaxis = {
            'title' : "Sentiment Score",
            'tickformat' : ".0%"},
        hovermode="x"

        )
    fig.update_xaxes(tickangle = 0)

    return fig

################################################################################################
# DATA LOADING FOR PSYCH ANALYSIS
################################################################################################
import map
token=map.token
####### Read local json file and parse it #iso_a3 #custom.geo
with open('data/custom.geo.json', 'r') as myfile:
		data=myfile.read()
   
gj = json.loads(data) #parse json file. This file is GeoJSON formatted
data_f = pd.read_csv('data/FinalGaming_data_cleaned.csv')

data_f.rename(columns = {"Narcissism_Label":'Narcissism Level'}, inplace = True)
data_f.rename(columns = {"GAD_T_Label":'Anxiety Level'}, inplace = True)
data_f.rename(columns = {"SPIN_T_Label":'Social Phobia Level'}, inplace = True)
data_f.rename(columns = {"SWL_T_Label":'Satisfaction with Life Level'}, inplace = True)


game_n=[i for i in data_f["Game"].unique()]
option_game = [
    {'label' : k, 'value' : k} for k in (game_n)
    ]

####################################################################################################
# CHOROPLETH MAP
####################################################################################################

@app.callback(
    dash.dependencies.Output('choropleth', 'figure'),
    [dash.dependencies.Input('my-dropdown', 'value'),
    dash.dependencies.Input('my-dropdown-game', 'value')]
)
def update_choropleth(value,glist):
    dff = data_f.copy()
    if(len(glist)>0):
        dff=dff[dff["Game"].isin(glist)]
    else:
        dff=data_f.copy()

    if re.search( "Narcissism", value ):
        color_code = "Narcissism_Level"
    elif re.search("Anxiety", value):
        color_code = "GAD_T_Level"
    elif re.search("Social", value):
        color_code = "SPIN_T_Level"
    elif re.search("Satisfaction", value):
        color_code = "SWL_T_Level"

    dff = dff.groupby([color_code,"Residence_ISO3"], as_index=False)[[value]].count()
    dff[color_code] = dff[color_code].astype(str)

    if re.search( "Narcissism_Level", color_code):
        for index, row in dff.iterrows():
            if row[color_code] == "0":
                label = "None"
            elif row[color_code] == "1":
                label = "Mild"
            elif row[color_code] == "2":
                label = "Moderate"
            elif row[color_code] == "3":
                label = "Severe"
            elif row[color_code] == "4":
                label = "Very Severe"
            dff.at[index,color_code] = label

    elif re.search("GAD_T_Level", color_code):
        for index, row in dff.iterrows():
            if row[color_code] == "0":
                label = "Minimal"
            elif row[color_code] == "1":
                label = "Mild"
            elif row[color_code] == "2":
                label = "Moderate"
            elif row[color_code] == "3":
                label = "Severe"
            dff.at[index,color_code] = label

    elif re.search("SPIN_T_Level", color_code):
        for index, row in dff.iterrows():
            if row[color_code] == "0":
                label = "None"
            elif row[color_code] == "1":
                label = "Mild"
            elif row[color_code] == "2":
                label = "Moderate"
            elif row[color_code] == "3":
                label = "Severe"
            elif row[color_code] == "4":
                label = "Very Severe"
            dff.at[index,color_code] = label
            
    elif re.search("SWL_T_Level", color_code):
        for index, row in dff.iterrows():
            if row[color_code] == "0":
                label = "Extremely Dissatisfied"
            elif row[color_code] == "1":
                label = "Dissatisfied"
            elif row[color_code] == "2":
                label = "Slightly Dissatisfied"
            elif row[color_code] == "3":
                label = "Neutral"
            elif row[color_code] == "4":
                label = "Slightly Satisfied"
            elif row[color_code] == "5":
                label = "Satisfied"
            elif row[color_code] == "6":
                label = "Extremely Satisfied"

            dff.at[index,color_code] = label


    dff.columns.values[1] = "Country"
    dff.columns.values[0] = "Level"
    dff.rename(columns = {str(value):'Count'}, inplace = True)
    df_country = data_f[['Residence_ISO3', 'Residence']]
    df_country.drop_duplicates(inplace = True)
    df_country_dict = dict(zip(df_country.Residence_ISO3, df_country.Residence))
    dff['Residence']=dff.Country.astype(str).map(df_country_dict)

    plot = px.choropleth_mapbox(data_frame=dff, locations="Country", color="Count",
                                animation_frame="Level",         
                                color_continuous_scale = "OrRd",
                                range_color=[1,20],geojson=gj, 
                                featureidkey="properties.iso_a3",
                                hover_name='Residence',
                                hover_data= {'Country':False},                       
                                center={"lat": 35, "lon": 1}, 
                                zoom=-0.2,
                                opacity = 0.7,

                                )
    plot.layout.update(dragmode=False)

    plot.update_layout(
    mapbox_style="white-bg",
    mapbox_layers=[
        {
            "below": 'traces',
            "sourcetype": "raster",
            "sourceattribution": "United States Geological Survey",
            "source": [
                "https://basemap.nationalmap.gov/arcgis/rest/services/USGSImageryOnly/MapServer/tile/{z}/{y}/{x}"
            ]
        },
        {
            "sourcetype": "raster",
            "sourceattribution": "Government of Canada",
            "source": ["https://geo.weather.gc.ca/geomet/?"
                       "SERVICE=WMS&VERSION=1.3.0&REQUEST=GetMap&BBOX={bbox-epsg-3857}&CRS=EPSG:3857"
                       "&WIDTH=1000&HEIGHT=1000&LAYERS=RADAR_1KM_RDBR&TILED=true&FORMAT=image/png"],
        }
      ],margin={"r":0,"t":20,"l":0,"b":0})

    plot.layout.xaxis.fixedrange = True
    plot.layout.yaxis.fixedrange = True
    
    plot.update_layout( mapbox_accesstoken=token,margin={"r":0,"t":0,"l":0,"b":0},xaxis={'fixedrange':True})
    return(plot)

########################################################################################################################################################################################################

@app.callback(
    dash.dependencies.Output('pie', 'figure'),
	[dash.dependencies.Input('game-list', 'value')])
def update_pie(value):
    # Filter df according to selection
    df=data_f.copy()
    # Build graph
    fig = px.sunburst(df, path=['Residence_ISO3','Game', 'Gender'],color=px.colors.sequential.Oranges, values="Hours",color_discrete_sequence=px.colors.sequential.Oranges)
    fig.update_traces(colormode=px.colors.sequential.Oranges)
    fig = go.Sunburst(data=fig, color_discrete_sequence=px.colors.sequential.Oranges)
    return{fig}

########################################################################################################################################################################################################
@app.callback(
    dash.dependencies.Output(component_id = 'Stacked', component_property = 'figure'),
    [dash.dependencies.Input(component_id = 'my-dropdown', component_property = 'value'),
    dash.dependencies.Input(component_id = 'my_dropdown_2', component_property = 'value')]
)
def update_stacked(my_dropdown, my_dropdown_2):
    dff = data_f.copy()
    if re.search( "Narcissism", my_dropdown ):
        color_code = "Narcissism_Level"
    elif re.search("Anxiety", my_dropdown):
        color_code = "GAD_T_Level"
    elif re.search("Social", my_dropdown):
        color_code = "SPIN_T_Level"
    elif re.search("Satisfaction", my_dropdown):
        color_code = "SWL_T_Level"
    
    
    dff = dff.groupby([color_code, my_dropdown_2], as_index=False)[[my_dropdown]].count()
    dff[color_code] = dff[color_code].astype(str)

    if re.search( "Narcissism_Level", color_code):
        for index, row in dff.iterrows():
            if row[color_code] == "0":
                label = "None"
            elif row[color_code] == "1":
                label = "Mild"
            elif row[color_code] == "2":
                label = "Moderate"
            elif row[color_code] == "3":
                label = "Severe"
            elif row[color_code] == "4":
                label = "Very Severe"
            dff.at[index,color_code] = label

    elif re.search("GAD_T_Level", color_code):
        for index, row in dff.iterrows():
            if row[color_code] == "0":
                label = "Minimal"
            elif row[color_code] == "1":
                label = "Mild"
            elif row[color_code] == "2":
                label = "Moderate"
            elif row[color_code] == "3":
                label = "Severe"
            dff.at[index,color_code] = label

    elif re.search("SPIN_T_Level", color_code):
        for index, row in dff.iterrows():
            if row[color_code] == "0":
                label = "None"
            elif row[color_code] == "1":
                label = "Mild"
            elif row[color_code] == "2":
                label = "Moderate"
            elif row[color_code] == "3":
                label = "Severe"
            elif row[color_code] == "4":
                label = "Very Severe"
            dff.at[index,color_code] = label

    elif re.search("SWL_T_Level", color_code):
        for index, row in dff.iterrows():
            if row[color_code] == "0":
                label = "Extremely Dissatisfied"
            elif row[color_code] == "1":
                label = "Dissatisfied"
            elif row[color_code] == "2":
                label = "Slightly Dissatisfied"
            elif row[color_code] == "3":
                label = "Neutral"
            elif row[color_code] == "4":
                label = "Slightly Satisfied"
            elif row[color_code] == "5":
                label = "Satisfied"
            elif row[color_code] == "6":
                label = "Extremely Satisfied"
            dff.at[index,color_code] = label

    dff = dff.rename(columns={my_dropdown: "Count", color_code: "Level"})
    fig_stacked = px.bar(dff, x=my_dropdown_2, y="Count", color="Level", color_discrete_sequence=px.colors.sequential.Oranges, text_auto=True)
    fig_stacked.update_layout(yaxis={'fixedrange':True},dragmode="pan",)
    fig = go.Figure(data=fig_stacked, layout=diva_layout)
    return(fig)

########################################################################################################################################################################################################
# SCATTER PLOT
########################################################################################################################################################################################################
@app.callback(
    dash.dependencies.Output(component_id = 'scatter', component_property = 'figure'),
    [dash.dependencies.Input(component_id = 'my-dropdown', component_property = 'value'),
    dash.dependencies.Input(component_id = 'my_dropdown_2', component_property = 'value'),
    dash.dependencies.Input(component_id = 'option_2', component_property = 'value')]
)
def update_scatter(my_dropdown, option_1, option_2):
    dff=data_f
    if re.search( "Narcissism", my_dropdown ):
        color_code = "Narcissism_Level"
    elif re.search("Anxiety", my_dropdown):
        color_code = "GAD_T_Level"
    elif re.search("Social", my_dropdown):
        color_code = "SPIN_T_Level"
    elif re.search("Satisfaction", my_dropdown):
        color_code = "SWL_T_Level"
    dff = dff.groupby([color_code, option_1, option_2], as_index=False)[[my_dropdown]].count()
    dff[color_code] = dff[color_code].astype(str)
    if re.search( "Narcissism_Level", color_code):
        for index, row in dff.iterrows():
            if row[color_code] == "0":
                label = "None"
            elif row[color_code] == "1":
                label = "Mild"
            elif row[color_code] == "2":
                label = "Moderate"
            elif row[color_code] == "3":
                label = "Severe"
            elif row[color_code] == "4":
                label = "Very Severe"
            dff.at[index,color_code] = label
    elif re.search("GAD_T_Level", color_code):
        for index, row in dff.iterrows():
            if row[color_code] == "0":
                label = "Minimal"
            elif row[color_code] == "1":
                label = "Mild"
            elif row[color_code] == "2":
                label = "Moderate"
            elif row[color_code] == "3":
                label = "Severe"
            dff.at[index,color_code] = label
    elif re.search("SPIN_T_Level", color_code):
        for index, row in dff.iterrows():
            if row[color_code] == "0":
                label = "None"
            elif row[color_code] == "1":
                label = "Mild"
            elif row[color_code] == "2":
                label = "Moderate"
            elif row[color_code] == "3":
                label = "Severe"
            elif row[color_code] == "4":
                label = "Very Severe"
            dff.at[index,color_code] = label
    elif re.search("SWL_T_Level", color_code):
        for index, row in dff.iterrows():
            if row[color_code] == "0":
                label = "Extremely Dissatisfied"
            elif row[color_code] == "1":
                label = "Dissatisfied"
            elif row[color_code] == "2":
                label = "Slightly Dissatisfied"
            elif row[color_code] == "3":
                label = "Neutral"
            elif row[color_code] == "4":
                label = "Slightly Satisfied"
            elif row[color_code] == "5":
                label = "Satisfied"
            elif row[color_code] == "6":
                label = "Extremely Satisfied"
            dff.at[index,color_code] = label
            # dff.at[index,color_code] = label
    dff = dff.rename(columns={my_dropdown: "Count", color_code: "Level"})
    print(color_code)
    fig_scatter = px.scatter(dff, x=option_1, y=option_2,
              size="Count", animation_frame="Level",
           size_max=60, color_discrete_sequence=['orange'])        
    
    fig_scatter.update_layout(showlegend=False,
    yaxis={'fixedrange':False,'autorange':True,},dragmode="pan",
    yaxis_autorange=True
    )

    return(fig_scatter)
########################################################################################################################################################################################################
########################################################################################################################################################################################################

########################################################################################################################################################################################################
# SYNC OPTIONS
########################################################################################################################################################################################################
df_option = data_f[['Game', 'Platform', 'Hours', 'earnings', 'whyplay', 'streams', 'Gender', 'Age', 'Work', 'Degree', 
              'Residence', 'Playstyle']]
########################################################################################################################################################################################################
########################################################################################################################################################################################################

@app.callback(
    dash.dependencies.Output(component_id = 'option_2', component_property = 'options'),
    [dash.dependencies.Input(component_id = 'my_dropdown_2', component_property = 'value')]
)
def set_cities_options(chosen_value):
    dff= df_option
    dff = dff.drop([chosen_value], axis=1)      
    return [{'label': c, 'value': c} for c in sorted(dff.columns.values)]

@app.callback(
    dash.dependencies.Output(component_id = 'my_dropdown_2_', component_property = 'value'),
    [dash.dependencies.Input(component_id = 'my_dropdown_2', component_property = 'value')]
)
def sync(chosen_value):
        
    return chosen_value

########################################################################################################################################################################################################
########################################################################################################################################################################################################

@app.callback(
    dash.dependencies.Output(component_id = 'my_dropdown_2', component_property = 'value'),
    [dash.dependencies.Input(component_id = 'my_dropdown_2_', component_property = 'value')]
)
def sync(chosen_value):
    
    return chosen_value

########################################################################################################################################################################################################
