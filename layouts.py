import plotly.express as px
import dash_core_components as dcc
import dash_html_components as html
import dash_dangerously_set_inner_html
import plotly.graph_objs as go
import pandas as pd
import numpy as np
import dash
import dash_table
from dash_table.Format import Format, Group
import dash_table.FormatTemplate as FormatTemplate
from datetime import datetime as dt
from app import app

####################################################################################################
# 000 - FORMATTING INFO
####################################################################################################

####################### diva css formatting
diva_colors = {
    'dark-blue-grey' : 'rgb(62, 64, 76)',
    'medium-blue-grey' : '',
    'superdark-green' : 'rgb(41, 56, 55)',
    'dark-green' : 'rgb(57, 81, 85)',
    'medium-green' : 'rgb(93, 113, 120)',
    'light-green' : 'rgb(186, 218, 212)',
    'pink-red' : 'rgb(255, 101, 131)',
    'dark-pink-red' : 'rgb(247, 80, 99)',
    'white' : 'rgb(251, 251, 252)',
    'light-grey' : 'rgb(208, 206, 206)'
}

externalgraph_rowstyling = {
    'margin-left' : '1px',
    'margin-right' : '1px',
    'box-shadow':' rgba(0, 0, 0, 0.35) 20px 15px 5px 15px;',
}

externalgraph_colstyling = {
    'width':'100%',

    'box-shadow': 'rgba(0, 0, 0, 0.25) 50px 54px 55px, rgba(0, 0, 0, 0.12) 0px -12px 30px, rgba(0, 0, 0, 0.12) 0px 4px 6px, rgba(0, 0, 0, 0.17) 0px 12px 13px, rgba(0, 0, 0, 0.09) 0px -3px 5px;',
    # 'box-shadow':' rgba(0, 0, 0, 0.35) 15px 5px 15px;',
    'border-radius' : '10px',
    'border-style' : 'solid',
    'border-width' : '1px',
    'border-color' : diva_colors['superdark-green'],
    'background-color' : diva_colors['superdark-green'],
    
    # 'box-shadow' : '0px 0px 17px 0px rgba(186, 218, 212, .5)',
    'padding-top' : '10px'
}

filterdiv_borderstyling = {
    
    'border-radius' : '0px 0px 10px 10px',
    'border-style' : 'solid',
    'border-width' : '1px',
    'border-color' : 'rgb(255, 130, 46)',
    # background for filters div
    # 'box-shadow':' rgba(0, 0, 0, 0.35) 15px 5px 15px;',
    'background-color' : 'rgb(255, 130, 46)', 
    'box-shadow' : '2px 5px 5px 1px rgb(241, 97, 0)'
    }

navbarcurrentpage = {
    'text-decoration' : '',
    'text-decoration-color' : diva_colors['pink-red'],
    'text-shadow': '0px 0px 1px rgb(251, 251, 252)'
    }

recapdiv = {
    
    'border-radius' : '10px',
    'border-style' : 'solid',
    'border-width' : '1px',
    'border-color' : 'rgb(251, 251, 252, 0.1)',
    'margin-left' : '5px',
    'margin-right' : '5px',
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


#####################################################################################
# GAME REVIEW ANALYSIS DATA MAPPING 
#####################################################################################
games_dict = {
'Metro Exodus':'data/Result_MetroExodus.csv',
'Rivals of Aether':'data/Result_RivalsofAether.csv',
'Risk of Rain 2':'data/Result_RiskofRain2.csv',
'METAL GEAR SOLID V THE PHANTOM PAIN':'data/Result_METALGEARSOLIDVTHEPHANTOMPAIN.csv',
'SCUM':'data/Result_SCUM.csv',
'No Mans Sky':'data/Result_NoMansSky.csv',
'Out of the Park Baseball 21':'data/Result_OutoftheParkBaseball21.csv',
'My Time At Portia':'data/Result_MyTimeAtPortia.csv',
'Scrap Mechanic':'data/Result_ScrapMechanic.csv',
'Sea of Thieves':'data/Result_SeaofThieves.csv',
'Raft':'data/Result_Raft.csv',
'Realm of the Mad God Exalt':'data/Result_RealmoftheMadGodExalt.csv',
'Medieval Dynasty':'data/Result_MedievalDynasty.csv',
'Oxygen Not Included':'data/Result_OxygenNotIncluded.csv',
'Sid Meiers Civilization IV Beyond the Sword':'data/Result_SidMeiersCivilizationIVBeyondtheSword.csv',
'Mount Blade II Banner lord':'data/Result_MountBladeIIBannerlord.csv',
'Portal 2':'data/Result_Portal2.csv',
'Mount Blade War band':'data/Result_MountBladeWarband.csv',
'Rim World':'data/Result_RimWorld.csv',
'MORDHAU':'data/Result_MORDHAU.csv',
'Monster Train':'data/Result_MonsterTrain.csv',
'Monster Hunter World':'data/Result_MonsterHunterWorld.csv',
'Planet Zoo':'data/Result_PlanetZoo.csv',
'Persona 4 Golden':'data/Result_Persona4Golden.csv',
'Nie R Automata':'data/Result_NieRAutomata.csv',
'Serious Sam 4':'data/Result_SeriousSam4.csv',
'People Playground':'data/Result_PeoplePlayground.csv',
'Rust':'data/Result_Rust.csv',
'PAYDAY 2':'data/Result_PAYDAY2.csv',
'Sid Meiers Civilization III Complete':'data/Result_SidMeiersCivilizationIIIComplete.csv',
'Microsoft Flight Simulator X Steam Edition':'data/Result_MicrosoftFlightSimulatorXSteamEdition.csv',
'Satisfactory':'data/Result_Satisfactory.csv',
'Men of War Assault Squad 2':'data/Result_MenofWarAssaultSquad2.csv',
'Mortal Kombat 11':'data/Result_MortalKombat11.csv',
'Rome Total War':'data/Result_RomeTotalWar.csv',
'Northgard':'data/Result_Northgard.csv',
'Microsoft Flight Simulator':'data/Result_MicrosoftFlightSimulator.csv',
'Planet Coaster':'data/Result_PlanetCoaster.csv',
'NBA 2K20':'data/Result_NBA2K20.csv',
'Rising Storm 2 Vietnam':'data/Result_RisingStorm2Vietnam.csv',
'NBA 2K21':'data/Result_NBA2K21.csv',
'Red Dead Redemption2':'data/Result_RedDeadRedemption2.csv',
'PLAYER UNKNOWNS BATTLE GROUNDS':'data/Result_PLAYERUNKNOWNSBATTLEGROUNDS.csv',
'Sekiro Shadows Die Twice':'data/Result_SekiroShadowsDieTwice.csv',
'Path finder King maker':'data/Result_PathfinderKingmaker.csv',
'Sands of Salzaar':'data/Result_SandsofSalzaar.csv',
'Middle earth Shadow of War':'data/Result_MiddleearthShadowofWar.csv',
'Phasmophobia':'data/Result_Phasmophobia.csv',
'Enter the Gungeon':'data/Result_EntertheGungeon.csv',
'Hollow Knight':'data/Result_HollowKnight.csv',
'Far Cry 5':'data/Result_FarCry5.csv',
'Face Rig':'data/Result_FaceRig.csv',
'Grounded':'data/Result_Grounded.csv',
'Factorio':'data/Result_Factorio.csv',
'Human Fall Flat':'data/Result_HumanFallFlat.csv',
'Empyrion Galactic Survival':'data/Result_EmpyrionGalacticSurvival.csv',
'Grim Dawn ':'data/Result_GrimDawn.csv',
'Geometry Dash':'data/Result_GeometryDash.csv',
'Kingdom Come Deliverance':'data/Result_KingdomComeDeliverance.csv',
'Kovaa K20':'data/Result_KovaaK20.csv',
'Fallout 4.':'data/Result_Fallout4.csv',
'Europa Universal is IV':'data/Result_EuropaUniversalisIV.csv',
'Football Manager 2020 Touch':'data/Result_FootballManager2020Touch.csv',
'F1 2020':'data/Result_F12020.csv',
'Halo The Master Chief Collection':'data/Result_HaloTheMasterChiefCollection.csv',
'Football Manager 2020':'data/Result_FootballManager2020.csv',
'Left 4 Dead 2':'data/Result_Left4Dead2.csv',
'e Football PES 2021 SEASON UPDATE':'data/Result_eFootballPES2021SEASONUPDATE.csv',
'Fall Guys':'data/Result_FallGuys.csv',
'e Football PES 2020':'data/Result_eFootballPES2020.csv',
'FTL Faster Than Light':'data/Result_FTLFasterThanLight.csv',
'FINAL FANTASY XIV Online':'data/Result_FINALFANTASYXIVOnline.csv',
'Mafia Definitive Edition':'data/Result_MafiaDefinitiveEdition.csv',
'Hunt Showdown':'data/Result_HuntShowdown.csv',
'Kenshi':'data/Result_Kenshi.csv',
'Football Manager 2019':'data/Result_FootballManager2019.csv',
'Horizon Zero Dawn':'data/Result_HorizonZeroDawn.csv',
'Insurgency Sandstorm':'data/Result_InsurgencySandstorm.csv',
'House Flipper':'data/Result_HouseFlipper.csv',
'Mad Max':'data/Result_MadMax.csv',
'Elite Dangerous':'data/Result_EliteDangerous.csv',
'Garrys Mod':'data/Result_GarrysMod.csv',
'Marvels Avengers':'data/Result_MarvelsAvengers.csv',
'Gun fire Reborn':'data/Result_GunfireReborn.csv',
'Kerbal Space Program':'data/Result_KerbalSpaceProgram.csv',
'Mafia II Definitive Edition':'data/Result_MafiaIIDefinitiveEdition.csv',
'Fishing Planet':'data/Result_FishingPlanet.csv',
'Farming Simulator 19':'data/Result_FarmingSimulator19.csv',
'Green Hell':'data/Result_GreenHell.csv',
'Killing Floor 2':'data/Result_KillingFloor2.csv',
'Euro Truck Simulator 2 ':'data/Result_EuroTruckSimulator2.csv',
'HITMAN 2':'data/Result_HITMAN2.csv',
'Hell Let Loose':'data/Result_HellLetLoose.csv',
'Hearts of Iron IV':'data/Result_HeartsofIronIV.csv',
'For Honor':'data/Result_ForHonor.csv',
'Fallout 76.':'data/Result_Fallout76.csv',
'Battle field 4':'data/Result_Battlefield4.csv',
'Dead by Day light':'data/Result_DeadbyDaylight.csv',
'DEATH STRANDING':'data/Result_DEATHSTRANDING.csv',
'Beam NG drive':'data/Result_BeamNGdrive.csv',
'7 Days to Die':'data/Result_7DaystoDie.csv',
'Black Desert Online':'data/Result_BlackDesertOnline.csv',
'Call of Duty Black Ops III':'data/Result_CallofDutyBlackOpsIII.csv',
'DARK SOULS REMASTERED':'data/Result_DARKSOULSREMASTERED.csv',
'Assassins Creed Odyssey':'data/Result_AssassinsCreedOdyssey.csv',
'Darkest Dungeon':'data/Result_DarkestDungeon.csv',
'Dying Light':'data/Result_DyingLight.csv',
'Aseprite':'data/Result_Aseprite.csv',
'Battle Brothers':'data/Result_BattleBrothers.csv',
'Beasts of Bermuda':'data/Result_BeastsofBermuda.csv',
'Batman Arkham Knight':'data/Result_BatmanArkhamKnight.csv',
'Dont Starve Together':'data/Result_DontStarveTogether.csv',
'Eco':'data/Result_Eco.csv',
'Age of Mythology Extended Edition':'data/Result_AgeofMythologyExtendedEdition.csv',
'Conan Exiles':'data/Result_ConanExiles.csv',
'DARK SOULS II Scholar of the First Sin':'data/Result_DARKSOULSIIScholaroftheFirstSin.csv',
'DOOM Eternal':'data/Result_DOOMEternal.csv',
'Among Us':'data/Result_AmongUs.csv',
'DRAGONBALL Fighter Z':'data/Result_DRAGONBALLFighterZ.csv',
'Arma 3':'data/Result_Arma3.csv',
'ATLAS':'data/Result_ATLAS.csv',
'Destiny 2':'data/Result_Destiny2.csv',
'ASTRONEER':'data/Result_ASTRONEER.csv',
'Dragons Dogma Dark Arisen':'data/Result_DragonsDogmaDarkArisen.csv',
'Beat Saber':'data/Result_BeatSaber.csv',
'Day Z':'data/Result_DayZ.csv',
'Age of Empires II Definitive Edition':'data/Result_AgeofEmpiresIIDefinitiveEdition.csv',
'Assetto Corsa Competizione':'data/Result_AssettoCorsaCompetizione.csv',
'Counter Strike':'data/Result_CounterStrike.csv',
'Counter Strike Source':'data/Result_CounterStrikeSource.csv',
'Age of Empires II 2013':'data/Result_AgeofEmpiresII2013.csv',
'Age of Empires III 2007':'data/Result_AgeofEmpiresIII2007.csv',
'DARK SOULS III':'data/Result_DARKSOULSIII.csv',
'Deep Rock Galactic':'data/Result_DeepRockGalactic.csv',
'Command Conquer Remastered Collection':'data/Result_CommandConquerRemasteredCollection.csv',
'Borderlands 3':'data/Result_Borderlands3.csv',
'Borderlands 2':'data/Result_Borderlands2.csv',
'Car Mechanic Simulator 2018':'data/Result_CarMechanicSimulator2018.csv',
'BATTLE TECH':'data/Result_BATTLETECH.csv',
'ARK Survival Evolved':'data/Result_ARKSurvivalEvolved.csv',
'Craftopia':'data/Result_Craftopia.csv',
'Bloons TD6':'data/Result_BloonsTD6.csv',
'American Truck Simulator':'data/Result_AmericanTruckSimulator.csv',
'Assetto Corsa':'data/Result_AssettoCorsa.csv',
'TEKKEN 7':'data/Result_TEKKEN7.csv',
'The Forest':'data/Result_TheForest.csv',
'Squad':'data/Result_Squad.csv',
'Stellaris':'data/Result_Stellaris.csv',
'Star bound':'data/Result_Starbound.csv',
'Total War MEDIEVAL II Definitive Edition':'data/Result_TotalWarMEDIEVALIIDefinitiveEdition.csv',
'STARWARS Empireat War Gold Pack':'data/Result_STARWARSEmpireatWarGoldPack.csv',
'Total War THREE KINGDOMS':'data/Result_TotalWarTHREEKINGDOMS.csv',
'Totally Accurate Battle Simulator':'data/Result_TotallyAccurateBattleSimulator.csv',
'War hammer Vermintide 2':'data/Result_WarhammerVermintide2.csv',
'Total War ROME II Emperor Edition':'data/Result_TotalWarROMEIIEmperorEdition.csv',
'The Elder Scrolls V Skyrim':'data/Result_TheElderScrollsVSkyrim.csv',
'The Elder Scrolls V Skyrim Special Edition':'data/Result_TheElderScrollsVSkyrimSpecialEdition.csv',
'Spelunky 2':'data/Result_Spelunky2.csv',
'Table top Simulator':'data/Result_TabletopSimulator.csv',
'XCOM 2':'data/Result_XCOM2.csv',
'Street Fighter V':'data/Result_StreetFighterV.csv',
'Wallpaper Engine':'data/Result_WallpaperEngine.csv',
'Sid Meiers Civilization VI':'data/Result_SidMeiersCivilizationVI.csv',
'The Sims 3':'data/Result_TheSims3.csv',
'Space Engineers':'data/Result_SpaceEngineers.csv',
'Total War WARHAMMER II':'data/Result_TotalWarWARHAMMERII.csv',
'The Isle':'data/Result_TheIsle.csv',
'Total War ATTILA':'data/Result_TotalWarATTILA.csv',
'The Elder Scrolls Online':'data/Result_TheElderScrollsOnline.csv',
'The Sims 4':'data/Result_TheSims4.csv',
'They Are Billions':'data/Result_TheyAreBillions.csv',
'Subnautica':'data/Result_Subnautica.csv',
'Town of Salem':'data/Result_TownofSalem.csv',
'Terraria':'data/Result_Terraria.csv',
'The Binding of Isaac Rebirth':'data/Result_TheBindingofIsaacRebirth.csv',
'Stormworks Build and Rescue':'data/Result_StormworksBuildandRescue.csv',
'Slay the Spire':'data/Result_SlaytheSpire.csv',
'Stardew Valley':'data/Result_StardewValley.csv',
'Total War NAPOLEON Definitive Edition':'data/Result_TotalWarNAPOLEONDefinitiveEdition.csv',
'Total War EMPIRE Definitive Edition':'data/Result_TotalWarEMPIREDefinitiveEdition.csv',
'Total War SHOGUN 2':'data/Result_TotalWarSHOGUN2.csv',
'Tom Clancys Rainbow Six Siege':'data/Result_TomClancysRainbowSixSiege.csv',
'The Elder Scrolls IV Oblivion':'data/Result_TheElderScrollsIVOblivion.csv',
'Yolo Mouse':'data/Result_YoloMouse.csv',
'Transport Fever 2':'data/Result_TransportFever2.csv',
'Wasteland 3':'data/Result_Wasteland3.csv'
    }
ops = [
    {'label' : k, 'value' : games_dict[k]} for k in (games_dict)
    ]

####################################################################################################
# 000 - IMPORT DATA for Psych
####################################################################################################
data_f = pd.read_csv('data/FinalGaming_data_cleaned.csv')


game_n=set([i for i in data_f["Game"]])
option_game = [
    {'label' : k, 'value' : k} for k in (game_n)
    ]


df_option = data_f[['Game', 'Platform', 'Hours', 'earnings', 'whyplay', 'streams', 'Gender', 'Age', 'Work', 'Degree', 
              'Residence', 'Playstyle']]
####################################################################################################

####################################################################################################
# 000 - DEFINE REUSABLE COMPONENTS AS FUNCTIONS
####################################################################################################

#####################
# Header with logo
def get_header():

    header = html.Div([

        html.Div([], className = 'col-2'), #Same as img width, allowing to have the title centrally aligned

        html.Div([
            dcc.Link(
            html.H2(children='Game Review Analysis and Psychological Impact of Gaming',
                    style = {'textAlign' : 'center'}
            ),href='/apps/home')],
            
            className='col-8',
            style = {'padding-top' : '1%'}
        ),
        
        html.Br()

        ],
        
        className = 'row',
        style = {'height' : '4%',
                'background-color' : diva_colors['superdark-green']}
        )

    return header

#####################
# Nav bar
def get_navbar(p = 'home'):

    navbar_diva = html.Div([

        html.Div([], className = 'col-3'),

     

        html.Div([
            dcc.Link(
                html.H5(children = 'Psychological Impact Analysis',
                        style = {'font-weight':'bold'}),
                href='/apps/psych-overview'
                )
        ],
        className='col-2'),

        html.Div([
            dcc.Link(
                html.H5(children = 'Game Review Analysis'),
                href='/apps/page2'
                )
        ],
        className='col-2'),

        html.Div([
            dcc.Link(
                html.H5(children = 'Interesting Findings'),
                href='/apps/page3'
                )
        ],
        className='col-2'),

        html.Div([], className = 'col-3')

    ],
    className = 'row',
    style = {'background-color' : diva_colors['dark-green'],
            'box-shadow': '2px 5px 5px 1px rgba(255, 101, 131, .5)'}
    )

    navbar_page2 = html.Div([

        html.Div([], className = 'col-3'),

        html.Div([
            dcc.Link(
                html.H5(children = 'Psychological Impact Analysis'),
                href='/apps/psych-overview'
                )
        ],
        className='col-2'),

        html.Div([
            dcc.Link(
                html.H5(children = 'Game Review Analysis',
                        style = {'font-weight':'bold'}),
                href='/apps/page2'
                )
        ],
        className='col-2'),

        html.Div([
            dcc.Link(
                html.H5(children = 'Interesting Findings'),
                
                href='/apps/page3'
                )
        ],
        className='col-2'),

        html.Div([], className = 'col-3')

    ],
    className = 'row',
    style = {'background-color' : diva_colors['dark-green'],
            'box-shadow': '2px 5px 5px 1px rgb(241, 97, 0)'}
    )

    navbar_page3 = html.Div([

        html.Div([], className = 'col-3'),

        html.Div([
            dcc.Link(
                html.H5(children = 'Psychological Impact Analysis'),
                href='/apps/psych-overview'
                )
        ],
        className='col-2'),

        html.Div([
            dcc.Link(
                html.H5(children = 'Game Review Analysis'),
                href='/apps/page2'
                )
        ],
        className='col-2'),

        html.Div([
            dcc.Link(
                html.H5(children = 'Interesting Findings',
                       style = {'font-weight':'bold'}), 
                href='/apps/page3'
                )
        ],
        className='col-2'),

        html.Div([], className = 'col-3')

    ],
    className = 'row',
    style = {'background-color' : diva_colors['dark-green'],
            'box-shadow': '2px 5px 5px 1px rgb(241, 97, 0)'}
    )

    navbar_page4 = html.Div([

        html.Div([], className = 'col-3'),

        html.Div([
            dcc.Link(
                html.H5(children = 'Psychological Impact Analysis'),
                href='/apps/psych-overview'
                )
        ],
        className='col-2'),

        html.Div([
            dcc.Link(
                html.H5(children = 'Game Review Analysis'),
                href='/apps/page2'
                )
        ],
        className='col-2'),

        html.Div([
            dcc.Link(
                html.H5(children = 'Interesting Findings',
                       style = {}), 
                href='/apps/page3'
                )
        ],
        className='col-2'),

        html.Div([], className = 'col-3')

    ],
    className = 'row',
    style = {'background-color' : diva_colors['dark-green'],
            'box-shadow': '2px 5px 5px 1px rgb(241, 97, 0)'}
    )

    if p == 'filters_list':
        return navbar_diva
    elif p == 'page2':
        return navbar_page2
    elif p=='page3':
        return navbar_page3
    else:
        return navbar_page4

#####################
# Empty row

def get_emptyrow(h='45px'):
    """This returns an empty row of a defined height"""

    emptyrow = html.Div([
        html.Div([
            html.Br()
        ], className = 'col-12')
    ],
    className = 'row',
    style = {'height' : h})

    return emptyrow


####################################################################################################
# 001 - Psychological Impact Analysis Page
####################################################################################################

filters_list = html.Div([

    #####################
    #Row 1 : Header
    get_header(),

    #####################
    #Row 2 : Nav bar
    get_navbar('filters_list'),

    #####################
    #Row 3 : Filters
    html.Div([ # External row

        html.Div([ # External 12-column

            html.Div([ # Internal row

                #Internal columns
                html.Div([
                ],
                className = 'col-2'), # Blank 2 columns

                #Filter pt 1
                html.Div([

                    html.Div([
                        html.H5(
                            children='Filters on Stacked Bar Graph and Scatter Plot:',
                            style = {'margin':'auto','text-align' : 'center', 'color' : 'white'}
                        ),
                        #Date range picker
                        html.Div([
                            
                                # html.H5(children='Select a Filter: ',style={'color':'white','text-align':'left'}),
                                
                            
                            dcc.Dropdown(
                                id='my_dropdown_2_',
                                options=[{'label': s, 'value': s} for s in (df_option.columns.values)],
                                                optionHeight = 35,
                                                value = 'Age',
                                                disabled = False,
                                                multi = False,
                                                search_value = '',
                                                placeholder = 'Please Select an option...',

                        style =  {'margin':'auto','font-size': '13px','width':'200px', 'color' : diva_colors['medium-blue-grey'], 'white-space': 'nowrap', 'text-overflow': 'ellipsis'}),
                        

                        ], style = {'color':'black','margin-top' : '5px'}
                        )

                    ],
                    style = {'margin-top' : '10px',
                            'margin-bottom' : '5px',
                            'text-align' : 'left',
                            'paddingLeft': 5
                            
                            }),
                           

                ],
                className = 'col-4'), # Filter part 1

                
                #Filter pt 2
                html.Div([

                    html.Div([
                        html.H5(
                            children='Filter Psychological Traits:',
                            style = {'margin':'auto','text-align' : 'left','padding-left':'65px', 'color' : 'white'}
                        ),
                        # PSYCH GEO MAP
                        html.Div([
                            dcc.Dropdown(id = 'my-dropdown',
                                options=[{'label':'Narcissism', 'value':'Narcissism Level'},
                                              {'label':'Social Phobia', 'value':'Social Phobia Level'},
                                              {'label':'Anxiety', 'value':'Anxiety Level'},
                                              {'label':'Satisfaction With Life', 'value':'Satisfaction with Life Level'}],
                                value = 'Social Phobia Level',
                                multi = False,
                                placeholder = "Select a trait..",
                                style =  {'margin':'auto','font-size': '13px','width':'200px', 'color' : diva_colors['medium-blue-grey'], 'white-space': 'nowrap', 'text-overflow': 'ellipsis'}
                                )
                            ],
                            style = {'width' : '70%', 'margin-top' : '5px'}),
                        
                    ],
                    style = {
                        'margin-top' : '10px',
                            'margin-bottom' : '5px',
                            'text-align' : 'left',
                            'paddingLeft': 0
                            })

                ],
                className = 'col-4'), # Filter part 2

                html.Div([
                ],
                className = 'col-2') # Blank 2 columns


            ],
            className = 'row') # Internal row

        ],
        className = 'col-12',
        style = filterdiv_borderstyling) # External 12-column

    ],style={},
    className = 'row sticky-top'), # External row

    #####################
    #Row 4
    get_emptyrow(),

    #####################
    #Row 5 : Charts
    html.Div([ # External row
    

        

        html.Div([ # External 10-column
        

            html.H2(children = "Psychological Impact Analysis",
                    style = {'color' : diva_colors['white']}),

            html.Div([ # Internal row - RECAPS

                html.Div([],className = 'col-4'), # Empty column

                html.Div([
                    dash_table.DataTable(
                        id='recap-table',
                        style_header = {
                            'backgroundColor': 'transparent',
                            'fontFamily' : diva_font_family,
                            'font-size' : '1rem',
                            'color' : diva_colors['light-green'],
                            'border': '0px transparent',
                            'textAlign' : 'center'},
                        style_cell = {
                            'backgroundColor': 'transparent',
                            'fontFamily' : diva_font_family,
                            'font-size' : '0.85rem',
                            'color' : diva_colors['white'],
                            'border': '0px transparent',
                            'textAlign' : 'center'},
                        cell_selectable = False,
                        column_selectable = False
                    )
                ],
                className = 'col-4'),

                html.Div([],className = 'col-4') # Empty column

            ],
            className = 'row',
            style = recapdiv
            ), # Internal row - RECAPS

            html.Div([ # Internal row

                # Chart Column
                
                html.Div([
                    dcc.Loading(
                    
                    dcc.Graph(
                        id='choropleth')
                    )],style={'background-color':'transparent','width':'50%', 'border-radius':'25px','padding-left':'15px','padding-top':'5px'},
                className = 'm'),
                #Right text
                 html.Div([
                    
                    html.Div([
                            html.H4('Select a Game to analyse Gamers\' traits on Geo Map'),
                            html.Br(),
                            dcc.Dropdown(id = 'my-dropdown-game',
                                options = option_game,
                                # value = list(data_f["Game"].unique()),
                                value=[],
                                multi = True,
                                # persistence=True,
                                placeholder = "Select Game(s) or Leave Blank to Include All",
                                style = {'font-size': '13px', 'color' : diva_colors['medium-blue-grey'], 'white-space': 'nowrap', 'text-overflow': 'ellipsis'}
                                ),
                #                   dcc.Checklist(id='select-all',
                #   options=[{'label': 'Select All', 'value': 1}],style={'color':'white'})  
                            ],
                            style = {'display': 'block','vertical-align':'Middle','max-height':'15px','width' : '90%', 'margin-top' : '0px','margin':'auto'}),
                    html.Br(),
                   
                    ],style={'margin':'auto','width':'50%'},
                className = 'col-4'),


            ],
            className = 'row'), # Internal row
# get_emptyrow(),
            html.Div([ # Internal row
                html.Div([
                    dcc.Loading(
                    # html.H3("Sunburst Chart"),
                    
                    html.H5("The amount of hours spent playing games in each country.")

                    )],style={'margin':'auto'},
                className = 'col-4'),            

                # Chart Column
                html.Div([
                    dcc.Loading(
                        
                    dcc.Graph(
                             figure = px.sunburst(data_f, path=['Residence_ISO3','Game', 'Gender'],color="Gender", values="Hours"))
                            #  figure.update_layout({'paper_bgcolor':'rgba(0,0,0,0)','plot_bgcolor':'rgba(0,0,0,0)',})
                    )],style={'width':'50%','height':'70%','padding-right':'15px'},
                className = 'm'),],className = 'row'),


                #ROW START
    html.Div([
                 # Chart Column
                html.Div([
                    dcc.Loading(
                    dcc.Graph(
                        id='Stacked',
                        config = dict({'scrollZoom': True}),
                        )
                        
                    )],style={'width':'50%','height':'70%','padding-left':'15px'},
                className = 'm'),


                # Chart Column

                                 html.Div([
                    
                    html.Div([
                            html.H4("Select a factor and check its influence on Gamers' traits:"),
                            html.Br(),
                            dcc.Dropdown(id = 'my_dropdown_2',
                                options = [{'label': s, 'value': s} for s in (df_option.columns.values)],
                                value = 'Age',
                                multi = False,
                                placeholder = "Please Select an option...",
                                style = {'font-size': '13px', 'color' : diva_colors['medium-blue-grey'], 'white-space': 'nowrap', 'text-overflow': 'ellipsis'}
                                )
                            ],
                            style = {'width' : '70%', 'margin-top' : '0px','margin':'auto'}),
                    html.Br(),
                   
                    ],style={'margin':'auto'},
                className = 'col-4'),],className = 'row'),

                #ROW END




            html.Div([ # Internal row

                # Chart Column
                html.Div([
                    dcc.Loading(
                    
                    html.H5("Add another filter to check interesting patterns..")
                    ),
                    html.Br(),
                                 dcc.Dropdown(
                                                id='option_2',
                                                options=[],
                                                optionHeight = 35,
                                                value = 'Game',
                                                disabled = False,
                                                multi = False,
                                                search_value = '',
                                                placeholder = 'Please Select an option...',

                        style = {'font-size': '13px', 'color' : diva_colors['medium-blue-grey'], 'white-space': 'nowrap', 'text-overflow': 'ellipsis'})
                ],style={'margin':'auto'},
                className = 'col-4'),

                # Chart Column
                html.Div([
                    dcc.Loading(
                    dcc.Graph(
                        id='scatter',
                        config = dict({'scrollZoom': True}),
                        animate=False
                        )
                    )],style={'width':'50%','height':'70%','padding-right':'15px'},
                className = 'm'),
            
            

            ],
            className = 'row'), # Internal row
            
        ],
        className = 'col-',
        style = {
            'width':'90%',
            'margin':'auto',
    'box-shadow':' rgba(0, 0, 0, 0.35) 15px 5px 15px;',
    'border-radius' : '10px',
    'border-style' : 'solid',
    'border-width' : '1px',
    'border-color' : diva_colors['superdark-green'],
    'background-color' : diva_colors['superdark-green'],
    'padding-top' : '10px'

        }), # External 10-column
    ],
    className = 'row',
    style = externalgraph_rowstyling
    ), # External row
html.Div([],className = 'col-1'), # Empty column
])

####################################################################################################
# 002 - Game Review Analysis Page
####################################################################################################

page2 = html.Div([

    #####################
    #Row 1 : Header
    get_header(),

    #####################
    #Row 2 : Nav bar
    get_navbar('page2'),

    #####################
    #Row 3 : Filters
    html.Div([ # External row

                        #Filter pt 2
                html.Div([
                    html.Br(),

                    html.Div([
                        html.H5(
                            children='Game Selection:',
                            style = {'margin':'','text-align' : 'left', 'color' : diva_colors['medium-blue-grey'], 'font-weight':'bold','font-size':'1.6em'}
                        ),
                        #Review Analysis Chart
                        html.Div([
                            dcc.Dropdown(id = 'game-list',
                                options = ops,
                                value = 'data/Result_RiskofRain2.csv',
                                multi = False,
                                placeholder = "Select a Game",
                                style = {'font-size': '13px', 'color' : diva_colors['medium-blue-grey'], 'white-space': 'nowrap', 'text-overflow': 'ellipsis'}
                                )
                            ],
                            style = {'width' : '80%', 'margin-top' : '5px'}),
                        
                    ],
                    style = {'margin-top' : '10px',
                            'margin-left' : '100px',
                            'margin-bottom' : '5px',
                            'text-align' : 'left',
                            'paddingLeft': 50})

                ],style={'margin':''},
                className = 'col-4'), # Filter part 2



    ],
    className = 'row sticky-top'), # External row

    #####################
    #Row 4
    get_emptyrow(),

    #####################
    #Row 5 : Charts
    html.Div([ # External row

html.Div([
        ],
        className = 'col-1'), # Blank 1 column

        html.Div([ # External 10-column

            html.H2(children = "Review Analysis",
                    style = {'color' : diva_colors['white']}),

            html.Div([ # Internal row - RECAPS

                html.Div([],className = 'col-4'), # Empty column

                html.Div([
                    dash_table.DataTable(
                        id='recap-table',
                        style_header = {
                            'backgroundColor': 'transparent',
                            'fontFamily' : diva_font_family,
                            'font-size' : '1rem',
                            'color' : diva_colors['light-green'],
                            'border': '0px transparent',
                            'textAlign' : 'center'},
                        style_cell = {
                            'backgroundColor': 'transparent',
                            'fontFamily' : diva_font_family,
                            'font-size' : '0.85rem',
                            'color' : diva_colors['white'],
                            'border': '0px transparent',
                            'textAlign' : 'center'},
                        cell_selectable = False,
                        column_selectable = False
                    )
                ],
                className = 'col-4'),

                html.Div([],className = 'col-4') # Empty column

            ],
            className = 'row',
            style = recapdiv
            ), # Internal row - RECAPS

            html.Div([ # Internal row

                
                # Chart Column
                html.Div([
                    dcc.Loading(
                    dcc.Graph(
                        id='rv-analysis')
                    )],style={'margin':'auto','height':'90%','width':'80%'},
                className = 'col4'),


            ],
            className = 'row'), # Internal row




        ],
        className = 'col-10',
        style = externalgraph_colstyling), # External 10-column

        html.Div([
        ],
        className = 'col-1'), # Blank 1 column

    ],
    className = 'row',
    style = externalgraph_rowstyling
    ), # External row

])


####################################################################################################
# 003 - Interesting Findings
####################################################################################################

page3 = html.Div([

    

    #####################
    #Row 1 : Header
    get_header(),

    #####################
    #Row 2 : Nav bar
    get_navbar('page3'),

    #####################
    #Row 3 : Filters
    html.Div([ # External row
    
    dash_dangerously_set_inner_html.DangerouslySetInnerHTML('''
           <br>
        <div class="header" class="jumbotron" style="background:rgb(255, 130, 46);opacity:1;box-shadow: 0 3px 10px rgb(0 0 0 / 0.2);border-shadow:5px 8px 8px 1px rgb(241, 97, 0);width:60%;margin:auto">
        <h1>Interesting Findings</span></h1>
        
      </div>
      
      <div class="sp-slideshow">
        
        <input type="radio" id="button-1" name="radio-set" class="sp-selector-1" checked="checked" />
        <label for="button-1" class="button-label-1"></label>
        
        <input type="radio" id="button-2" name="radio-set" class="sp-selector-2" />
        <label for="button-2" class="button-label-2"></label>
        

        <input type="radio" id="button-3" name="radio-set" class="sp-selector-3" />
        <label for="button-3" class="button-label-3"></label>

        <input type="radio" id="button-4" name="radio-set" class="sp-selector-4" />
        <label for="button-4" class="button-label-4"></label>                
        
   
        
        <label for="button-1" class="sp-arrow sp-a1"></label>
        <label for="button-2" class="sp-arrow sp-a2"></label>
        <label for="button-3" class="sp-arrow sp-a3"></label>
        <label for="button-4" class="sp-arrow sp-a4"></label>
       
        
        <div class="sp-content">
          <div class="sp-parallax-bg"></div>
          <ul class="sp-slider clearfix">
            <li><img src="../assets/intro.png" alt="image01" /></li>
            <li><img src="../assets/3.png" alt="image02" /></li>
            <li><img src="../assets/4.png" alt="image02" /></li>
            <li><img src="../assets/5.png" alt="image03" /></li>
            <li><img src="https://i.imgur.com/af56d4C.png" alt="image04" /></li>
            <li><img src="https://i.imgur.com/qVefFH7.png" alt="image05" /></li>
          </ul>
        </div><!-- sp-content -->
          
        
      </div> <!-- sp-slideshow -->
        
      </div>
      <style>
          body {
  font-family: 'Open Sans Condensed', 'Arial Narrow', serif;
  font-weight: 400;
  font-size: 15px;
  color: #333;
  background: url("back-g.jpg") repeat top left;
}

.container {
  width: 100%;
  position: relative;
}

.container > .header {
  margin: 10px;
  padding: 20px 10px 10px 10px;
  position: relative;
  display: block;
  
  text-align: center;  
}

.container > .header h1 {
  font-size: 40px;
  line-height: 40px;
  margin: 0;
  position: relative;
  font-weight: 300;
  color: white;
  
}

.container > .header h1 span{
  font-weight: 700;
}

.container > .header h2 {
  font-size: 14px;
  font-weight: 300;
  margin: 0;
  padding: 15px 0 5px 0;
  color: white;
  font-family: Cambria, Georgia, serif;
  font-style: italic;
  
}



.sp-slideshow {
  position: relative;
  margin: 10px auto;
  width: 80%;
  max-width: 1000px;
  min-width: 260px;
  height: 460px;
  border: 10px solid #ddd;
  border: 10px solid rgba(255,255,255,0.9);
  box-shadow: 0 2px 6px rgba(0,0,0,0.4);
}

.sp-content {
  background: #7d7f72 url("../assets/wO5gzQp.png") repeat scroll 0 0;
  position: relative;
  width: 100%;
  height: 100%;
  overflow: hidden;
}

.sp-parallax-bg {
  background: url('../assets/bg.png') repeat-x scroll 0 0;
  background-size: cover;
  position: absolute;
  top: 0;
  left: 0;
  width: 100%;
  height: 100%;
  overflow: hidden;
}

.sp-slideshow input {
  position: absolute;
  bottom: 15px;
  left: 50%;
  width: 9px;
  height: 9px;
  z-index: 1001;
  cursor: pointer;
  opacity: 0;
}

.sp-slideshow input + label {
  position: absolute;
  bottom: 15px;
  left: 50%;
  width: 6px;
  height: 6px;
  display: block;
  z-index: 1000;
  border: 3px solid #fff;
  border: 3px solid rgba(255,255,255,0.9);
  border-radius: 50%;
  transition: background-color linear 0.1s;
}

.sp-slideshow input:checked + label {
  background-color: #fff;
  background-color: rgba(255,255,255,0.9);
}

.sp-selector-1, .button-label-1 {
  margin-left: -36px;
}

.sp-selector-2, .button-label-2 {
  margin-left: -18px;
}
.sp-selector-4, .button-label-4 {
  margin-left: 18px;
}
.sp-selector-5, .button-label-5 {
  margin-left: 36px;
}

.sp-arrow {
  position: absolute;
  top: 50%;
  width: 28px;
  height: 38px;
  margin-top: -19px;
  display: none;
  opacity: 0.8;
  cursor: pointer;
  z-index: 1000;
  background: transparent url('../assets/yeMfI0Q.png') no-repeat;
  transition: opacity linear 0.3s;
}

.sp-arrow: hover {
  opacity: 1;
}

.sp-arrow:active {
  margin-top: -18px;
}

.sp-selector-1:checked ~ .sp-arrow.sp-a2,
.sp-selector-2:checked ~ .sp-arrow.sp-a3,
.sp-selector-3:checked ~ .sp-arrow.sp-a4,
.sp-selector-4:checked ~ .sp-arrow.sp-a5 {
  right: 15px;
  display: block;
  background-position: top right;
}
.sp-selector-2:checked ~ .sp-arrow.sp-a1,
.sp-selector-3:checked ~ .sp-arrow.sp-a2,
.sp-selector-4:checked ~ .sp-arrow.sp-a3,
.sp-selector-5:checked ~ .sp-arrow.sp-a4 {
  left: 15px;
  display: block;
  background-position: top left;
}

.sp-slideshow input:checked ~ .sp-content {
  transition: background-position linear 0.6s, background-color linear 0.8s;
}

.sp-slideshow input:checked ~ .sp-content .sp-parallax-bg {
  transition: background-position linear 0.7s;
}

input.sp-selector-1:checked ~ .sp-content {
  background-position: 0 0;
  background-color: #727b7f;
}

input.sp-selector-2:checked ~ .sp-content {
  background-position: -100px 0;
  background-color: #7f7276;
}

input.sp-selector-3:checked ~ .sp-content {
  background-position: -200px 0;
  background-color: #737f72;
}

input.sp-selector-4:checked ~ .sp-content {
  background-position: -300px 0;
  background-color: #79727f;
}

input.sp-selector-5:checked ~ .sp-content {
  background-position: -400px 0;
  background-color: #7d7f72;
}

input.sp-selector-1:checked ~ .sp-content .sp-parallax-bg {
    background-position: 0 0;
}
 
input.sp-selector-2:checked ~ .sp-content .sp-parallax-bg {
    background-position: -200px 0;
}
 
input.sp-selector-3:checked ~ .sp-content .sp-parallax-bg {
    background-position: -400px 0;
}
 
input.sp-selector-4:checked ~ .sp-content .sp-parallax-bg {
    background-position: -600px 0;
}
 
input.sp-selector-5:checked ~ .sp-content .sp-parallax-bg {
    background-position: -800px 0;
}

.sp-slider {
  position: relative;
  left: 0;
  width: 500%;
  height: 100%;
  list-style: none;
  margin: 0;
  padding: 0;
  transition: left ease-in 0.8s;
}

.sp-slider > li {
  color: #fff;
  width: 20%;
  box-sizing: border-box;
  height: 100%;
  padding: 0 60px;
  float: left;
  text-align: center;
  opacity: 0.4;
  transition: opactiy ease-in 0.4s 0.8s;
}

.sp-slider > li img {
  box-sizing: border-box;
  display: block;
  margin: 0 auto;
  padding: 40px 0 50px 0;
  max-height: 100%;
  max-width: 100%;
}

input.sp-selector-1:checked ~ .sp-content .sp-slider {
    left: 0;
}
 
input.sp-selector-2:checked ~ .sp-content .sp-slider {
    left: -100%;
}
 
input.sp-selector-3:checked ~ .sp-content .sp-slider {
    left: -200%;
}
 
input.sp-selector-4:checked ~ .sp-content .sp-slider {
    left: -300%;
}
 
input.sp-selector-5:checked ~ .sp-content .sp-slider {
    left: -400%;
}

input.sp-selector-1:checked ~ .sp-content .sp-slider > li:first-child,
input.sp-selector-2:checked ~ .sp-content .sp-slider > li:nth-child(2),
input.sp-selector-3:checked ~ .sp-content .sp-slider > li:nth-child(3),
input.sp-selector-4:checked ~ .sp-content .sp-slider > li:nth-child(4),
input.sp-selector-5:checked ~ .sp-content .sp-slider > li:nth-child(5){
    opacity: 1;
}

.note {
  padding: 5px;
  text-align: center;
}

      </style>
    '''),

    ],
    className = 'container'), # External row

    #####################
    #Row 4
    get_emptyrow(),

    #####################
    #Row 5 : Charts
    html.Div([ # External row

        html.Br()

    ])

])

####################################################################################################
# 002 -  PageHOME
####################################################################################################

home = html.Div([

    #####################
    #Row 1 : Header
    get_header(),

    #####################
    #Row 2 : Nav bar
    get_navbar('home'),

    #####################
    #Row 3 : Filters
    html.Div([ # External row
     dash_dangerously_set_inner_html.DangerouslySetInnerHTML('''
     <body>
     <br>
    <style>
        body, html {
  height: 100%;
  margin: 0;
  font-size: 16px;
  font-family: "Lato", sans-serif;
  font-weight: 1.8em;
  color: grey;
}

.pimg1, .pimg2, .pimg3 {
  position: relative;
  opacity: 0.70;
  background-position: center;
  background-size: 200vw 250vh;
  background-repeat: no-repeat;
  background-attachment: fixed;
}

.pimg1 {
  background-image: url('/assets/s.png');
  min-height: 400px;
}

.pimg2 {
  background-image: url('/assets/Gm.png');
  min-height: 400px;
}
.pimg3 {
  background-image: url('/assets/123.png');
  min-height: 400px;
}

.section {
  text-align: center;
  padding: 50px 80px;
  
}

.section-light {
  background-color: #f4f4f4;
  color: #666;
}

.section-dark {
  background-color: #282e34;
  color: #ddd;
}

.ptext {
  position: absolute;
  top: 50%;
  transform: translateY(-50%);
  text-align: center;
  width: 100%;
  color: #fff;
  font-size: 27px;
  letter-spacing: 8px;
  text-transform: uppercase;
 }

.ptext .border {
  background-color: #111;
  padding: 20px;
  border-radius: 10px 20px;
  box-shadow: 0 4px 0rgba(0, 0, 0, 0.5%);
  
  
}


.ptext .trans {
  background-color: transparent;
}


@media(max-width: 568px) {
  .pimg1, .pimg2, .pimg3 {
    background-attachment: scroll;
  }
}
    </style>
    <div class="pimg1">
        <div class="ptext">
          <span class="border">Overview</span>
        </div>
      </div>
      
      
      <section class="section section-light">
        
        <p><h3 style="color:black;font-size:1.4em;text-align:justified;">
        <li>The purpose of this project is to demonstrate effective visualization techniques that will assist target users in understanding <b style="color:red">key aspects of gaming</b>.</li><br>
        <li> Understanding the behavioral/psychological components of games by looking at the gamers' <b style="color:red">socio-economic factors</b>, such as anxiety, social phobia, narcissism, and life satisfaction, is a good way to learn more about them.</li><br>
        <li> With the help of our visualizations, we present the aforementioned parts in a way that's easier to comprehend.</li> </h3></p>
       
      </section>
      
      <div class="pimg2">
        <div class="ptext">
          <span class="border">Dataset 1: Psychological Impact of Gaming </span>
        </div>
      </div>
      
      <section class="section section-dark">
        <h2>Gaming Habits and Psychological Well-being Data</h2>
        <p><h3>With over <b style="color:red">13000</b> participants, this is the biggest openly available dataset connecting gaming habits, various socio-economic factors and measures of anxiety, social phobia, life satisfaction and narcissism.<br><br>

<h2></h2>

<br>
<p style="font-size:1em;">Between 18 and 63 years (M = 20.93) completed the survey. Participants resided in 109 different countries with most of the participants coming from the USA (4569), Germany (1413), the UK (1032) and Canada (994).</p></p>
     <p>For more:<a href="https://osf.io/vnbxk/?view_only=4c54da075e164ea2a5329f5669d03c41">Source</a>
     <br>
     Scales Used: <a href="https://adaa.org/sites/default/files/GAD-7_Anxiety-updated_0.pdf">1</a>
                    <a href="https://help.greenspacehealth.com/article/95-social-anxiety-spin">2</a>
                    <a href="https://fetzer.org/sites/default/files/images/stories/pdf/selfmeasures/SATISFACTION-SatisfactionWithLife.pdf">3</a>
                    <a href="https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0103469">4</a>
     </p> </section>
      
      <div class="pimg3">
       <div class="ptext">
          <span class="border">Dataset 2: Game Review Analysis</span>
        </div>
      </div>
      
      <section class="section section-dark">
        <h2>Game Review Analysis</h2>
        <p style="font-size:1.5em">The data for most popular games have been collected from Steam Game Reviews Dataset (Kaggle) for <br>
<li style="font-size:1.5em"><b style="color:red">186 Games</b>~1.66 GB</li>
<p style="font-size:1.5em">The review texts from these games are analyzed and the most accurate aspects describing the gamers’ opinion is visualized based on the sentiment of the aspects discussed</p>
</p>
Source: <a href="https://www.kaggle.com/datasets/smeeeow/steam-game-reviews">Link</a>
      </section>
      
      
      <div class="pimg1">
        <div class="ptext">
        By
          <span class="border" >
          Neelkanth Poosa
          </span>
          <br><br>
          <span class="border" >
          Nysa P Ginu
          </span>
        </div>
      </div>
      
      

</body>
     '''),
     ],style={'background-color':''},
    className = 'container'),

    #####################
    #Row 4
    get_emptyrow(),

    #####################
    #Row 5 : Charts
    html.Div([ # External row

        html.Br()

    ])

])


