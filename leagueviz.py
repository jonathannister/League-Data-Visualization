# Jonathan Nister
# Advanced Programming Data Visualization
# March 8 2023
# This program explores the league of legends dataset which I am using
# for my final project, comtaining lots of information from more than
# fifty thousand ranked games from the EUW region.

# The program generates the following graphs: an uniteractive
# histogram of gamelength, countplot of game winner, categorical plot of first
# tower versus game duration, categorical first dragon kills versus winner,
# most common champion picks, bans, and highest champion winrates;
# Three interactive graphs of dragons taken for gamelengths, the average team
# to take objectives when blue side wins, and win rate per objective;
# A non-numeric graph showing past Worlds events and winners on a map of
# the Earth.
# Dataset: League.csv
# Can be found at: https://www.kaggle.com/datasets/datasnaek/league-of-legends?resource=download&select=games.csv

# Bias:
# While I don't think I am particularly biased in the representation of the
# data, places where data could slightly be misrepresented include uses of
# random sampling, and relatively arbitrary cutoffs (such as champion pick counts)
# needing to be above 7000 to be included in that graph. It is also important
# to note that none of these graphs prove causation, only correlation between
# any two (or more) variables. League of Legends is an extremely complex game
# and even a dataset of 50,000 games will not capture everything.
# I am biased, however, in what I chose to explore. I was particularly
# interested in winrates as a player myself, so I chose to focus especially
# on winrates for objectives and champions. Spells, for instance, I left out.

# import necessary libraries
import pandas as pd
import folium
from matplotlib import pyplot as plt
import plotly.express as px
import seaborn as sns
import numpy as py
import numpy as np
import json
import plotly.express as px
import plotly

# Get champ name for an ID


def getChampName(ID, dict):
    return dict['name'][ID]


# create pandas dataframe by reading the dataset csv
df = pd.read_csv("League.csv")

# print info about the dataframe and its columns
print(df.info)
print(df.columns)

# Column names in array
champCols = ['t1_champ1id', 't1_champ2id', 't1_champ3id', 't1_champ4id', 't1_champ5id',
             't2_champ1id', 't2_champ2id', 't2_champ3id', 't2_champ4id', 't2_champ5id']
banCols = ['t1_ban1', 't1_ban2', 't1_ban3', 't1_ban4', 't1_ban5',
           't2_ban1', 't2_ban2', 't2_ban3', 't2_ban4', 't2_ban5', ]
spellCols = ['t1_champ1_sum1', 't1_champ1_sum2', 't1_champ2_sum1', 't1_champ2_sum2', 't1_champ3_sum1', 't1_champ3_sum2',
             't1_champ4_sum1', 't1_champ4_sum2', 't1_champ5_sum1', 't1_champ5_sum2', 't2_champ1_sum1', 't2_champ1_sum2',
             't2_champ2_sum1', 't2_champ2_sum2', 't2_champ3_sum1', 't2_champ3_sum2', 't2_champ4_sum1', 't2_champ4_sum2',
             't2_champ5_sum1', 't2_champ5_sum2']

# Drop creationTime and the columns contained within spellCols
# Don't need creationTime for anything and I didn't get a chance to work with
# the spells at all (nor are they particularly related to the questions I
# I looked at)
df.drop(["creationTime"], axis=1, inplace=True)
df.drop(columns=spellCols, axis=1, inplace=True)

# Import json data about the champions. This is useful since
# all the champion data in the dataset has dictionary keys (IDs) rather than
# their respective names as strings
jsonDict = pd.read_json("champion_info_2.json")
champInfo = pd.read_json((jsonDict['data']).to_json(), orient='index')

# Set index to ID and replace all the champion IDs with their
# correct strings from the json dictionaries
champInfo.set_index(['id'], inplace=True)
for col in champCols:
    df[col] = df[col].apply(
        lambda x: getChampName(x, champInfo))

for col in banCols:
    df[col] = df[col].apply(
        lambda x: getChampName(x, champInfo))

# Create some easy graphs to quickly get a sense of the dataset pre processing
# Histogram of game duration
sns.histplot(df, x='gameDuration', kde=True)
plt.show()

# Countplot of game winner
sns.countplot(x="winner", data=df)
plt.show()

# Categorical plot of first tower versus game duration
sns.catplot(df, x="firstTower", y="gameDuration")
plt.show()

# Categorical plot of first dragon kills versus winner (using small sample)
sns.catplot(df.sample(frac=0.002, replace=True, random_state=1),
            x="winner", y="t1_dragonKills")
plt.show()


# INTERACTIVE GRAPH 1
# plotly scatter for game duration versus dragon kills using 1% sample of frame
fig1 = px.scatter(df.sample(frac=0.01, replace=True, random_state=1), x="gameDuration", y="t1_dragonKills", hover_name="gameId",
                  size_max=30, color="gameDuration", size="t1_dragonKills", title="Game Duration versus Dragon Kills")
fig1.show()

# INTERACTIVE GRAPH 2
# plotly bar graph for first objectives versus average winning team
firstObjNames = ['firstBlood', 'firstTower', 'firstInhibitor',
                 'firstBaron', 'firstDragon', 'firstRiftHerald']

# query for winning team 1 (we will assume team 2 is quite similar)
winningTeam = df.query("winner == 1 ")
avgs = []
# for every objective column, add its average to the list for graphing
for col in firstObjNames:
    avgs.append(pd.Series.mean(winningTeam[col]))

# make list into a dataframe
d = {'First Objectives': firstObjNames, 'Average Team': avgs}
avgdf = pd.DataFrame(data=d)

# plot with plotly of first objectives
fig = px.bar(avgdf, x='First Objectives', y='Average Team',
             title="Average Team to Take First Objectives When Team 1 Wins", color='First Objectives', range_y=(0, 1.8))
fig.show()


# INTERACTIVE GRAPH 3
# winrates based on first objective take

firstObjWinrates = []
# Go through the first objective name columns
for col in firstObjNames:
    # Set counts to zero for new col
    wonCount = 0
    totCount = 0
    # Go through the column element by element
    for idx, elem in enumerate(df[col]):
        # If not zero and the same team won, increment won count
        # Else only increment total count
        if elem != 0:
            if elem == df['winner'][idx]:
                wonCount += 1
            totCount += 1
    # Now calculate average and add to array
    firstObjWinrates.append((wonCount / totCount) * 100)

# Convert list to dataframe for plotting
d = {'First Objectives': firstObjNames, 'Winrate': firstObjWinrates}
winratedf = pd.DataFrame(data=d)
# Plotly bar graph of first objectives versus winrate
fig = px.bar(winratedf, x='First Objectives', y='Winrate',
             title="Winrate Based on First Objectives", color='First Objectives')
fig.show()


# Now looking at champion picks and bans
# get all picks and bans entries, sorted
sumPicks = pd.concat([df['t1_champ1id'], df['t1_champ2id'], df['t1_champ3id'], df['t1_champ4id'], df['t1_champ5id'],
                      df['t2_champ1id'], df['t2_champ2id'], df['t2_champ3id'], df['t2_champ4id'], df['t2_champ5id']],
                     ignore_index=True)
sortedPicks = sorted(sumPicks)
sumBans = pd.concat([df['t1_ban1'], df['t1_ban2'], df['t1_ban3'], df['t1_ban4'], df['t1_ban5'],
                     df['t2_ban1'], df['t2_ban2'], df['t2_ban3'], df['t2_ban4'], df['t2_ban5']],
                    ignore_index=True)
sortedBans = sorted(sumBans)

# Send to dataframe
sortedPicks = pd.Series(sortedPicks)
picksdf = sortedPicks.value_counts().rename_axis(
    'champions').reset_index(name='counts')

# Two plots
fig, (ax1, ax2) = plt.subplots(ncols=2, sharey=False, figsize=(15, 30))
# Rotate axis labels
ax1.tick_params(axis='x', labelrotation=45)
# First plot is most common champion picks
picksdf = picksdf[(picksdf.counts > 7000)]
# Seaborn barplot of champion versus pick count
sns.barplot(data=picksdf, x='champions', y='counts', ax=ax1)
ax1.set_title('Most Common Champion Picks')

# Make bans to dataframe
sortedBans = pd.Series(sortedBans)
bansdf = sortedBans.value_counts().rename_axis(
    'champions').reset_index(name='counts')
ax2.tick_params(axis='x', labelrotation=45)
# Filter here as well to get most common bans
bansdf = bansdf[(bansdf.counts > 7000)]
# Seaborn barplot of champion versus ban count
sns.barplot(data=bansdf, x='champions', y='counts', ax=ax2)
ax2.set_title('Most Common Champion Bans')
plt.show()

# Now looking at champion winrates to compare to their pick and banrates
# Win counts dictionary which keeps track of the champion and a list of their
# total picks and total wins
wincountsdict = {}
# Loop over champion colums
for col in champCols:
    # Look over rows in each column
    for idx, elem in enumerate(df[col]):
        # If not already in the dictionary, add new entry
        if(elem not in wincountsdict):
            wincountsdict[elem] = [0, 0]
        # If already existing, check if game was won and increment entry
        # accordingly
        else:
            # checking for team by substring
            if(df['winner'][idx] == int(col[1])):
                wincountsdict[elem][0] += 1
            wincountsdict[elem][1] += 1
# Winrate dictionary
wrdict = {}
i = 0
# Calculate champion winrates and place into new dictionary
for champ in wincountsdict:
    wrdict[i] = [champ, float(
        wincountsdict[champ][0] * 100 / wincountsdict[champ][1])]
    i += 1
# Convert to dataframe for graphing
wrdf = pd.DataFrame.from_dict(
    wrdict, orient='index', columns=["champion", "winrate"])
# Filter for highest winrate (to compare to previous graphs)
filterwrdf = wrdf[(wrdf.winrate > 52)]
# from https://stackoverflow.com/questions/43770507/seaborn-bar-plot-ordering
# Sort by winrate
result = filterwrdf.groupby(["champion"])['winrate'].aggregate(
    np.median).reset_index().sort_values('winrate')
# Seaborn barplot of champion versus winrate
sns.barplot(data=filterwrdf, x='champion',
            y='winrate', order=result['champion']).set(title="Champions with Highest Winrates")
# Rotate x axis labels
plt.xticks(rotation=45)
plt.show()


# Non-numeric graph
# Folium map showing past worlds locationa and Riot HQ
# Generates HTML file leagueviz.html
my_map = folium.Map(
    location=[34.05543939506092, -118.20714943770727], zoom_start=5)

# Tile layers to select
folium.TileLayer("stamenterrain").add_to(my_map)
folium.TileLayer("stamentoner").add_to(my_map)
folium.TileLayer("stamenwatercolor").add_to(my_map)
folium.TileLayer("cartodbpositron").add_to(my_map)

folium.LayerControl().add_to(my_map)

# Add all markers of past worlds locations and special star for Riot Games HQ
# Marker location is the tournament location, popup is the location description
# and the winner, tooltip is date
folium.Marker(
    [34.03264, -118.45747], popup="<i>Riot Headquarters, Los Angeles</i>", tooltip="HQ", icon=folium.Icon(icon="star")
).add_to(my_map)

folium.Marker(
    [57.78229, 14.16216],
    popup="<i>Jonkoping, Sweden. Winner: FNATIC</i>",
    tooltip="Worlds 2011"
).add_to(my_map)

folium.Marker(
    [34.02126, -118.27949],
    popup="<i>LA, USA. Winner: Taipei Assassins</i>",
    tooltip="Worlds 2012",
).add_to(my_map)

folium.Marker(
    [34.04317, -118.26720],
    popup="<i>LA, USA. Winner: SKT T1</i>",
    tooltip="Worlds 2013",
).add_to(my_map)

folium.Marker(
    [37.56836, 126.89734],
    popup="<i>Seoul, South Korea. Winner: Samsung White</i>",
    tooltip="Worlds 2014",
).add_to(my_map)

folium.Marker(
    [52.50462, 13.44409],
    popup="<i>Berlin, Germany. Winner: SKT T1</i>",
    tooltip="Worlds 2015",
).add_to(my_map)

folium.Marker(
    [34.04317, -118.26720],
    popup="<i>LA, USA. Winner: SKT T1</i>",
    tooltip="Worlds 2016",
).add_to(my_map)

folium.Marker(
    [39.99335, 116.39806],
    popup="<i>Beijing, China. Winner: SKT T1</i>",
    tooltip="Worlds 2017",
).add_to(my_map)

folium.Marker(
    [37.4352918966479, 126.69096796117776],
    popup="<i>Incheon, South Korea. Winner: Invictus Gaming</i>",
    tooltip="Worlds 2018",
).add_to(my_map)

folium.Marker(
    [48.83865792531836, 2.3790348094465177],
    popup="<i>Paris, France. Winner: FunPlus Phoenix</i>",
    tooltip="Worlds 2019",
).add_to(my_map)

folium.Marker(
    [31.21757427010379, 121.56308725603814],
    popup="<i>Shanghai, China. Winner: DAMWON Gaming</i>",
    tooltip="Worlds 2020",
).add_to(my_map)

folium.Marker(
    [64.14069427469879, -21.876702234375298],
    popup="<i>Reykjavik, Iceland. Winner: EDward Gaming</i>",
    tooltip="Worlds 2021",
).add_to(my_map)

folium.Marker(
    [37.76816245853461, -122.38755533695605],
    popup="<i>San Francisco, USA. Winner: DRX</i>",
    tooltip="Worlds 2022",
).add_to(my_map)

# Save map to html, which can be opened via live server
my_map.save("leaguemap.html")
