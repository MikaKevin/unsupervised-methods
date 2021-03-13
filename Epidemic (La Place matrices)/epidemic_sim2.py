import json
import csv
import numpy as np

"""
Complete jsondata
get with id as string : jsonData.get("id")
id's are in range 0...8310

get returns a binary string
"""
with open('preferences.json') as i:
    jsonData = json.load(i)
    
"""
the csv file has 55484 entries
"""
def loadCSV(filename):
    f = open(filename)
    data = csv.reader(f, delimiter=',')
    
    return data

data = loadCSV("friends.csv")

"""
save id on the left into list a
save id on the right into lis b
"""
Number_of_friends = np.zeros((8311))

concert_visitors_per_1000_concerts = np.zeros((84))

skipFirst = 0

#row[0] is left id, row.pop() is right id
for row in data:
    if(skipFirst != 0):
        left = int(row[0])
        Number_of_friends[left] += 1
        rigth = int(row.pop())
        
        #for each pair we gather the preference binary string
        pref_left = jsonData.get(str(left))
        pref_right = jsonData.get(str(rigth))
        
        #loop over both strings and add visitation rates to concert_visitors array
        for i in range(84):
            if pref_left[i] == "1" and pref_right[i] == "1":
                concert_visitors_per_1000_concerts[i] += 393
            elif pref_left[i] == "0" and pref_right[i] == "0":
                concert_visitors_per_1000_concerts[i] += 2
            else:
                concert_visitors_per_1000_concerts[i] += 18
    else:
        skipFirst = 1

# calculate yearly visitors for one genre according to amount in n_concerts txt
visited_classical_concerts = concert_visitors_per_1000_concerts[0] / 1000 * 24

yearly_visitors_per_genre = np.zeros((84))

# Using readlines() 
file1 = open('n_concerts.txt', 'r') 
Lines = file1.readlines() 
  
#get yearly concerts and calculate yearly visitors according to amount of concerts
index = 0
skip = 0
# Strips the newline character 
for line in Lines: 
    if skip != 0:
        t = line.strip().split(":")
        n_concert = int(t.pop())
        yearly_visitors_per_genre[index] = concert_visitors_per_1000_concerts[index]/1000 * n_concert
        index += 1
    else:
        skip = 1

#Dictionary to convert an index into according genre
GenreDictionary = {
        0:"Classical",
        1:"Folk",
        2:"Jazz Hip Hop",
        3:"Electro Pop/Electro Rock",
        4:"Dancefloor",
        5:"Indie Rock/Rock pop",
        6:"Singer & Songwriter",
        7:"Comedy",
        8:"Musicals",
        9:"Chill Out/Trip-Hop/Lounge",
        10:"Soundtracks",
        11:"Disco",
        12:"Old school soul",
        13:"Rock",
        14:"Romantic",
        15:"Bluegrass",
        16:"Indie Rock",
        17:"Contemporary Soul",
        18:"Blues",
        19:"Old School",
        20:"Baroque",
        21:"Instrumental jazz",
        22:"Urban Cowboy",
        23:"Asian Music",
        24:"Tropical",
        25:"Early Music",
        26:"Classic Blues",
        27:"Indie Pop",
        28:"Bolero",
        29:"Spirituality & Religion",
        30:"Dancehall/Ragga",
        31:"Dance",
        32:"R&B",
        33:"Pop",
        34:"Film Scores",
        35:"Grime",
        36:"Electro Hip Hop",
        37:"Metal",
        38:"West Coast",
        39:"Acoustic Blues",
        40:"Indie Pop/Folk",
        41:"International Pop",
        42:"Sports",
        43:"Trance",
        44:"Ska",
        45:"Brazilian Music",
        46:"Bollywood",
        47:"Nursery Rhymes",
        48:"Alternative Country",
        49:"Indian Music",
        50:"TV shows & movies",
        51:"Dubstep",
        52:"Classical Period",
        53:"Chicago Blues",
        54:"Vocal jazz",
        55:"TV Soundtracks",
        56:"Latin Music",
        57:"Rock & Roll/Rockabilly",
        58:"Delta Blues",
        59:"African Music",
        60:"Opera",
        61:"Ranchera",
        62:"Oldschool R&B",
        63:"Kids & Family",
        64:"Modern",
        65:"Soul & Funk",
        66:"Electro",
        67:"Alternative",
        68:"Dub",
        69:"Electric Blues",
        70:"Rap/Hip Hop",
        71:"Techno/House",
        72:"Country Blues",
        73:"Traditional Country",
        74:"Country",
        75:"East Coast",
        76:"Contemporary R&B",
        77:"Jazz",
        78:"Game Scores",
        79:"Films/Games",
        80:"Reggae",
        81:"Hard Rock",
        82:"Kids",
        83:"Dirty South"
    }

import pandas as pd 
def init_infect_status(jsonData):
    preferences = pd.DataFrame(dict.items(jsonData))
    #0 = not infected, 1 = infected, 2 = immune, 3 = RiP
    start_infection = np.zeros((len(preferences),1))
    preferences[2] = start_infection
    preferences[3] = start_infection
    return preferences

def spread_infec(df):
    start_infected = np.random.randint(0,len(df),round(len(df)/100))
    indi = 0
    while indi < len(start_infected):
        if df[2][start_infected[indi]] != 2 and df[2][start_infected[indi]] != 1:
            df.iloc[start_infected[indi],2] = 1
            indi = indi+1
        else:
            start_infected[indi] = np.random.randint(0,len(df))
    return df            
friends_csv_left = np.genfromtxt('friends.csv',delimiter=',',comments='#',usecols=0)
friends_csv_right = np.genfromtxt('friends.csv',delimiter=',',comments='#',usecols=1)
def find_friends(df):
    friends = np.empty(len(df))
    df[4] = friends
    df[4] = df[4].astype(object)
    for t in df[0].astype('int'):
        ind_of_friends_left = np.where(friends_csv_left==df[0].astype('int')[t])
        ind_of_friends_right = np.where(friends_csv_right==df[0].astype('int')[t])
        friends_temp = friends_csv_right[ind_of_friends_left].tolist()
        for h in friends_csv_left[ind_of_friends_right].tolist():
            friends_temp.append(h)
        df.at[t,4] = [friends_temp]
    return df
n_concerts = np.genfromtxt('n_concerts.txt',delimiter=':',comments='#',usecols=1)

pref = init_infect_status(jsonData)
pref = find_friends(pref)
import networkx as nx

G = nx.Graph()
for i in pref[0]:
    G.add_node(int(i))
    print(int(i)/len(pref[0]))
    for lkq in pref[4]:
        for x in lkq:
            for y in x:
                G.add_edge(int(i),int(y))

import matplotlib.pyplot as plt
nx.draw(G)
plt.show()
