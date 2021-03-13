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

def friend_infect(df,genre,user):
    user = int(user)
    print(user/len(df))
    for lkq in df[4]:
        for x in lkq:
            for y in x:
                if df[2][user] == 1 or df[2][y] == 1:
                    if int(df[1][user][genre]) == int(df[1][y][genre]) & int(df[1][user][genre] == 1):    
                        temp_ch = np.random.randint(0,1000)
                        if temp_ch <393:
                            df.iloc[int(user),2] = 1
                            df.iloc[int(y),2] = 1
                    elif df[1][user][genre] != df[1][y][genre]:
                        temp_ch=np.random.randint(0,1000)
                        if temp_ch < 18:
                            df.iloc[int(user),2] = 1
                            df.iloc[int(y),2] = 1
                    elif df[1][user][genre] == df[1][y][genre] & df[1][user][genre] == 0:
                        temp_ch =np.random.randint(0,1000)
                        if temp_ch < 2:
                            df.iloc[int(user),2] = 1
                            df.iloc[int(y),2] = 1
    return df

def infect_ends(df):
    for u in df[0]:
        if df[2][int(u)] == 1:
            if df[3][int(u)] == 14:
                temp_rand = np.random.randint(0,100)
                if temp_rand < 8:
                    df[3][int(u)] = 3 
                elif temp_rand > 8:
                    df[2][int(u)] = 2
            else:
                df[3][int(u)] == df[3][int(u)] + 1
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

def init_calendar(list_genre):
    days_per_conc = []   
    for q in list_genre:
        if n_concerts[q] != 0:
           days_per_conc.append(round(365/n_concerts[q]))
        elif n_concerts[q] == 0:
            days_per_conc.append(0)
    return days_per_conc
def simulate_one_day(df,days_per_conc,day):
    ind_temp = 0
    for i in days_per_conc:
        if i != 0:
            if   day  % i == 0:
                for k in df[0]:
                    df = friend_infect(df,ind_temp,k)
        ind_temp = ind_temp+1
    
    return df
            
        
def simulate_a_year(jsondata):
    df = init_infect_status(jsondata)
    df = find_friends(df)
    df = spread_infec(df)
    day = np.array([])
    healty = np.array([])
    infected = np.array([])
    immune = np.array([])
    rip = np.array([])
    w = 1
    while w < 365:
        print("Day:",w)
        k = 0
        j = 0
        l = 0
        p = 0
        df = simulate_one_day(df,init_calendar(GenreDictionary),w)
        df = infect_ends(df)
        day = np.append(day,w)
        for g in df[2]:
            if g == 0:
                k = k+1
            elif g == 1:
                j = j+1
            elif g == 2:
                l = l+1
            elif g == 3:
                p = p+1
        healty = np.append(healty,k)
        infected = np.append(infected,j)
        immune = np.append(immune,l)
        rip = np.append(rip,p)
        w = w+1
    return df, day, healty, infected, immune, rip
