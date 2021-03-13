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

friends = np.genfromtxt('friends.csv',delimiter=',',comments='#')

n_concerts = np.genfromtxt('n_concerts.txt',delimiter=':',comments='#',usecols=1)
def init_calendar(list_genre):
    days_per_conc = []   
    for q in list_genre:
        if n_concerts[q] != 0:
           days_per_conc.append(round(365/n_concerts[q]))
        elif n_concerts[q] == 0:
            days_per_conc.append(0)
    return days_per_conc

def simulate_one_day(df,days_per_conc,day,csv):
    ind_temp = 0
    for i in days_per_conc:
        if i != 0:
            if   day  % i == 0:
                friend_infec(csv,df,ind_temp)
        ind_temp = ind_temp+1
    return df
def vacc(df,arr):
    for i in arr:
        df.iloc[int(i),2] = 2
    return df
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


def friend_infec(csv,df,genre):
    for k in csv:
        if df[2][k[0]] == 1 or df[2][k[1]] == 1 and df[2][k[0]] != 2 and df[2][k[1]] != 2 and df[2][k[0]] != 3 and df[2][k[1]] != 3:
            if df[1][k[0]][genre] == df[1][k[1]][genre] and df[1][k[0]][genre] == 1:
                temp_ch = np.random.randint(0,1000)
                if temp_ch <393:
                    df.loc[k[0],2] = 1
                    df.loc[k[1],2] = 1
            elif df[1][k[0]][genre] != df[1][k[1]][genre]:
                temp_ch=np.random.randint(0,1000)
                if temp_ch < 18:
                    df.loc[k[0],2] = 1
                    df.loc[k[1],2] = 1
            elif df[1][k[0]][genre] == df[1][k[1]][genre] and df[1][k[0]][genre] == 0:
                temp_ch =np.random.randint(0,1000)
                if temp_ch < 2:
                    df.loc[k[0],2] = 1
                    df.loc[k[1],2] = 1
    return df
def infect_ends(df):
    for u in df[0]:
        if df[2][int(u)] == 1:
            if df.loc[int(u),3] == 14:
                temp_rand = np.random.randint(0,100)
                if temp_rand < 8:
                    df.loc[int(u),2] = 3 
                elif temp_rand > 8:
                    df.loc[int(u),2] = 2
            else:
                df.loc[int(u),3] = df.loc[int(u),3] + 1
    return df

def simulate_a_year(jsondata):
    df = init_infect_status(jsondata)
    arr = np.genfromtxt("IDs_using_ground_truth.csv",delimiter=',')
    df = vacc(df,arr)
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
        df = simulate_one_day(df,init_calendar(GenreDictionary),w,friends)
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
        print("infiziert:",max(infected))
        print("immun:",max(immune))
        print("death:",max(rip))
        w = w+1
    return df, day, healty, infected, immune, rip
import matplotlib.pyplot as plt 
#plt.title("Vaccinated IDs using eigencentrality")
plt.plot(day,healty,label="Non Infected")
plt.plot(day,infected,label="Infected")
plt.plot(day,immune,label="Immune")
plt.plot(day,rip,label="Death")
plt.xlabel("Days since first Infection")
plt.ylabel("Amount of People")
plt.grid()
plt.legend()
plt.show()
with open("IDs_using_alis_method2.csv","w") as f:
    writer=csv.writer(f)
    writer.writerows(zip(healty,infected,immune,rip))
#import os
#os.chdir('figures')
#import glob
#all_healty = pd.DataFrame()
#all_infected = pd.DataFrame()
#all_immune = pd.DataFrame()
#all_rip = pd.DataFrame()
#for i in glob.glob("*.csv"):
#    healty = pd.DataFrame(np.genfromtxt(i,delimiter=',',usecols=0))
#    infected = pd.DataFrame(np.genfromtxt(i,delimiter=',',usecols=1))
#    immune = pd.DataFrame(np.genfromtxt(i,delimiter=',',usecols=2))
#    rip = pd.DataFrame(np.genfromtxt(i,delimiter=',',usecols=3))
#    all_healty = all_healty.append(np.transpose(healty))
#    all_infected = all_infected.append(np.transpose(infected))
#    all_immune = all_immune.append(np.transpose(immune))
#    all_rip = all_rip.append(np.transpose(rip))
#    
#mean_healty = all_healty.mean(axis=0)
#std_healty = all_healty.std(axis=0)
#
#mean_infected = all_infected.mean(axis=0)
#std_infected = all_infected.std(axis=0)
#
#mean_immune = all_immune.mean(axis=0)
#std_immune = all_immune.std(axis=0)
#
#mean_rip = all_rip.mean(axis=0)
#std_rip = all_rip.std(axis=0)
#plt.title("Mean Infection over time without vaccines")
#plt.plot(day,mean_healty,'b',label='mean Not Infected')
#plt.plot(day,mean_healty+std_healty,'b--')
#plt.plot(day,mean_healty-std_healty,'b--')
#plt.plot(day,mean_infected,'y',label='mean Infected')
#plt.plot(day,mean_infected+std_infected,'y--')
#plt.plot(day,mean_infected-std_infected,'y--')
#plt.plot(day,mean_immune,'g',label='mean Immune')
#plt.plot(day,mean_immune+std_immune,'g--')
#plt.plot(day,mean_immune-std_immune,'g--')
#plt.plot(day,mean_rip,'r',label = 'mean death')
#plt.plot(day,mean_rip+std_rip,'r--')
#plt.plot(day,mean_rip-std_rip,'r--')
#plt.legend()
#plt.xlabel("Days since first Infection")
#plt.ylabel("Amount of People")
#plt.grid()
#plt.show()

