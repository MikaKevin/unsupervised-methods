import numpy as np
import matplotlib.pyplot as plt
import json
from pandas.io.json import json_normalize
import csv
import networkx as nx

file = "standard_weighted.csv"
friends = []
with open(file) as f:
    reader = csv.reader(f, delimiter="\t")
    for i in reader:
        friends.append(i[0].split(','))
    del friends[0]
    friends = np.array(friends)
    friends = friends.astype(np.float64)
    friends.tolist()
    friends = np.array(sorted(friends, key = lambda friends_entry : friends_entry[0]))
    
prefrences = []
with open('preferences.json') as j:
    jsonData = json.load(j)
    for i in jsonData.keys():
        prefrences.append(jsonData.get(i))

user_ids = np.array(list(jsonData.keys()))
user_ids = user_ids.astype(np.int64)

# Concerts = {
#      "Classical":24,
#      "Folk":23,
#      "Jazz Hip Hop":8,
#      "Electro Pop/Electro Rock":24,
#      "Dancefloor":36,
#      "Indie Rock/Rock pop":38,
#      "Singer & Songwriter":30,
#      "Comedy":12,
#      "Musicals":10,
#      "Chill Out/Trip-Hop/Lounge":3,
#      "Soundtracks":0,
#      "Disco":0,
#      "Old school soul":9,
#      "Rock":14,
#      "Romantic":22,
#      "Bluegrass":2,
#      "Indie Rock":25,
#      "Contemporary Soul":9,
#      "Blues":14,
#      "Old School":0,
#      "Baroque":1,
#      "Instrumental jazz":8,
#      "Urban Cowboy":0,
#      "Asian Music":4,
#      "Tropical":0,
#      "Early Music":0,
#      "Classic Blues":9,
#      "Indie Pop":30,
#      "Bolero":1,
#      "Spirituality & Religion":60,
#      "Dancehall/Ragga":10,
#      "Dance":8,
#      "R&B":24,
#      "Pop":29,
#      "Film Scores":0,
#      "Grime":0,
#      "Electro Hip Hop":13,
#      "Metal":15,
#      "West Coast":12,
#      "Acoustic Blues":2,
#      "Indie Pop/Folk":18,
#      "International Pop":40,
#      "Sports":0,
#      "Trance":11,
#      "Ska":0,
#      "Brazilian Music":0,
#      "Bollywood":0,
#      "Nursery Rhymes":0,
#      "Alternative Country":6,
#      "Indian Music":1,
#      "TV shows & movies":0,
#      "Dubstep":22,
#      "Classical Period":0,
#      "Chicago Blues":0,
#      "Vocal jazz":23,
#      "TV Soundtracks":0,
#      "Latin Music":10,
#      "Rock & Roll/Rockabilly":6,
#      "Delta Blues":0,
#      "African Music":4,
#      "Opera":4,
#      "Ranchera":0,
#      "Oldschool R&B":0,
#      "Kids & Family":30,
#      "Modern":0,
#      "Soul & Funk":8,
#      "Electro":20,
#      "Alternative":18,
#      "Dub":22,
#      "Electric Blues":1,
#      "Rap/Hip Hop":28,
#      "Techno/House":29,
#      "Country Blues":13,
#      "Traditional Country":2,
#      "Country":2,
#      "East Coast":13,
#      "Contemporary R&B":27,
#      "Jazz":12,
#      "Game Scores":0,
#      "Films/Games":0,
#      "Reggae":5,
#      "Hard Rock":18,
#      "Kids":3,
#      "Dirty South":0,
#     }

# concertsOrder = {
#         0:"Classical",
#         1:"Folk",
#         2:"Jazz Hip Hop",
#         3:"Electro Pop/Electro Rock",
#         4:"Dancefloor",
#         5:"Indie Rock/Rock pop",
#         6:"Singer & Songwriter",
#         7:"Comedy",
#         8:"Musicals",
#         9:"Chill Out/Trip-Hop/Lounge",
#         10:"Soundtracks",
#         11:"Disco",
#         12:"Old school soul",
#         13:"Rock",
#         14:"Romantic",
#         15:"Bluegrass",
#         16:"Indie Rock",
#         17:"Contemporary Soul",
#         18:"Blues",
#         19:"Old School",
#         20:"Baroque",
#         21:"Instrumental jazz",
#         22:"Urban Cowboy",
#         23:"Asian Music",
#         24:"Tropical",
#         25:"Early Music",
#         26:"Classic Blues",
#         27:"Indie Pop",
#         28:"Bolero",
#         29:"Spirituality & Religion",
#         30:"Dancehall/Ragga",
#         31:"Dance",
#         32:"R&B",
#         33:"Pop",
#         34:"Film Scores",
#         35:"Grime",
#         36:"Electro Hip Hop",
#         37:"Metal",
#         38:"West Coast",
#         39:"Acoustic Blues",
#         40:"Indie Pop/Folk",
#         41:"International Pop",
#         42:"Sports",
#         43:"Trance",
#         44:"Ska",
#         45:"Brazilian Music",
#         46:"Bollywood",
#         47:"Nursery Rhymes",
#         48:"Alternative Country",
#         49:"Indian Music",
#         50:"TV shows & movies",
#         51:"Dubstep",
#         52:"Classical Period",
#         53:"Chicago Blues",
#         54:"Vocal jazz",
#         55:"TV Soundtracks",
#         56:"Latin Music",
#         57:"Rock & Roll/Rockabilly",
#         58:"Delta Blues",
#         59:"African Music",
#         60:"Opera",
#         61:"Ranchera",
#         62:"Oldschool R&B",
#         63:"Kids & Family",
#         64:"Modern",
#         65:"Soul & Funk",
#         66:"Electro",
#         67:"Alternative",
#         68:"Dub",
#         69:"Electric Blues",
#         70:"Rap/Hip Hop",
#         71:"Techno/House",
#         72:"Country Blues",
#         73:"Traditional Country",
#         74:"Country",
#         75:"East Coast",
#         76:"Contemporary R&B",
#         77:"Jazz",
#         78:"Game Scores",
#         79:"Films/Games",
#         80:"Reggae",
#         81:"Hard Rock",
#         82:"Kids",
#         83:"Dirty South"
#     }    
  
# def prefrence_validation(u_id):
#     usr_prefrence =prefrences[u_id]
#     user_favorit = []
#     j=0
#     for i in usr_prefrence:
#         if i =="1":
#             user_favorit.append(j)
#         j = j+1
#     user_favorit = np.array(user_favorit)
#     return user_favorit   

# def creat_graph( concert_order ):
#     G = nx.Graph()
#     G.add_nodes_from(user_ids)   
#     weights = []
#     for i in user_ids:
#         j = 0
#         while (friends[j,0] == i):
#            if concert_order in prefrence_validation(i):
#                if concert_order in prefrence_validation(int(friends[j,1])):
#                    weights.append(0.393)
#                else:
#                    weights.append(0.018)
#            elif (concert_order not in prefrence_validation(i))and(concert_order not in prefrence_validation(int(friends[j,1]))):
#                weights.append(0.002) 
#            j = j+1      
#     edges = zip(friends[:,0],friends[:,1], weights)        
#     G.add_weighted_edges_from(edges)
#     adj_matrix = nx.to_numpy_matrix(G, nodelist=user_ids)
#     return G , adj_matrix

# graphs_of_concerts = []
# adj_matrix_concerts =[]
# for i in concertsOrder.keys():
#     if Concerts[concertsOrder[i]] != 0:
#         graph , adj_matrix  = creat_graph(i)
#         graphs_of_concerts.append(graph)
#         adj_matrix_concerts.append(adj_matrix)

def creat_graph():
    G = nx.Graph()
    G.add_nodes_from(user_ids)
    edges = zip(friends[:,0],friends[:,1],friends[:,2])
    G.add_weighted_edges_from(edges)
    adj_matrix = nx.to_numpy_matrix(G, nodelist=user_ids)
    return G, adj_matrix

infection_G, infection_matrix = creat_graph()  
# nx.draw_networkx(infection_G) 
# plt.show() 

def calculate_total_probability_inaday(infectious_list):
   partial = 1
   for i in infectious_list:
       partial = partial * (1-i)
   total_prob = 1 -partial
   return total_prob

probability_of_getting_infected = []
for i in range(8311):
      infectious_friends = []
      for j in range(8311):
           if infection_matrix[i,j] != 0.0:
               infectious_friends.append(infection_matrix[i,j])
      probability_of_getting_infected.append(calculate_total_probability_inaday(infectious_friends)) 
      probability_of_getting_infected_daily = np.array(probability_of_getting_infected)
      
people = dict(zip(user_ids,probability_of_getting_infected_daily))
people_ = sorted(people.items(), key=lambda x: x[1], reverse=True)     
num_candidates = int(len(people_)*0.12)
candidates = []
for i in people_:
    candidates.append(i[0])
candidates_ = np.array(candidates[:num_candidates])
with open("IDs_using_ground_truth.csv","w") as f:
    writer=csv.writer(f)
    writer.writerows(zip(candidates_))
      
               


































        
    
