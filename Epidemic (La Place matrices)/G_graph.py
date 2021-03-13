
import json
import csv
import numpy as np
from matplotlib import pyplot as plt
import networkx as nx
from bitarray import bitarray
"""
Complete jsondata
get with id as string : jsonData.get("id")
id's are in range 0...8310

get returns a binary string
"""

G = nx.Graph()

with open('preferences.json') as i:
    jsonData = json.load(i)
    
"""
the csv file has 55484 entries
"""
def loadCSV(filename):
    f = open(filename)
    data = csv.reader(f, delimiter=',')
    
    return data

data = loadCSV("standard_weighted.csv")

"""
save id on the left into list a
save id on the right into lis b
"""
Number_of_friends = np.zeros((8311))

concert_visitors_per_1000_concerts = np.zeros((84))
edge_concert_assistance = np.zeros([1,84])

skipFirst = 0

#row[0] is left id, row.pop() is right id
for row in data:
    
    if(skipFirst != 0):
        
       G.add_edge(row[0],row[1],weight=float(row[2]))
       
       #For comparison
       left   = int(row[0])
       right  = int(row[1])
       Number_of_friends[left] += 1
       Number_of_friends[right] += 1
       
    else:
        skipFirst = 1
        

# Histogram
hist, bin_edges = np.histogram(Number_of_friends)
plt.bar(bin_edges[1:], hist, width=10)
plt.xlim(min(bin_edges), max(bin_edges))
plt.yscale('log', nonposy='clip')
plt.show()   

# Boxplot
outliers = dict(markerfacecolor='r', marker='X')
plt.boxplot(Number_of_friends, flierprops=outliers)

nodes = list(G.nodes)
edges = list(G.edges)

# pos = nx.spring_layout(G)
# node_color = [20000.0 * G.degree(v) for v in G]
# plt.figure(figsize=(20,20))
# nx.draw_networkx(G, pos=pos, with_labels=False,
#                  node_color=node_color,
#                  width=0.01,
#                  node_size=0.4)
# plt.axis('off')

from pyvis.network import Network
net = Network()
for i in nodes:
    net.add_node(int(i))
friends_csv = np.genfromtxt('friends.csv',delimiter=',',comments='#')
i=0
while i< len(friends_csv):
    net.add_edge(friends_csv[i][0],friends_csv[i][1])
    i=i+1
    print(i/len(friends_csv)*100)