from collections import deque
import csv
import pandas as pd
import numpy as np

frontiers = deque([])


df = pd.read_csv('Map1.csv')
array = df.to_numpy()
rowcount = 0
colcount = 0
   
for i in array:
    colcount = 0
    for j in i:
        if j == 2:
            start = [rowcount,colcount]
        colcount += 1
    rowcount += 1
#print(start)
#print(array[4][15])

dis = np.ones((100,100))
dis = dis*-1
dis[start] = 0

visisted = np.zeros((100,100))

curr = start

    
    if curr[0] != 0:
        if array[curr[0]+1,curr[1]] == 1:
            dis[curr[0]+1,curr[1]] = dis[curr] + 1
        elif array[curr[0]+1,curr[1]] == 0:

        else array[curr[0]+1,curr[1]] == 2:    
    if curr[0] != 99:
    if curr[1] != 0:
    if curr[1] != 0:    


