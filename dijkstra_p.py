from collections import deque
import csv
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import matplotlib.cm as cm
from matplotlib.colors import ListedColormap


global frontiers 
frontiers = []


df = pd.read_csv('Map3.csv')

array = np.array(df)
colorarray = np.array(df)


colors = ['red', 'yellow', 'green', 'cyan', 'blue','purple','pink']

new_cmap = ListedColormap(colors)



fig, ax = plt.subplots()

im = ax.imshow(colorarray,cmap=new_cmap)
im.set_clim(0,6)



plt.title('Grid from CSV Data')
plt.xlabel('Column Index')
plt.ylabel('Row Index')


start = np.where(array == 2)
start = (start[0][0],start[1][0])

end = np.where(array == 3)
end = (end[0][0],end[1][0])


visited = np.zeros((100,100))
prev = np.zeros((100,100),dtype='i,i')
prev[:] = (-1,-1)
weight = np.zeros((100,100))
weight[:] = np.inf
weight[start] = 0

#print(prev)
#print(visited)

# print(start)
# print(end)
frontiers.append((start,weight[start]))
#print("HERE")
#print(frontiers)
visited[start] = 1
global foundend
foundend = False
global counter
counter = 0
def BFS(frame):
    global foundend
    global frontiers
    global counter
    
    if not foundend and len(frontiers) > 0:
        for _ in range(500):  # limit the number of nodes processed per frame
            if len(frontiers) == 0:
                #print("here")
                break
            curr = frontiers.pop(0)  # pop from the front of the queue
            reduced_list = []
            for i in frontiers:
                if i[0] != curr[0]:
                    reduced_list.append(i)
            frontiers = reduced_list
            # if counter % 2 == 0:
                    
            #print(curr)
            # counter += 1
            curr_w = curr[1]
            #print(curr_w)
            curr = curr[0]
            visited[curr[0]][curr[1]] = 1
            if curr != end and curr != start and colorarray[curr[0]][curr[1]] != 4:
                colorarray[curr[0]][curr[1]] = 5
            elif curr == end:
                colorarray[curr[0]][curr[1]] = 3
            
            colorarray[end] = 3
            im.set_data(colorarray)
                                
            for i in range(-1, 2):
                for j in range(-1, 2):
                    # print((curr[0] + i,curr[1] + j))
                    # print(weight[curr[0] + i][curr[1] + j])
                    # print(visited[curr[0] + i][curr[1]+j])
                    if (i == 0 and j == 0): #or (i == -1 and j == -1) or (i == 1 and j == 1) or (i == -1 and j == 1)or (i == 1 and j == -1):
                        continue
                    if curr[0] + i >= 0 and curr[0] + i < 100 and curr[1] + j >= 0 and curr[1] + j < 100:
                        if visited[curr[0] + i][curr[1] + j] == 1:
                            continue
                        if array[curr[0] + i][curr[1] + j] == 1:
                            continue
                        elif array[curr[0] + i][curr[1] + j] == 0 and visited[curr[0] + i][curr[1] + j] == 0:
                            if (i == -1 and j == -1) or (i == 1 and j == 1) or (i == -1 and j == 1)or (i == 1 and j == -1):
                                # print((curr[0] + i, curr[1] + j))
                                # print(curr_w + np.sqrt(2))
                                # print(((curr[0] + i, curr[1] + j),curr_w + np.sqrt(2)))
                                if visited[curr[0] + i][curr[1] + j] == 0:
                                    frontiers.append(((curr[0] + i, curr[1] + j),curr_w + np.sqrt(2)))
                                
                                if(curr_w+np.sqrt(2) < weight[curr[0] + i][curr[1] + j]):
                                    prev[curr[0] + i][curr[1] + j] = (curr[0], curr[1])
                                    weight[curr[0] + i][curr[1] + j] = curr_w+np.sqrt(2)
                                else:
                                    continue
                                
                            else:
                                if visited[curr[0] + i][curr[1] + j] == 0:
                                    frontiers.append(([curr[0] + i, curr[1] + j],curr_w + 1))
                                
                                if(curr_w + 1 < weight[curr[0] + i][curr[1] + j]):
                                    prev[curr[0] + i][curr[1] + j] = (curr[0], curr[1])
                                    weight[curr[0] + i][curr[1] + j] = curr_w+1
                            #print("Frontiers:",frontiers)
                            #print("Sorted:",sorted(frontiers, key=lambda x:x[1]))
                            frontiers = sorted(frontiers, key=lambda x:x[1])

                        elif array[curr[0] + i][curr[1] + j] == 3:
                            frontiers.append(([curr[0] + i, curr[1] + j],curr_w+1))
                            visited[curr[0] + i][curr[1] + j] = 1
                            
                            prev[curr[0] + i][curr[1] + j] = (curr[0], curr[1])
                            
                            foundend = True
                            path_generate()
                            colorarray[end] = 3
                            im.set_data(colorarray)
                            #print(colorarray[end])
                            
                            ani.event_source.stop()
                            exit

            
            #if curr == (5,16):
                #print(frontiers)
            
    return im

def path_generate():
    current = end
    path = []
    while (current[0] != start[0] or current[1] != start[1]) and (current[0] != -1 and current[1] != -1):
        #print(curr)
        path.append(current)
        #print(current)
        if(current[0] != end[0] or current[1] != end[1]):
            #print(current)
            colorarray[current[0]][current[1]] = 4
        #print(prev[curr[0]][curr[1]])
        colorarray[end] = 3
        current = prev[current[0]][current[1]]
        #print(curr)
    #print(colorarray[end])
    #im.set_data(colorarray)
    return path
foundend = False

ani = animation.FuncAnimation(fig, BFS,interval=1, blit=False)

#print(colorarray[end])
colorarray[end] = 3
im.set_data(colorarray)
#print(colorarray[end])
#print(colorarray[end])

writervideo = animation.PillowWriter(fps=60)
ani.save('DijkstraMAP3.gif',writer=writervideo)

plt.colorbar(im,ax=ax)
plt.show()
plt.show()
