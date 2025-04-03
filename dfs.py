from collections import deque
import csv
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import matplotlib.cm as cm
from matplotlib.colors import ListedColormap

global frontiers
frontiers = deque([])


df = pd.read_csv("Map3.csv")

array = np.array(df)
colorarray = np.array(df)


colors = ["red", "yellow", "green", "cyan", "blue", "purple", "pink"]

new_cmap = ListedColormap(colors)


fig, ax = plt.subplots()

im = ax.imshow(colorarray, cmap=new_cmap)
im.set_clim(0, 6)


plt.title("Grid from CSV Data")
plt.xlabel("Column Index")
plt.ylabel("Row Index")


start = np.where(array == 2)
start = (start[0][0], start[1][0])

end = np.where(array == 3)
end = (end[0][0], end[1][0])


visited = np.zeros((100, 100))
prev = np.zeros((100, 100), dtype="i,i")
prev[:] = (-1, -1)

# print(prev)
# print(visited)

# print(start)
# print(end)
frontiers.append(start)
visited[start] = 1
global foundend
foundend = False


def BFS(frame):
    global frontiers
    global foundend

    if not foundend and len(frontiers) > 0:
        for _ in range(100):  # limit the number of nodes processed per frame
            if len(frontiers) == 0:
                break

            curr = frontiers.pop()  # pop from the front of the queue
            visited[curr[0]][curr[1]] = 1
            if curr != start or curr != end or colorarray[curr] != 4:
                colorarray[curr[0]][curr[1]] = 5
            im.set_data(colorarray)
            reduced_list = []

            for i in frontiers:
                if i != curr:
                    reduced_list.append(i)
            frontiers = reduced_list
            # print(frontiers)

            for i in range(-1, 2):
                for j in range(-1, 2):
                    if (
                        (i == 0 and j == 0)
                        or (i == -1 and j == -1)
                        or (i == 1 and j == 1)
                        or (i == -1 and j == 1)
                        or (i == 1 and j == -1)
                    ):
                        continue
                    if (
                        curr[0] + i >= 0
                        and curr[0] + i < 100
                        and curr[1] + j >= 0
                        and curr[1] + j < 100
                    ):
                        if visited[curr[0] + i][curr[1] + j] == 1:
                            continue
                        if array[curr[0] + i][curr[1] + j] == 1:
                            continue
                        elif (
                            array[curr[0] + i][curr[1] + j] == 0
                            and visited[curr[0] + i][curr[1] + j] == 0
                        ):
                            frontiers.append([curr[0] + i, curr[1] + j])

                            prev[curr[0] + i][curr[1] + j] = (curr[0], curr[1])
                        elif array[curr[0] + i][curr[1] + j] == 3:
                            frontiers.append([curr[0] + i, curr[1] + j])
                            # visited[curr[0] + i][curr[1] + j] = 1

                            prev[curr[0] + i][curr[1] + j] = (curr[0], curr[1])

                            foundend = True
                            path_generate()
                            ani.event_source.stop()
    return im


def path_generate():
    curr = end
    path = []
    while (curr[0] != start[0] or curr[1] != start[1]) and (
        curr[0] != -1 and curr[1] != -1
    ):
        # print(curr)
        path.append(curr)
        if curr != end:
            colorarray[curr[0]][curr[1]] = 4
        # print(prev[curr[0]][curr[1]])
        curr = prev[curr[0]][curr[1]]
        # print(curr)

    return path


foundend = False

ani = animation.FuncAnimation(fig, BFS, interval=1, blit=False)

# print(path_generate())
im.set_data(colorarray)


writervideo = animation.PillowWriter(fps=60)
ani.save("DFSMAP3.gif", writer=writervideo)

plt.colorbar(im, ax=ax)
plt.show()
