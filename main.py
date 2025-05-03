import csv
import matplotlib.pyplot as plt
from collections import Counter

PATH = "knn_data.csv"
NUM_FEATURES = 2
NUM_NEIGHBORS = 14
DIST_METHOD = {0:'euclidian',1:'manhattan'}[0]
COLOR_CONV = {0:'red',1:'blue',2:'green'}
Xs = []
Ys = []


with open(PATH) as file:
    reader = csv.reader(file)
    next(reader)
    for row in reader:
        features = [float(i) for i in row[:NUM_FEATURES]]
        label = int(float(row[-1]))
        Xs.append(features)
        Ys.append(label)
        plt.scatter(features[0],features[1],color=COLOR_CONV[label])
plt.show()

def distance(p1,p2):
    x1,y1 = p1
    x2,y2 = p2
    if DIST_METHOD == 'euclidian':
        return ((x1-x2)**2 + (y1-y2)**2)**.5
    elif DIST_METHOD == 'manhattan':
        return abs(x1-x2)+abs(y1-y2)
    else:
        print("INVALID DISTANCE METHOD")
        exit(-1)

def basicKN(point):
    count = [ (0,distance(Xs[0],point)) ]
    for idx,p in enumerate(Xs):
        if idx==0:
            continue
        dist = distance(point,p)
        if dist<count[-1][1]:
            id = 0
            for i in range(len(count)):
                id = i
                if count[i][1]>dist:
                    break
            count.insert(id,(idx,dist))
            if len(count) > NUM_NEIGHBORS:
                count.pop()
    ids = set(i[0] for i in count)
    for i,p in enumerate(Xs):
        if i in ids:
            plt.scatter(p[0],p[1],color=COLOR_CONV[Ys[i]])
            continue
        plt.scatter(p[0], p[1], color='grey')
    plt.scatter(point[0],point[1],color='black')
    plt.show()
    labels = [Ys[i] for i in ids]
    c = Counter(labels)
    guess = max(c, key=c.get)
    print(f"that points {COLOR_CONV[guess]}, i'm like {100*c[guess]/NUM_NEIGHBORS:.2f}% sure")



basicKN((50,50))