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
medians = []
tree = [[[],[]],[[],[]]]
print(tree)

with open(PATH) as file:
    reader = csv.reader(file)
    next(reader)
    for row in reader:
        features = [float(i) for i in row[:NUM_FEATURES]]
        label = int(float(row[-1]))
        Xs.append(features)
        Ys.append(label)
        plt.scatter(features[0],features[1],color=COLOR_CONV[label])
for i in range(len(Xs[0])):
    s = sorted(Xs,key=lambda x:x[i])
    medians.append(s[len(s)//2][i])
plt.axvline(x=medians[0],color='black')
plt.axhline(y=medians[1],color='black')
for idx,p in enumerate(Xs):
    indexes = [1 if p[i]>medians[i] else 0 for i in range(len(p))]
    tree[indexes[0]][indexes[1]].append((idx,p))
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

def BasicKN(point):
    count = [ (0,distance(Xs[0],point)) ]
    if len(Xs) > NUM_NEIGHBORS:
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
    else:
        count = [(idx, distance(point, p)) for idx,p in enumerate(Xs)]
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
    print(f"that points {COLOR_CONV[guess]}, i'm like {100*c[guess]/len(ids):.2f}% sure")

def KDTree(point):
    indexes = [1 if point[i] > medians[i] else 0 for i in range(len(point))]
    sector = tree[indexes[0]][indexes[1]]
    count = [(0, distance(sector[0][1], point))]
    if len(sector) > NUM_NEIGHBORS:
        for idx, p in sector:
            if idx == 0:
                continue
            dist = distance(point, p)
            if dist < count[-1][1]:
                id = 0
                for i in range(len(count)):
                    id = i
                    if count[i][1] > dist:
                        break
                count.insert(id, (idx, dist))
                if len(count) > NUM_NEIGHBORS:
                    count.pop()
    else:
        count = [(p[0],distance(point,p[1])) for p in sector]
    ids = set(i[0] for i in count)
    for i, p in enumerate(Xs):
        if i in ids:
            plt.scatter(p[0], p[1], color=COLOR_CONV[Ys[i]])
            continue
        plt.scatter(p[0], p[1], color='grey')
    plt.scatter(point[0], point[1], color='black')
    plt.axvline(x=medians[0], color='black')
    plt.axhline(y=medians[1], color='black')
    plt.show()
    labels = [Ys[i] for i in ids]
    c = Counter(labels)
    guess = max(c, key=c.get)

    print(f"that points {COLOR_CONV[guess]}, i'm like {100 * c[guess] / len(ids):.2f}% sure")

BasicKN((54,60))
KDTree((54,60))