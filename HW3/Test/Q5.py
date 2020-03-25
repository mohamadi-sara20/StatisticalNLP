#!/usr/bin/env python3

n = int(input())
centers = []
for i in range(n):
    center = input().split()
    center[0] = float(center[0])
    center[1] = float(center[1])
    centers.append(center)
while True:
    xy = input().split()
    if not xy:
        break

    x = float(xy[0])
    y = float(xy[1])
    dists = []
    for center in centers:
        dist = (x - center[0]) ** 2 + (y - center[1]) ** 2
        dists.append(dist)
    chosenCluster = dists.index(min(dists))
    print(chosenCluster)
