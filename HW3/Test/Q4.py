#!/usr/bin/env python3

n = int(input())
data = {}
for i in range(n):
    kv = input()
    kv = kv.split()
    k = int(kv[0])
    v = kv[1]
    data[k] = v

sortedKeys = sorted(data.keys())
for i in (sortedKeys):
    print(data[i])
