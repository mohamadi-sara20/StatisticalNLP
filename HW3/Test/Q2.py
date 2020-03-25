#!/usr/bin/env python3

def findDistance(s1, s2):
    if len(s1) != len(s2):
        return ''
    if not len(s1):
        return 0
    dist = ord(s2[0]) - ord(s1[0])
    for i in range(1, len(s1)):
        if ord(s2[i]) - ord(s1[i]) != dist:
            return ''
    return dist


words = input()
words = words.split()
if len(words) != 2:
    print('')
print(findDistance(words[0], words[1]))
