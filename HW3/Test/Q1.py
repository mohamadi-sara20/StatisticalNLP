#!/usr/bin/env python3

def computeWhiteExpectation(n):
    e = (1 / 2) ** n
    eString = '{0:1.2e}'.format(e)
    return eString[0] + eString[2]


n = int(input())
print(computeWhiteExpectation(n))
