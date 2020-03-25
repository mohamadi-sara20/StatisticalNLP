#!/usr/bin/env python3

tp = 0
tn = 0
fn = 0
fp = 0
while True:
    numbers = input()
    if not numbers:
        break
    real, pred = numbers.split()
    real = int(real)
    pred = int(pred)
    if real == pred == 1:
        tp += 1
    elif real == pred == 0:
        tn += 1
    elif real == 1 and pred == 0:
        fn += 1
    else:
        fp += 1

precision = tp / (tp + fp)
recall = tp / (tp + fn)
f1 = (2 * precision * recall) / (precision + recall)
print(f'{f1:.2f}')
