import json
import re
from collections import Counter
from math import log2, sqrt
from os import path, walk

import matplotlib.pyplot as plt
import numpy as np
import scipy.stats as stats


class PersianEntropy:
    def __init__(self, corpusDirectoryPath):
        self.corpusDirectoryPath = corpusDirectoryPath
        self.rawCorpusContent = self._mergeCorpusFiles()

    # بخش الف
    # بدون در نظر گرفتن فاصله عدد ۳۲ و با در نظر گرفتن فاصله عدد ۳۳ را به متد می‌دهیم.
    def calculateFZero(self, numberOfLetters):
        entropy = 0
        p = 1 / numberOfLetters
        for i in range(numberOfLetters):
            entropy -= p * log2(p)
        return entropy

    def _mergeCorpusFiles(self):
        corpusContent = ''
        allFilesPaths = [ path.join(dp, f) for dp, dn, filenames in walk(self.corpusDirectoryPath) for f in filenames if
                          path.splitext(f)[ 1 ] == '.txt' ]
        allFiles = [ ]
        for file in allFilesPaths:
            with open(file) as f:
                allFiles.append(f.read())
        for file in allFiles:
            corpusContent += file
        return corpusContent

    # نرمال‌کننده پیکره که عمدتا برای یکسان‌سازی صورت‌های مختلف حروفی مثل «ک» و «ی» و امثال آن،
    # یکسان‌سازی انواع فاصله، و تبدیل اعداد لاتین به اعداد عربی استفاده می‌شود.
    def normalizeCorpus(self):
        corpusContent = self.rawCorpusContent
        with open('CharacterMapping.json') as j:
            characterMapping = json.load(fp=j)
        for ch in characterMapping:
            corpusContent = re.sub(chr(int(ch)), characterMapping[ ch ], corpusContent)
        corpusContent = re.sub(r'[A-Za-z]', '', corpusContent)
        return corpusContent

    # مربوط به بخش ب. این متد احتمال‌های هر حرف رادر پیکره محاسبه می‌کند.
    def _findLettersMonograms(self, normalized=True):
        corpusContent = self.rawCorpusContent
        if normalized:
            corpusContent = self.normalizeCorpus()
        allLetters = {}
        for letter in corpusContent:
            allLetters[ letter ] = allLetters.get(letter, 0) + 1
        total = sum(allLetters.values())
        for letter in allLetters:
            allLetters[ letter ] = allLetters[ letter ] / total
        # print(allLetters)
        return allLetters

    # پاسخ بخش ب
    def calculateMonogramEntropy(self, normalized=True):
        frequencies = self._findLettersMonograms(normalized=normalized)
        entropy = 0
        for letter in frequencies:
            p = frequencies[ letter ]
            entropy -= p * log2(p)
        return entropy

    # مربوط به بخش ج. این متد برای خارج کردن هر کاراکتری جز حروف استاندارد از فارسی است.
    def standardizeCorpus(self, removeSpace=True):
        corpusContent = self.normalizeCorpus()
        corpusContent = re.sub(r'\s', ' ', corpusContent)

        if removeSpace:
            corpusContent = re.sub(r' ', '', corpusContent)

        with open('NonAlphaCharacters.json') as j:
            nonStandardCharacters = json.load(fp=j)

        for ch in nonStandardCharacters:
            corpusContent = corpusContent.replace(ch, '')
        return corpusContent

    # مربوط به بخش ج - محاسبه احتمال‌های مونوگرام حروف استاندارد
    def _findStandardLettersMonograms(self, removeSpace=True):
        corpusContent = self.standardizeCorpus(removeSpace=removeSpace)
        allLetters = {}
        for letter in corpusContent:
            allLetters[ letter ] = allLetters.get(letter, 0) + 1
        total = sum(allLetters.values())
        for letter in allLetters:
            allLetters[ letter ] = allLetters[ letter ] / total
        return allLetters

    # پاسخ بخش ج - محاسبه آنتروپی برای حروف استاندارد
    def calculateStandardEntropy(self, removeSpace=True):
        frequencies = self._findStandardLettersMonograms(removeSpace=removeSpace)
        entropy = 0
        for letter in frequencies:
            p = frequencies[ letter ]
            entropy -= p * log2(p)
        return entropy

    # مربوط به بخش د - محاسبه میانگین طول کلمات
    def findAverageWordLength(self):
        corpusContent = self.normalizeCorpus()
        words = corpusContent.split()
        wordLength = sum(len(word) for word in words)
        return wordLength / len(words)

    # پاسخ بخش د - محاسبه آنتروپی برای کلمه
    def calculateEntropyForWord(self, removeSpace=False):
        wordLength = self.findAverageWordLength()
        letterEntropy = self.calculateStandardEntropy(removeSpace=removeSpace)
        return wordLength * letterEntropy

    # بخش ه - هیستوگرام حروف فارسی
    def drawLetterProbabilityHistogram(self):
        probabilities = self._findStandardLettersMonograms()
        probabityTuples = [ ]
        for i in probabilities:
            probabityTuples.append((i, probabilities[ i ]))
        probabityTuples.sort(key=lambda x: x[ 1 ], reverse=True)
        letters = [ l[ 0 ] for l in probabityTuples ]
        probs = [ l[ 1 ] for l in probabityTuples ]
        plt.bar(letters, probs)
        plt.xlabel('Letter')
        plt.ylabel('Probablity')
        plt.title('Persian Letters Probability')
        plt.show()
        return

    # بخش و - محاسبه فراوانی طول کلمات
    def findWordLengthProbability(self):
        wordLenght = {}
        corpusContent = self.normalizeCorpus()
        words = corpusContent.split()
        for word in words:
            wordLenght[ len(word) ] = wordLenght.get(len(word), 0) + 1
        total = sum(wordLenght.values())
        for l in wordLenght:
            wordLenght[ l ] = wordLenght[ l ] / total
        return wordLenght

    # بخش و - کشیدن  هیستوگرام طول کلمات
    def drawWordLenghtProbabilityHistogram(self):
        probabilities = self.findWordLengthProbability()
        probabityTuples = [ ]
        for i in probabilities:
            probabityTuples.append((i, probabilities[ i ]))
        probabityTuples.sort(key=lambda x: x[ 1 ], reverse=True)
        wl = [ l[ 0 ] for l in probabityTuples ]
        probs = [ l[ 1 ] for l in probabityTuples ]
        plt.bar(wl, probs)
        plt.xlabel('Word Length')
        plt.ylabel('Probablity')
        plt.title('Persian Word Length Probability')
        plt.show()

    # مربوط به بخش و - محاسبه پارامترهای تابع گوسی
    def calculateGaussianParameters(self):
        data = self.findWordLengthProbability()
        mean = 0
        eX2 = 0
        for i in data:
            mean += data[ i ] * i
            eX2 += data[ i ] * (i * i)
        var = eX2 - mean * mean
        return mean, var

    # مربوط به بخش و - رسم تابع گوسی
    def drawGaussianCurve(self):
        mu, var = self.calculateGaussianParameters()
        s = sqrt(var)
        probabilities = self.findWordLengthProbability()
        probabityTuples = [ ]
        for i in probabilities:
            probabityTuples.append((i, probabilities[ i ]))
        probabityTuples.sort(key=lambda x: x[ 1 ], reverse=True)
        wl = [ l[ 0 ] for l in probabityTuples ]
        probs = [ l[ 1 ] for l in probabityTuples ]
        plt.bar(wl, probs)
        plt.xlabel('Word Length')
        plt.ylabel('Probablity')
        plt.title('Persian Word Length Probability')
        x = np.linspace(0, 20, 100)
        plt.plot(x, stats.norm.pdf(x, mu, s), 'r')
        plt.show()

    # محاسبه بایگرام‌ها
    def findBigramsProbability(self, normalized=True, removeSpace=False):
        corpusContent = self.rawCorpusContent
        corpusContent = self.standardizeCorpus(removeSpace=removeSpace)
        bigrams = Counter(x + y for x, y in zip(*[ corpusContent[ i: ] for i in range(2) ]))
        bigramFreq = {}
        for bigram in bigrams.keys():
            count = bigrams[ bigram ]
            bigramFreq[ bigram ] = count
        total = sum(bigramFreq.values())
        for b in bigramFreq:
            bigramFreq[ b ] = bigramFreq[ b ] / total
        return bigramFreq

    # محاسبه آنتروپی بر حسب احتمال‌های بایگرام
    def calculateF2(self, removeSpace=False):
        monograms = self._findStandardLettersMonograms(removeSpace=removeSpace)
        bigrams = self.findBigramsProbability(removeSpace=removeSpace)
        entropy = 0
        for i in monograms:
            pi = monograms[ i ]
            for j in monograms:
                pj = monograms[ j ]
                if i + j in bigrams:
                    pi_j = bigrams[ i + j ] / pj
                    entropy -= bigrams[ i + j ] * log2(pi_j)
        return entropy

    # محاسبه تریگرام‌ها
    def findTrigramsProbability(self, normalized=True, removeSpace=False):
        corpusContent = self.rawCorpusContent
        if normalized:
            corpusContent = self.normalizeCorpus()
        if removeSpace:
            corpusContent = re.sub(r' ', '', corpusContent)
        from collections import Counter
        trigrams = Counter(x + y + z for x, y, z in zip(*[ corpusContent[ i: ] for i in range(3) ]))
        total = sum(trigrams.values())
        for t in trigrams:
            trigrams[ t ] = trigrams[ t ] / total
        return trigrams

    # محاسبه آنتروپی بر حسب احتمال‌های تریگرام
    def calculateF3(self, removeSpace=False):
        monograms = self._findStandardLettersMonograms(removeSpace=removeSpace)
        bigrams = self.findBigramsProbability(removeSpace=removeSpace)
        trigrams = self.findTrigramsProbability(removeSpace=removeSpace)
        entropy = 0
        for i in bigrams:
            pi = bigrams[ i ]
            for j in monograms:
                pj = monograms[ j ]
                if i + j in trigrams:
                    pj_i = trigrams[ i + j ] / pi
                    entropy -= trigrams[ i + j ] * log2(pj_i)
        return entropy


if __name__ == "__main__":
    per = PersianEntropy('./ZebRa')
    print('خروجی قسمت الف - بدون فاصله')
    print(per.calculateFZero(32))
    print('خروجی قسمت الف - با فاصله')
    print(per.calculateFZero(33))
    print('خروجی قسمت ب - آنتروپی تمام حروف بدون نرمال کردن')
    print(per.calculateMonogramEntropy(normalized=False))
    print('خروجی قسمت ب - آنتروپی تمام حروف با نرمال کردن')
    print(per.calculateMonogramEntropy(normalized=True))
    print('خروجی قسمت ج - با در نظر گرفتن فاصله')
    print(per.calculateStandardEntropy(removeSpace=False))
    print('خروجی قسمت ج - بدون در نظر گرفتن فاصله')
    print(per.calculateStandardEntropy(removeSpace=True))
    print('خروجی قسمت د - متوسط طول کلمات پیکره')
    print(per.findAverageWordLength())
    print('خروجی قسمت د - آنتروپی کلمه با فاصله = آنتروپی حرف با احتساب فاصله *‌ طول کلمه')
    print(per.calculateEntropyForWord(removeSpace=False))
    print('خروجی قسمت د - آنتروپی کلمه بدون فاصله = آنتروپی حرف بدون احتساب فاصله *‌ طول کلمه')
    print(per.calculateEntropyForWord(removeSpace=True))
    # print('قسمت ه - احتمالات طول کلمات')
    # print(per.findWordLengthProbability())
    # خروجی قسمت ه : هیستوگرام احتمالات حروف
    per.drawLetterProbabilityHistogram()
    # خروجی قسمت و - هیستوگرام طول کلمات
    per.drawWordLenghtProbabilityHistogram()
    print('خروجی قسمت و - محاسبه پارامترهای تابع گوسی')
    print(per.calculateGaussianParameters())
    # خروجی قسمت و - رسم منحنی گوسی و فیت کردن آن روی هیستوگرام طول کلمات
    per.drawGaussianCurve()

    print('\n\n\n**************************************************')
    print('محاسبه F0 تا F3 در پیکره زبرا برای زبان فارسی:')
    # محاسبه  F2 , F3 برای فارسی طبق پیکره زبرا
    print('F0 with space:')
    print(per.calculateFZero(numberOfLetters=33))
    print('F0 without space:')
    print(per.calculateFZero(numberOfLetters=32))
    print('F1 with space:')
    print(per.calculateStandardEntropy(removeSpace=False))
    print('F1 without space:')
    print(per.calculateStandardEntropy(removeSpace=True))
    print('F2 with space:')
    print(per.calculateF2(removeSpace=False))
    print('F2 without space:')
    print(per.calculateF2(removeSpace=True))
    print('F3 with space:')
    print(per.calculateF3(removeSpace=False))
    print('F3 without space:')
    print(per.calculateF3(removeSpace=True))
    print('**************************************************')
