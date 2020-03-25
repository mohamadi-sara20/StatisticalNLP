import copy
import json
import numpy as np
import os
import re
import shutil
from collections import Counter
from hazm import *
from math import log10


class NaiveBayesTextClassifier():
    def __init__(self, corpusPath, model):
        self.corpusPath = corpusPath
        self.topics = [ ]
        for fileName in os.listdir(self.corpusPath):
            if not (fileName.startswith('.') or os.path.isfile("{}/{}".format(corpusPath, fileName))):
                self.topics.append(fileName)

        self.classProbability = {}
        # Set class probabilities
        for topic in self.topics:
            self.classProbability[ topic ] = 1 / len(self.topics)

    # نرمال‌کننده پیکره که عمدتا برای یکسان‌سازی صورت‌های مختلف حروفی مثل «ک» و «ی» و امثال آن،
    # یکسان‌سازی انواع فاصله، و تبدیل اعداد لاتین به اعداد عربی استفاده می‌شود.
    def _normalize(self, text, removePunc=True, removeStopWords=True):
        with open('CharacterMapping.json', encoding='utf8') as j:
            characterMapping = json.load(fp=j)
        for ch in characterMapping:
            text = re.sub(chr(int(ch)), characterMapping[ ch ], text)
        text = re.sub(r'[A-Za-z]', ' ', text)
        if removeStopWords:
            text = self._removeStopwords(text)
        if removePunc:
            text = re.sub(r'[.،,:؛"»«!؟?@#)({}+=*_/><|&]', ' ', text)
            text = text.replace('[', ' ')
            text = text.replace(']', ' ')
        text = re.sub('[۰-۹0-9]', ' ', text)
        text = re.sub(' {2,}', ' ', text)
        text = text.strip()
        return text

    def _removeStopwords(self, doc):
        with open('StopWords.txt', encoding='utf8') as f:
            stopWords = f.read()
        stopWordsSet = stopWords.split('\n')

        newDoc = [ ]
        for word in doc.split():
            if word not in stopWordsSet:
                newDoc.append(word)
        return ' '.join(newDoc)

    def splitTrainTest(self):
        for topic in self.topics:
            catFiles = [ ]
            for fileName in os.listdir(self.corpusPath + '/' + topic):
                if not (fileName.startswith('.') or os.path.isfile("{}/{}/{}".format(self.corpusPath, self.topics, fileName))):
                    catFiles.append(fileName)

            catFiles.sort()
            if not os.path.exists(f'./Test/{topic}'):
                os.mkdir(f'./Test/{topic}')
            testFilePath1 = f'./ZebRa/{topic}/{catFiles[ 0 ]}'
            testFilePath2 = f'./ZebRa/{topic}/{catFiles[ 1 ]}'
            shutil.copy(testFilePath1, f'./Test/{topic}')
            shutil.copy(testFilePath2, f'./Test/{topic}')
            os.remove(testFilePath1)
            os.remove(testFilePath2)

    def mergeCategoryFiles(self):
        if not os.path.exists('./MergedCats'):
            os.mkdir('./MergedCats')

        for topic in self.topics:
            catContent = ''
            fileNames = os.listdir(f'./ZebRa/{topic}')
            for file in fileNames:
                with open(f'./ZebRa/{topic}/{file}', encoding='utf8') as f:
                    catContent += f.read() + '\n'
                    catContent = re.sub('\n{2,}', '\n', catContent)
            catContent = self._normalize(catContent)
            with open(f'./MergedCats/{topic}', 'w', encoding='utf8') as g:
                g.write(catContent)

    def calculateConditionalProbabilities(self):
        lemmer = Lemmatizer()
        content = ''
        for cat in self.topics:
            with open(f'MergedCats/{cat}', encoding='utf8') as f:
                content = f.read()
        content = self._normalize(content)
        words = content.split()
        lemmatizedWords = [ ]
        for word in words:
            lemma = lemmer.lemmatize(word).split('#')[ -1 ]
            lemmatizedWords.append(lemma)
        lemmatizedWords = self._normalize(' '.join(lemmatizedWords)).split()
        lemmatizedWords = filter(None, lemmatizedWords)
        counts = list(Counter(lemmatizedWords).items())
        counts.sort(key=lambda tup: tup[ 1 ], reverse=True)
        lexicon = [ c[0] for c in counts ][:1000]

        condProbs = {}
        for cat in self.topics:
            condProbs[ cat ] = {}
            with open(f'MergedCats/{cat}', encoding='utf8') as f:
                catContent = f.read()
                catContent = self._normalize(catContent)
            catWords = catContent.split()
            lemmatizeds = [ ]
            for word in catWords:
                lemma = lemmer.lemmatize(word).split('#')[ -1 ]
                if lemma in lexicon:
                    lemmatizeds.append(lemma)

            for word in lemmatizeds:
                condProbs[ cat ][ word ] = condProbs[ cat ].get(word, 0) + 1
            for word in lexicon:
                condProbs[ cat ][ word ] = condProbs[ cat ].get(word, 0)

        for cat in self.topics:
            catSum = sum(condProbs[ cat ].values()) + len(lexicon)
            for k in condProbs[ cat ]:
                condProbs[ cat ][ k ] = (condProbs[ cat ][ k ] + 1) / catSum
        return condProbs

    def dumpConditionalProbsToJson(self, fileName, condProbs):
        with open(fileName, 'w', encoding='utf8') as j:
            json.dump(condProbs, j, ensure_ascii=False, indent=4)

    def loadConditionalProbsJson(self, fileName):
        with open(fileName, encoding='utf8') as j:
            condProbs = json.load(j)
        return condProbs

    def classifyTestDoc(self, testFilePath, condProbs):
        with open(testFilePath, encoding='utf8') as f:
            docContent = self._normalize(f.read())
            docWords = docContent.split()

        lemmer = Lemmatizer()
        catProbs = {}
        for cat in self.topics:
            catProbs[ cat ] = 0
            for word in docWords:
                lemma = lemmer.lemmatize(word).split('#')[ -1 ]
                catProbs[ cat ] -= log10(condProbs[ cat ].get(lemma, 1))
            chosenCat = [ catProbs[ cat ], cat ]

        for cat in catProbs:
            if catProbs[ cat ] < chosenCat[ 0 ]:
                chosenCat[ 0 ] = catProbs[ cat ]
                chosenCat[ 1 ] = cat
        return chosenCat

    def evaluate(self, testPath, condProbs):
        evalPerClass = {}
        matrixIndices = {}
        cmLength = 0

        for folder in self.topics:
            cmLength += 1
            matrixIndices[ folder ] = len(matrixIndices)

        confusionMatrix = np.zeros((cmLength, cmLength))

        for folder in self.topics:
            files = os.listdir(testPath + '/' + folder)
            for file in files:
                if file.startswith('.') or os.path.isdir("{}/{}/{}".format(testPath, folder, file)):
                    continue
                classification = self.classifyTestDoc(f'./Test/{folder}/{file}', condProbs)[ 1 ]
                row = matrixIndices[ folder ]
                col = matrixIndices[ classification ]
                confusionMatrix[ row ][ col ] += 1

        # Overall Precision and Recall
        numerator = np.trace(confusionMatrix)
        # compute recall from confusion matrix
        sum1 = confusionMatrix.sum(axis=0)
        den1 = 0
        for i in sum1:
            den1 += i

        microPrecision = numerator / den1
        # compute precision from confusion matrix
        sum2 = confusionMatrix.sum(axis=1)
        den2 = 0
        for i in sum2:
            den2 += i

        microRecall = numerator / den2
        # compute f measure
        microFScore = (2 * microPrecision * microRecall) / (microRecall + microPrecision)

        confMatRowSum = np.sum(confusionMatrix, axis=1)
        confMatColSum = np.sum(confusionMatrix, axis=0)

        sumPrecision = 0
        sumRecall = 0
        count = 0
        # Precision and Class per class
        for key in matrixIndices:
            count += 1
            ind = matrixIndices[ key ]
            numerator = confusionMatrix[ ind ][ ind ]
            den1 = confMatColSum[ ind ]
            den2 = confMatRowSum[ ind ]
            if den1 == 0:
                precision = 0
                recall = numerator / den2


            elif den2 == 0:
                recall = 0
                precision = numerator / den1

            else:
                precision = numerator / den1
                recall = numerator / den2
            if precision + recall != 0:
                f1Score = (2 * precision * recall) / (precision + recall)
            else:
                f1Score = 0
            sumRecall += recall
            sumPrecision += precision
            evalPerClass[ key ] = {'Precision': precision, 'Recall': recall, 'F-score': f1Score}
        macroPrecision = sumPrecision / count
        macroRecall = sumRecall / count
        macroFscore = (2 * macroPrecision * macroRecall) / (macroRecall + macroPrecision)
        
        print('\n\nMatrix Indices: \n')
        print(matrixIndices)
        print('\n\n\nConfusion Matrix: ')
        print(confusionMatrix)
        print()

        return microPrecision, microRecall, microFScore, macroPrecision, macroRecall, macroFscore, evalPerClass




if __name__ == '__main__':
    nbc = NaiveBayesTextClassifier('./ZebRa', model='./probs.json')
    #nbc.splitTrainTest()
    nbc.mergeCategoryFiles()
    condProbs = nbc.calculateConditionalProbabilities()
    nbc.dumpConditionalProbsToJson('probs.json', condProbs)
    microPrecision, microRecall, microFScore, macroPrecision, macroRecall,macroFscore, evalPerClass = nbc.evaluate('./Test', condProbs)
    
    # Evaluation Reports : Micro, Micro, Per class
    evaluationReport = f'\nOverall Micro Precision: {microPrecision:.3f}\nOverall Micro Recall: {microRecall:.3f}\nOverall Micro F-score:{microFScore:.3f}\n\nOverall Macro Precision: {macroPrecision:.3f}\nOverall Macro Recall: {macroRecall:.3f}\nOverall Macro F-score: {macroFscore:.3f}\n\n\n'
    for c in evalPerClass:
        print(f"{c}:\nPrecision: {evalPerClass[c]['Precision']:.3f}\tRecall: {evalPerClass[c]['Recall']:.3f}\tF-Score: {evalPerClass[c]['F-score']:.3f}")
    print(evaluationReport)
