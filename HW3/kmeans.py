import numpy as np
import json
import re
import os
import glob
import random
from scipy.spatial.distance import cosine as cosineDist
import matplotlib.pyplot as plt
from copy import deepcopy


class KmeansTextClustering():
    def __init__(self, datasetPath, k, tolerance, maxIterations):
        self.datasetPath = datasetPath
        self.idf = {}
        self.lexicon = {}
        self.k = k
        self.centroids = {}
        self.tolerance = tolerance
        self.maxIterations = maxIterations
        self.clusters = {}
        self.fileClass = []
        
        # نرمال‌کننده پیکره که عمدتا برای یکسان‌سازی صورت‌های مختلف حروفی مثل «ک» و «ی» و امثال آن،
    # یکسان‌سازی انواع فاصله، و تبدیل اعداد لاتین به اعداد عربی استفاده می‌شود.
    def _normalize(self, text, removePunc=True, removeStopWords=True):
        with open('CharacterMapping.json') as j:
            characterMapping = json.load(fp=j)
        for ch in characterMapping:
            text = re.sub(chr(int(ch)), characterMapping[ ch ], text)
        text = re.sub(r'[A-Za-z]', ' ', text)
        if removePunc:
            text = re.sub(r'[.,،:؛"»«!؟?@&#)({}+\-=*_/><|]', ' ', text)
            text = text.replace('[', ' ')
            text = text.replace(']', ' ')
        if removeStopWords:
            text = self._removeStopwords(text)
        text = re.sub('[0-9۰-۹]', ' ', text)
        text = re.sub(' {2,}', ' ', text)
        text = text.strip()
        return text

    def _removeStopwords(self, doc):
        with open('StopWords.txt') as f:
            stopWords = f.read()
        stopWordsSet = stopWords.split('\n')
        newDoc = [ ]
        for word in doc.split():
            if word not in stopWordsSet:
                newDoc.append(word)
        return ' '.join(newDoc)

    def outputNormalizeCorpus(self, outfilename):
        folders = os.listdir(f'./{self.datasetPath}')
        corpusContent = ''
        for folder in folders:
            if not folder.startswith('.'):
                textFiles = glob.glob(f"./{self.datasetPath}/{folder}/*.txt")
                for file in textFiles:
                    with open(file) as f:
                        corpusContent += f.read()
        with open(outfilename, 'w') as f:
            f.write(self._normalize(corpusContent))

    # پاسخ بخش الف
    def selectLexicon(self, normalizedCorpusName):
        with open(normalizedCorpusName) as f:
            data = f.read()
        normalizedLexicon = set(data.split())
        for word in normalizedLexicon:
            self.lexicon[word] = self.lexicon.get(word, len(self.lexicon))
        with open('Dic.txt', 'w') as f:
            for word in normalizedLexicon:
                f.write(word)
                f.write('\n')

    def doc2vec(self, doc):
        words = self._normalize(doc).split()
        vec = np.zeros(len(self.lexicon))
        for word in words:
            indWord = self.lexicon[ word ]
            vec[ indWord ] += 1
        vec[:] += 1
        return np.log(vec)

    def convertAllDocs(self):
        folders = os.listdir(f'./{self.datasetPath}')
        allSamples = []
        for folder in folders:
            if not folder.startswith('.'):
                textFiles = glob.glob(f"./{self.datasetPath}/{folder}/*.txt")
                textFiles.sort()
                for file in textFiles:
                    with open(file) as f:
                        fContent = self._normalize(f.read())
                    allSamples.append(self.doc2vec(fContent))
                    self.fileClass.append(folder)
        return allSamples

    def kmeansClustering(self, allSamples, drawErrorPlot=False):
        iteration = 0
        centroidInds = []
        # Centroids Random Initialization
        while len(centroidInds) < self.k:
            j = random.randint(0, len(allSamples)-1)
            if j not in centroidInds:
                centroidInds.append(j)

        for i in range(len(centroidInds)):
            self.centroids[i] = allSamples[centroidInds[i]]

        errorPerIteration = {}
        while True:
            error = 0
            self.clusters = {i:[] for i in range(len(self.centroids))}
            # assignment (expectation step)
            for sample in allSamples:
                chosenCluster = -2000
                chosenDist = 1000
                for i in range(len(self.centroids)):
                    d = abs(cosineDist(self.centroids[i], sample))
                    if d < chosenDist:
                        chosenDist = d
                        chosenCluster = i
                error += chosenDist
                self.clusters[chosenCluster].append(sample)
            errorPerIteration[iteration] = error
            # maximization
            prev_centroids = deepcopy(self.centroids)
            for c in self.clusters:
                if len(self.clusters[c]) > 0:
                    self.centroids[c] = np.average(self.clusters[c], axis=0)

            # stop condition checking
            optimized = True
            changes = []
            for c in self.centroids:
                original_centroid = prev_centroids[c]
                current_centroid = self.centroids[c]
                dist = abs(cosineDist(original_centroid, current_centroid))
                # print(dist)
                changes.append(dist)

            if max(changes) > self.tolerance:
                optimized = False
            iteration += 1
            if optimized:
                break
            if iteration >= self.maxIterations:
                break

        # Plot error per iteration
        if drawErrorPlot:
            plt.xlabel('Iteration')
            plt.ylabel('SSE')
            plt.plot(list(errorPerIteration.keys()), list(errorPerIteration.values()), 'ro')
            plt.title('SSE by Iteration')
            plt.show()
        return self.clusters

    def computePurity(self, allSamples):
        sum = 0
        for cluster in self.clusters:
            sampleClasses = {}
            for doc in self.clusters[cluster]:
                docInd = -1
                for i in range(len(allSamples)):
                    if (allSamples[i] == doc).all():
                        docInd = i
                        break
                docClass = self.fileClass[docInd]
                sampleClasses[docClass] = sampleClasses.get(docClass, 0) + 1
            sum += max(sampleClasses.values())
        return sum/len(allSamples)


# پاسخ بخش الف - خوشه‌بندی با k=7 و کشیدن نمودار خطا و گزارش خلوص
km = KmeansTextClustering('ZebRa', 7, 0.000001, 1000)
km.outputNormalizeCorpus('NormalizedZebra')
km.selectLexicon('NormalizedZebra')
allSamples = km.convertAllDocs()
clusts = km.kmeansClustering(allSamples=allSamples, drawErrorPlot=True)
purity = km.computePurity(allSamples=allSamples)
print('#######################################\nPurity with k = 7 this time:')
print(purity)

# پاسخ بخش الف - خوشه‌بندی با k=14 و گزارش خلوص
km = KmeansTextClustering('ZebRa', 14, 0.000001, 1000)
km.outputNormalizeCorpus('NormalizedZebra')
km.selectLexicon('NormalizedZebra')
allSamples = km.convertAllDocs()
clusts = km.kmeansClustering(allSamples=allSamples)
purity = km.computePurity(allSamples=allSamples)
print('#######################################\nPurity with k = 14 this time:')
print(purity)

# پاسخ بخش الف - خوشه‌بندی با k=70 و گزارش خلوص
km = KmeansTextClustering('ZebRa', 70, 0.000001, 1000)
km.outputNormalizeCorpus('NormalizedZebra')
km.selectLexicon('NormalizedZebra')
allSamples = km.convertAllDocs()
clusts = km.kmeansClustering(allSamples=allSamples)
purity = km.computePurity(allSamples=allSamples)
print('#######################################\nPurity with k = 70 this time:')
print(purity)
