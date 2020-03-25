import json
import numpy as np
import re
from math import log, sqrt, pow

class SimilarityDetector():
    def __init__(self, datasetPath):
        self.datasetPath = datasetPath
        self.idf = {}
        

        # نرمال‌کننده پیکره که عمدتا برای یکسان‌سازی صورت‌های مختلف حروفی مثل «ک» و «ی» و امثال آن،
    # یکسان‌سازی انواع فاصله، و تبدیل اعداد لاتین به اعداد عربی استفاده می‌شود.
    def _normalize(self, text, removePunc=True, removeStopWords=True):
        with open('CharacterMapping.json') as j:
            characterMapping = json.load(fp=j)
        for ch in characterMapping:
            text = re.sub(chr(int(ch)), characterMapping[ ch ], text)
        text = re.sub(r'[A-Za-z]', ' ', text)
        if removePunc:
            text = re.sub(r'[.,،:؛"»«!؟?@#)({}+\-=*_/><|]', ' ', text)
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

    def _prepareSentenceSet(self):
        with open(self.datasetPath) as f:
            lines = f.readlines()
        sentences = [ ]
        for l in lines:
            sentences.append(tuple(l.split('\t')))
        return sentences

    # پاسخ بخش الف
    def outputNormalizeCorpus(self, outfileName):
        sentences = self._prepareSentenceSet()
        normalizedSentences = [ ]
        for i in range(len(sentences)):
            normalizedSentences.append((self._normalize(sentences[ i ][ 0 ]), self._normalize(sentences[ i ][ 1 ])))
        with open(outfileName, 'w') as f:
            for i in normalizedSentences:
                f.write(i[ 0 ])
                f.write(',')
                f.write(i[ 1 ])
                f.write('\n')

    # پاسخ بخش الف
    def selectLexicon(self, normalizedCorpusName):
        with open(normalizedCorpusName) as f:
            data = f.read()
            normaliData = self._normalize(data)
        normalizedLexicon = set(normaliData.split())
        with open('Dic.txt', 'w') as f:
            for word in normalizedLexicon:
                f.write(word)
                f.write('\n')

    # پاسخ بخش ب
    def computeTFSimilarity(self, s1, s2):
        words1 = self._normalize(s1).split()
        words2 = self._normalize(s2).split()
        mergedSentences = words1 + words2
        lexicon = {}
        for i in range(len(mergedSentences)):
            if mergedSentences[ i ] not in lexicon:
                lexicon[ mergedSentences[ i ] ] = len(lexicon)
        # Form and fill matrices 
        vec1 = np.zeros(len(lexicon))
        vec2 = np.zeros(len(lexicon))
        for word in words1:
            indWord = lexicon[ word ]
            vec1[ indWord ] += 1
        for word in words2:
            indWord = lexicon[ word ]
            vec2[ indWord ] += 1

        # Normalize TFs
        vec1 /= np.sum(vec1)
        vec2 /= np.sum(vec2)

        # Compute Similarity
        cosineSimilarity = np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))
        jacardSimilarity = np.dot(vec1, vec2) / (np.linalg.norm(vec1) + np.linalg.norm(vec2) - np.dot(vec1, vec2))

        return cosineSimilarity, jacardSimilarity


    # پاسخ بخش ب
    def computeAverageTFSimilarity(self):
        sentences = self._prepareSentenceSet()
        sumCosTF = 0
        sumJacardTF = 0
        for i in range(len(sentences)):
            s1, s2, simi = sentences[ i ][ 0 ], sentences[ i ][ 1 ], float(sentences[ i ][ 2 ])
            cosineSim, jacardSim = self.computeTFSimilarity(s1, s2)
            sumCosTF += cosineSim
            sumJacardTF += jacardSim
        cosAverageTF = sumCosTF/len(sentences)
        jacardAverageTF = sumJacardTF/len(sentences)
        return cosAverageTF, jacardAverageTF

    
    # پاسخ بخش ج
    def computeTFIDFSimilarity(self, s1, s2):
        words1 = self._normalize(s1).split()
        words2 = self._normalize(s2).split()

        mergedSentences = words1 + words2
        lexicon = {}
        for i in range(len(mergedSentences)):
            if mergedSentences[ i ] not in lexicon:
                lexicon[ mergedSentences[ i ] ] = len(lexicon)
        vec1 = np.zeros(len(lexicon))
        vec2 = np.zeros(len(lexicon))

        for word in words1:
            indWord = lexicon[ word ]
            vec1[ indWord ] += 1

        for word in words2:
            indWord = lexicon[ word ]
            vec2[ indWord ] += 1

        vec1 /= np.sum(vec1)
        vec2 /= np.sum(vec2)

        for word in words1:
            indWord = lexicon[ word ]
            vec1[ indWord ] *= self.idf[ word ]

        for word in words2:
            indWord = lexicon[ word ]
            vec2[ indWord ] *= self.idf[ word ]
        cosineSimilarity = np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))
        vec1Hat = vec1 / np.linalg.norm(vec1)
        vec2Hat = vec2 / np.linalg.norm(vec2)
        jacardSimilarity = np.dot(vec1Hat, vec2Hat) / (2 - np.dot(vec1Hat, vec2Hat))
        
        return cosineSimilarity, jacardSimilarity

    # بخشی از پاسخ قسمت ج
    def computeIDF(self, normalizedCorpus):
        with open('Dic.txt') as f:
            lexicon = f.read().split('\n')
        with open(normalizedCorpus) as f:
            corpus = f.read()
        corpusDocs = list(filter(None, re.split('[,\n]', corpus)))
        X = np.zeros((len(corpusDocs), len(lexicon)))
        vocabulary = {}
        for i in range(len(lexicon)):
            if lexicon[ i ] not in vocabulary:
                vocabulary[ lexicon[ i ] ] = len(vocabulary)
    
        for i in range(len(corpusDocs)):
            docWords = self._normalize(corpusDocs[ i ]).split()
            for word in docWords:
                ind = vocabulary[ word ]
                X[ i ][ ind ] = 1

        del vocabulary['']
        lexicon = list(filter(None, lexicon))

        for word in lexicon:
            indWord = vocabulary[ word ]
            col = X[ :, [ indWord ] ]
            nj = np.sum(col)
            idf = log(len(lexicon) / nj)
            self.idf[ word ] = idf
        return self.idf

        # پاسخ بخش ج
    def computeAverageTFIDFSimilarity(self):
        sentences = self._prepareSentenceSet()
        sumCosTFIDF = 0
        sumJacardTFIDF = 0
        for i in range(len(sentences)):
            s1, s2, simi = sentences[ i ][ 0 ], sentences[ i ][ 1 ], float(sentences[ i ][ 2 ])
            cosineSim, jacardSim = self.computeTFIDFSimilarity(s1, s2)
            sumCosTFIDF += cosineSim
            sumJacardTFIDF += jacardSim
        cosAverageTFIDF = sumCosTFIDF/len(sentences)
        jacardAverageTFIDF = sumJacardTFIDF/len(sentences)
        return cosAverageTFIDF, jacardAverageTFIDF


    
    # پاسخ بخش د
    def computeCorrelationCoefficient(self, filename):
        sentences = self._prepareSentenceSet()
        realSimilarity = np.zeros((len(sentences)))
        cosineTFsimilarity = np.zeros((len(sentences)))
        jacardTFsimilarity = np.zeros((len(sentences)))

        cosineTFIDFsimilarity = np.zeros((len(sentences)))
        jacardTFIDFsimilarity = np.zeros((len(sentences)))

        for i in range(len(sentences)):
            s1, s2, simi = sentences[ i ][ 0 ], sentences[ i ][ 1 ], float(sentences[ i ][ 2 ])
            # Form real similarity vector
            realSimilarity[ i ] = simi

            # Form TF similarity vectors
            cosineTFsimi, jacardTFsimi = self.computeTFSimilarity(s1, s2)
            cosineTFsimilarity[ i ] = cosineTFsimi
            jacardTFsimilarity[ i ] = jacardTFsimi

            # Form TFIDF similarity vectors
            cosineTFIDFsimi, jacardTFIDFsimi = self.computeTFIDFSimilarity(s1, s2)
            cosineTFIDFsimilarity[ i ] = cosineTFIDFsimi
            jacardTFIDFsimilarity[ i ] = jacardTFIDFsimi

        cosCoefficientTF = np.corrcoef(realSimilarity, cosineTFsimilarity)[ 0 ][ 1 ]
        cosCoefficientTFIDF = np.corrcoef(realSimilarity, cosineTFIDFsimilarity)[ 0 ][ 1 ]
        jacardCoefficientTF = np.corrcoef(realSimilarity, jacardTFsimilarity)[ 0 ][ 1 ]
        jacardCoefficientTFIDF = np.corrcoef(realSimilarity, jacardTFIDFsimilarity)[ 0 ][ 1 ]

        return cosCoefficientTF, cosCoefficientTFIDF, jacardCoefficientTF, jacardCoefficientTFIDF

    # پاسخ آلترناتیو بخش د

    def computeCorrelationCoefficientFromScratch(self, filename):
        sentences = self._prepareSentenceSet()
        realSimilarity = np.zeros((len(sentences)))
        cosineTFsimilarity = np.zeros((len(sentences)))
        jacardTFsimilarity = np.zeros((len(sentences)))

        cosineTFIDFsimilarity = np.zeros((len(sentences)))
        jacardTFIDFsimilarity = np.zeros((len(sentences)))

        for i in range(len(sentences)):
            s1, s2, simi = sentences[ i ][ 0 ], sentences[ i ][ 1 ], float(sentences[ i ][ 2 ])
            # Form real similarity vector
            realSimilarity[ i ] = simi

            # Form TF similarity vectors
            cosineTFsimi, jacardTFsimi = self.computeTFSimilarity(s1, s2)
            cosineTFsimilarity[ i ] = cosineTFsimi
            jacardTFsimilarity[ i ] = jacardTFsimi

            # Form TFIDF similarity vectors
            cosineTFIDFsimi, jacardTFIDFsimi = self.computeTFIDFSimilarity(s1, s2)
            cosineTFIDFsimilarity[ i ] = cosineTFIDFsimi
            jacardTFIDFsimilarity[ i ] = jacardTFIDFsimi

        meanReal = np.mean(realSimilarity)
        meanCosineTF = np.mean(cosineTFsimilarity)
        meanJacardTF = np.mean(jacardTFsimilarity)
        meanCosineTFIDF = np.mean(cosineTFIDFsimilarity)
        meanJacardTFIDF = np.mean(jacardTFIDFsimilarity)

        varReal = np.var(realSimilarity)
        varCosineTF = np.var(cosineTFsimilarity)
        varJacardTF = np.var(jacardTFsimilarity, ddof=1)
        varCosineTFIDF = np.var(cosineTFIDFsimilarity, ddof=1)
        varJacardTFIDF = np.var(jacardTFIDFsimilarity, ddof=1)
        vars = [ varCosineTF, varCosineTFIDF, varJacardTF, varJacardTFIDF ]

        numerators = np.zeros((4))
        denominators = np.zeros((4))

        xi = [ 0, 0, 0, 0 ]
        varReal = 0
        varCosTF, varCosTFIDF, varJacardTF, varJacardTFIDF = 0, 0, 0, 0
        for i in range(len(realSimilarity)):
            y = realSimilarity[ i ] - meanReal
            xi[ 0 ] = cosineTFsimilarity[ i ] - meanCosineTF
            xi[ 1 ] = cosineTFIDFsimilarity[ i ] - meanCosineTFIDF
            xi[ 2 ] = jacardTFsimilarity[ i ] - meanJacardTF
            xi[ 3 ] = jacardTFIDFsimilarity[ i ] - meanJacardTFIDF
            for j in range(len(numerators)):
                numerators[ j ] += (xi[ j ] * y)

            varReal += pow(realSimilarity[ i ] - meanReal, 2)
            varCosTF += pow(cosineTFsimilarity[ i ] - meanCosineTF, 2)
            varCosTFIDF += pow(cosineTFIDFsimilarity[ i ] - meanCosineTFIDF, 2)
            varJacardTF += pow(jacardTFsimilarity[ i ] - meanJacardTF, 2)
            varJacardTFIDF += pow(jacardTFIDFsimilarity[ i ] - meanJacardTFIDF, 2)

        cosCoefTF = numerators[ 0 ] / ((sqrt(varCosTF) * sqrt(varReal)))
        cosCoefTFIDF = numerators[ 1 ] / ((sqrt(varCosTFIDF) * sqrt(varReal)))
        jacCoefTF = numerators[ 2 ] / ((sqrt(varJacardTF) * sqrt(varReal)))
        jacCoefTFIDF = numerators[ 3 ] / ((sqrt(varJacardTFIDF) * sqrt(varReal)))

        return cosCoefTF, cosCoefTFIDF, jacCoefTF, jacCoefTFIDF


if __name__ == '__main__':
    
    normalizedCorpusName = 'normalizedSimilarityCorpusSamples.csv'
    similarityDetector = SimilarityDetector('SimilarityCorpusSamples.csv')
    # پاسخ با بخش الف - نرمال کردن پیکره و ریختن آن در فایل دیگر
    similarityDetector.outputNormalizeCorpus(normalizedCorpusName)
    # پاسخ با بخش الف - استخراج واژگان پیکره
    similarityDetector.selectLexicon(normalizedCorpusName)
    # محاسبه IDF کلمات
    similarityDetector.computeIDF(normalizedCorpusName)
    # پاسخ بخش ب
    cosineAverageTF, jaccardAverageTF = similarityDetector.computeAverageTFSimilarity()
    print('################################\nAverage TF Similarity for Corpus:')
    print(f'Cosine: {cosineAverageTF:.3f}\nJaccard: {jaccardAverageTF:.3f}')
    print('################################\n')
    # پاسخ بخش ج
    cosineAverageTFIDF, jaccardAverageTFIDF = similarityDetector.computeAverageTFIDFSimilarity()
    print('@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@\nAverage TFIDF Similarity for Corpus:')
    print(f'Cosine: {cosineAverageTFIDF:.3f}\nJaccard: {jaccardAverageTFIDF:.3f}')
    print('@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@\n')
    # پاسخ بخش د
    print('&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&')
    print('Correlation Coefficients (Used numpy.corrcoef): ')
    cosCoefTF, cosCoefTFIDF, jacCoefTF, jacCoefTFIDF = similarityDetector.computeCorrelationCoefficient(normalizedCorpusName)
    print(f'Cosine Correlation Coefficient for TF: {cosCoefTF:.3f}\nCosine Correlation Coefficient for TFIDF: {cosCoefTFIDF:.3f}\nJaccard Correlation Coefficient for TF: {jacCoefTF:.3f}\nJaccard Correlation Coefficient for TFIDF:{jacCoefTFIDF:.3f}')
    print('********************************')
    print('Correlation Coefficients (from scratch): ')
    cosCoefTF, cosCoefTFIDF, jacCoefTF, jacCoefTFIDF = similarityDetector.computeCorrelationCoefficientFromScratch(normalizedCorpusName)
    print(f'Cosine Correlation Coefficient for TF: {cosCoefTF:.3f}\nCosine Correlation Coefficient for TFIDF: {cosCoefTFIDF:.3f}\nJaccard Correlation Coefficient for TF: {jacCoefTF:.3f}\nJaccard Correlation Coefficient for TFIDF:{jacCoefTFIDF:.3f}')
    

