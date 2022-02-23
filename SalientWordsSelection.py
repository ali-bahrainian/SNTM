'''
Created on Feb 22, 2022

@author: ali
'''
from __future__ import division
from nltk.corpus import stopwords
import nltk
from nltk.collocations import BigramCollocationFinder
from nltk.probability import FreqDist
from collections import Counter
import copy
from math import log
from sklearn.datasets import fetch_20newsgroups
import pickle
import re



class PrepareVocab:
    def __init__(self, filteredBigramPMIs, wordsPMITable, vocabList, word2count, docs, word2docID):
        self.vocab = vocabList
        self.filteredBigramPMIs = filteredBigramPMIs
        self.wordsPMITable =  wordsPMITable
        self.word2docID = word2docID
        self.docs = docs
        
 
    def returnVocabDist(self, docs, topN):
        allWords = []
        for doc in docs:
            allWords.extend(doc.split(" "))
        fdist = FreqDist(allWords)
        common = fdist.most_common(topN)
        common = [item[0] for item in common]
        return common
    
    
    def _returnWord2Index (self):
        word2index = {}
        for i in range(len(self.vocab)):
            word2index[self.vocab[i]] = i
        return word2index
    



class PrepareNPMITable:
    def __init__(self, docs):
        #vocab = self.returnVocabDist(docs)
        self.StopWords = self.readWordsList("stopwords_topicm1.txt")
        self.docs = self.preprocessing(docs)
        
        
        
    def readWordsList(self, dicFile):
        negatorList = set(line.strip() for line in open(dicFile))
        return negatorList
    
    """def returnVocabDist(self, docs):
        allWords = []
        for doc in docs:
            allWords.extend(doc.split(" "))
        fdist = FreqDist(allWords)
        common = fdist.most_common()
        print(len(common))
        return"""
    
    def preprocessing(self, docs):
        stopWords = set(stopwords.words('english')).union(self.StopWords)
        normalizedDocs = []
        for s in docs:
            s = re.sub("\n","",s)
            s = s.split(" ")
            filteredWords = []
            for w in s:
                alphabet = re.search(r"^[a-zA-Z][a-zA-Z0-9]*$", w)
                if alphabet is None or len(w)<4 or w in stopWords:
                    continue
                else:
                    filteredWords.append(w)
            normalizedDocs.append(" ".join(filteredWords))

        
        return normalizedDocs
        
        
    def limitDocsToVocab(self, docs, vocab):
        reducedDocs = []
        for s in docs:
            doc = []
            s = s.split(" ")[0]
            for w in s:
                if w in vocab:
                    doc.append(w)
            reducedDocs.append(" ".join(doc))
        return reducedDocs
    
    
    def npmi_scorer(self, worda_count, wordb_count, bigram_count, min_count, corpus_word_count):
        if bigram_count >= min_count:
            pa = worda_count / corpus_word_count
            pb = wordb_count / corpus_word_count
            pab = bigram_count / corpus_word_count
            return log(pab / (pa * pb)) / -log(pab)
        else:
            # Return -infinity to make sure that no phrases will be created
            # from bigrams less frequent than min_count
            return float('-inf')
               
    
    
    def returnWord2DocID(self):#THis function returns a dictionary where each key is a word from vocab and each value is a list of DocIDs that word is present in.
        word2doc_id = {}
        tokenizedDocs = [doc.split(" ") for doc in self.docs]
        all_tokens = [item for sublist in tokenizedDocs for item in sublist]
        vocab = list(set(all_tokens))
        
        for w in vocab:
            if w not in word2doc_id:
                word2doc_id[w] = []
            for i in range(len(self.docs)):
                if w in self.docs[i]:
                    word2doc_id[w].append(i)
        
        return word2doc_id
        
        
    
    def returnPMIScores(self, vocabSize = 5000, readFromDisk = False):
        
        if not readFromDisk:
        
            tokenizedDocs = [doc.split(" ") for doc in self.docs]
            
            all_tokens = [item for sublist in tokenizedDocs for item in sublist]
            token2count = Counter(all_tokens)
            totalTokenCount = sum(token2count.values())
            #print totalTokenCount
            
            finder = BigramCollocationFinder.from_documents(tokenizedDocs)
            finder.apply_freq_filter(3)
            bigram_measures = nltk.collocations.BigramAssocMeasures()
            bigramPMIs =  (finder.score_ngrams(bigram_measures.pmi))

            unsuccessfulNPMICases = 0
            NPMIs = []
            for item in bigramPMIs:
                if item[1]>0:
                    try:
                        NPMIs.append((item[0], self.npmi_scorer(float(token2count[item[0][0]]), float(token2count[item[0][1]]), float(finder.ngram_fd[(item[0][0], item[0][1])]), 0.0, float(totalTokenCount))))
                    except:
                        unsuccessfulNPMICases+=1
 
            sortedVocab = sorted(token2count, key=token2count.get, reverse = True)
            vocab = sortedVocab[:vocabSize]
            
            """Pickle and save outputs"""
            
            output_NPMIs = open('/Users/ali/Coding/newTopicModel/NPMIs.pkl', 'wb')
            pickle.dump(NPMIs, output_NPMIs,protocol=pickle.HIGHEST_PROTOCOL)
            
            output_token2count = open('/Users/ali/Coding/newTopicModel/token2count.pkl', 'wb')
            pickle.dump(token2count, output_token2count,protocol=pickle.HIGHEST_PROTOCOL)
            
            
            output_vocab = open('/Users/ali/Coding/newTopicModel/vocab.pkl', 'wb')
            pickle.dump(vocab, output_vocab,protocol=pickle.HIGHEST_PROTOCOL)
            
            
        else:
            NPMIs =  pickle.load(open('/Users/ali/Coding/newTopicModel/NPMIs.pkl', 'rb'))
            token2count = pickle.load(open('/Users/ali/Coding/newTopicModel/token2count.pkl', 'rb'))
            vocab = pickle.load(open('/Users/ali/Coding/newTopicModel/vocab.pkl', 'rb'))
        return NPMIs, token2count, vocab#bigramPMIs


    def costructWordGraphs(self, bigramPMIs):
        words = []
        for i in xrange (len(bigramPMIs)):
            words.append(bigramPMIs[i][0][0])
            words.append(bigramPMIs[i][0][1])
        
        cnt = Counter(words)
        return cnt
    
      

    def createWordPairPMITable(self, vocab, PMIstorage, cnt_words, readFromDisk = False):
        if not readFromDisk:
            table = {}
            temp = {}
            for w in vocab:
                table[w] = 0
                temp[w] = 0
                
                
            for w in vocab:
                table[w] = copy.deepcopy(temp)
            
            for item in PMIstorage:
                if cnt_words[item [0][0]] == 1 and cnt_words[item [0][1]] == 1:
                    pass
                else:
                    table[item[0][0]][item[0][1]] = item[1]
                    table[item[0][1]][item[0][0]] = item [1]
                    
            output_table = open('/Users/ali/Coding/newTopicModel/table.pkl', 'wb')
            pickle.dump(table, output_table,protocol=pickle.HIGHEST_PROTOCOL)
        else:

            table =  pickle.load(open('/Users/ali/Coding/newTopicModel/table.pkl', 'rb'))
        return table

def readNewsGroupsDataset (cats = ['alt.atheism','comp.graphics','comp.os.ms-windows.misc','comp.sys.ibm.pc.hardware','comp.sys.mac.hardware','comp.windows.x','misc.forsale','rec.autos', 'rec.motorcycles', 'rec.sport.baseball', 'rec.sport.hockey','sci.crypt','sci.electronics','sci.med','sci.space','soc.religion.christian','talk.politics.guns','talk.politics.mideast','talk.politics.misc', 'talk.religion.misc']):
            
    newsgroups_train = fetch_20newsgroups(subset='train', remove=('headers', 'footers', 'quotes'), categories=cats)
    trainData = newsgroups_train.data
    trainLabels =  newsgroups_train.target
        
    newsgroups_test = fetch_20newsgroups(subset='test',remove=('headers', 'footers', 'quotes'), categories=cats)
    testData = newsgroups_test.data
    testLabels = newsgroups_test.target
    
    trainData = [str(doc.encode('utf-8').lower()) for doc in trainData]
    testData = [str(doc.encode('utf-8').lower()) for doc in testData]

    return trainData, trainLabels, testData, testLabels
    



def returnNPMIEmbeddingOfWordsList (wordsList, wordsPMITable):
    NPMI_Embeddings = []
    for i in range(len(wordsList)):
        try:
            NPMI_Embeddings.append(returnWordPairPMI(wordsList[i],[x for ind,x in enumerate(wordsList) if ind!=i], wordsPMITable))  
        except:
            pass
    return NPMI_Embeddings

def returnWordPairPMI(word,cellWordsList, wordsPMITable):
    PMIs = []
    for w in cellWordsList:
        PMIs.append(wordsPMITable [word][w])
        
    return sum(PMIs)/len(PMIs)

def returnWord2NumOfDocuments (documents):
    word_frequencies = [Counter(document.split()) for document in documents]

    #calculate document frequency
    document_frequencies = Counter()
    map(document_frequencies.update, (word_frequency.keys() for word_frequency in word_frequencies))

    return document_frequencies


if __name__ == '__main__':
    
    docs,_,_,_ = readNewsGroupsDataset()
    print("Running PrepareNPMITable")
    npm = PrepareNPMITable(docs)
    
    print("returnPMIScores")
    bigramPMIs, word2count, vocab = npm.returnPMIScores(vocabSize = 10000, readFromDisk = True)
 
    
    filteredBigramPMIs = []
    for item in bigramPMIs:
        if item[1]>0:
            filteredBigramPMIs.append(item)
    
    filteredBigramPMIs = sorted(filteredBigramPMIs, key=lambda x: x[1])
    
    
        
    

    cnt_words = npm.costructWordGraphs(bigramPMIs)
    

    print("creating PMI Table")
    wordsPMITable = npm.createWordPairPMITable(cnt_words.keys(), filteredBigramPMIs, cnt_words, readFromDisk = True)
    
    
    
    doc = "encryption secure system space"
    print(returnNPMIEmbeddingOfWordsList(doc.split(" "), wordsPMITable))
    
    
    
    word2docID = npm.returnWord2DocID()
    #docs = ntm.limitDocsToVocab(ntm.docs,cnt_words.keys())
    print('Initializing with prepVocab')
    prepVocab = PrepareVocab(filteredBigramPMIs, wordsPMITable, cnt_words.keys(), cnt_words, npm.docs, word2docID)
    numTopics = 20
    word2NumOfDocs = returnWord2NumOfDocuments(npm.docs)
    word2index = prepVocab._returnWord2Index()
    vocab = word2index.keys()
    
    
    """The following lines produce npmi embeddings for the entire vocabulary (Equation 6)"""
    vocab_npmiEmbedds = returnNPMIEmbeddingOfWordsList(vocab, wordsPMITable)
    print(vocab)
    wordsNPMIs = []
    for i in range(len(vocab)):
        wordsNPMIs.append((vocab[i], vocab_npmiEmbedds[i]))
    wordsNPMIs = sorted(wordsNPMIs, key=lambda x: x[1], reverse=True)
    print(len(wordsNPMIs))
    print(wordsNPMIs)    
    
    
    

    