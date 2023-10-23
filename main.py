import sys
import glob
import os
import argparse
import nltk
import scipy.sparse
import numpy as np
import token_count
import shutil
import json
from collections import defaultdict
import gensim.downloader as api
import gensim
from gensim import corpora
import math
from gensim.models import LdaModel

'''CATEGORIES PLANNED
Folktales, World War I, Witchcraft
Current Count of documents
---------
FOLKTALES
---------
Book of Myths: 34
Russian Folktales: 6
Irish Folktales: 89
Total: 174
---------
WITCHCRAFT
---------
Witch-Cult: 41
Letters: 11
Superstitions: 11
Amber: 14
History of Witchcraft in England: 15
Witchcraft Delusion Connecticut: 11
Total: 103
---------
World War I
---------
'''

def process_text(args, filename):
    command_line = "python3 token_count.py " + filename
    if args.stem:
        command_line += " -s"
    if args.lower:
        command_line += " -l"
    if args.word:
        command_line += " -w"
    if args.number:
        command_line += " -n"
    new_name = filename.split("/")[-1]
    new_name = "processed_txt/" + new_name[:-3] + "json"
    # print(new_name)
    command_line += " > " + new_name
    # print(command_line)
    os.system(command_line)
    
    return new_name

def create_matrix(corpus, vocab:set, word_index:dict):
    category_matrix = np.empty([1,3], dtype = int)
    category_matrix[0][0] = 0
    category_matrix[0][1] = 1
    category_matrix[0][2] = 2
    word_doc_matrix = np.empty([2, len(corpus) + 1], dtype = int)
    num_w = 0
    num_f = 0
    count = 0
    size_vocab = 0
    for doc in corpus:
        count += 1
        corpus[doc]["id"] = count
        word_doc_matrix[0][count] = count
        category_index = 1
        if doc.split("/")[1][0] == "w":
            category_index = 1
            num_w += 1
        elif doc.split("/")[1][0] == "f":
            category_index = 2
            num_f += 1
        with open(corpus[doc]["location"], 'r') as file:
            data = json.load(file)
            for i in data:
                if i not in vocab:
                    vocab.add(i)
                    size_vocab += 1
                    word_index[i] = size_vocab
                    if size_vocab == 1:
                        word_doc_matrix[size_vocab][0] = size_vocab
                        for j in range(1, len(corpus) + 1):
                            if j == count:
                                word_doc_matrix[size_vocab][j] = data[i]
                            else:
                                word_doc_matrix[size_vocab][j] = 0
                    else:
                        # category_matrix = np.append(category_matrix, i)
                        temp_doc = np.empty(len(corpus) + 1, dtype = int)
                        temp_doc[0] = size_vocab
                        for j in range(1, len(corpus) + 1):
                            if j == count:
                                temp_doc[j] = data[i]
                            else:
                                temp_doc[j] = 0
                        # print(temp_doc)
                        # print(temp_doc)
                        word_doc_matrix = np.vstack([word_doc_matrix, temp_doc])
                    temp = np.empty(3, dtype = int)
                    temp[0] = size_vocab
                    
                    if category_index == 1:
                        temp[1] = data[i]
                        temp[2] = 0
                        category_matrix = np.vstack([category_matrix, temp])
                    elif category_index == 2:
                        temp[1] = 0
                        temp[2] = data[i]
                        category_matrix = np.vstack([category_matrix, temp])
                    
                else:
                    location = word_index[i]
                    category_matrix[location][category_index] += data[i]
                    
                    try:
                        word_doc_matrix[location][count] += data[i]
                    except:
                        exit("At" + str(location) + str(count) + "is" + str(word_doc_matrix[location][count]))
    print("Number of witchcraft docs:", num_w)
    print("Number of folklore docs:", num_f)
    print("Avg # of tokens per witchcraft doc:", np.sum(category_matrix[:, 1]) / num_w)
    print("Avg # of tokens per folklore doc:", np.sum(category_matrix[:, 2]) / num_f)
    
    sparse_categories = scipy.sparse.csr_matrix(category_matrix)
    word_matrix = np.delete(word_doc_matrix, (0), axis=1)
    # sub_term_f = np.array(word_matrix)
    sub_term_f = word_matrix[:, 0:num_f]
    # sub_term_w = np.array(word_matrix)
    sub_term_w = word_matrix[:, num_f:num_f+num_w]
    # sub_term_f = np.delete(sub_term_f, [])
    sparse_word = scipy.sparse.csr_matrix(word_matrix)
    gensim_corpus = gensim.matutils.Sparse2Corpus(sparse_word, documents_columns=False)
    sparse_word.eliminate_zeros()
    sparse_categories.eliminate_zeros()
    
    
    class_prob = bayes(category_matrix, word_doc_matrix, num_w, num_f, word_index)
    # print(word_index)
    indexer = {}
    for k,v in word_index.items():
        indexer[v] = k
    # dict((v,k) for k,v in word_index)
    
    return lda(gensim_corpus, indexer, 10, sub_term_f, sub_term_w)
                

def bayes(categories, term_matrix, num_w, num_f, word_index):
    ''' 
    log-likelihood-ratio(word, context) = log(P(w|c)) - log(P(w|C_0))
    '''
    # Find Probability Matrix
    # Prob is log((num_w_in_c + 1)/ (num_tokens_in_c + vocab_size))
    class_prob = np.empty(3, dtype= float)
    prob_w = num_w / (num_w + num_f)
    prob_f = num_f / (num_w + num_f)
    count = 0
    current_word = 1
    vocab_size = len(word_index)
    for category in categories:
        word_id = category[0]
        w_in_c1 = category[1]
        w_in_c2 = category[2]
        token_in_c1 = np.sum(categories[:, 1])
        token_in_c2 = np.sum(categories[:, 2])
        prob_c1 = math.log((w_in_c1 + 1) / (token_in_c1 + vocab_size))
        prob_c2 = math.log((w_in_c2 + 1) / (token_in_c2 + vocab_size))
        
        # find log(P(w|c_0))
        class_prob = np.vstack([class_prob, [word_id, prob_c1 - prob_c2, prob_c2 - prob_c1]])
    # print(class_prob)
    class_prob = np.delete(class_prob, (0), axis=0)   
    # print(class_prob)
    col_f = class_prob[:, 2]
    col_w = class_prob[:, 1]
    top_10_f = -np.sort(-col_f)  
    top_10_w = -np.sort(-col_w)
    words_f = []
    word_w = []
    indexer = {}
    for k,v in word_index.items():
        indexer[v] = k
    for word in top_10_f[:10]:
        prob_id = class_prob[np.where(class_prob[:, 2] == word)]
        # print(prob_id)
        prob_id = prob_id[0]
        # print(prob_id)
        # print(prob_id[0])
        words_f.append(indexer[prob_id[0]])
    for word in top_10_w[:10]:
        prob_id = class_prob[np.where(class_prob[:, 1] == word)]
        # print(prob_id)
        prob_id = prob_id[0]
        # print(prob_id)
        # print(prob_id[0])
        word_w.append(indexer[prob_id[0]])
    
    final_prob = np.delete(class_prob, (0), axis=1)
    print("Most common witchcraft words:", word_w)
    print("Most common folklore words:", words_f)        
    
    # Calculate log-likelihood-ratio and give top 10 words per class
    
    # Need corpus, id2word
    
    return class_prob

def lda(term_matrix, idword, topics, sub_term_f, sub_term_w):
    term_f = scipy.sparse.csr_matrix(sub_term_f)
    term_w = scipy.sparse.csr_matrix(sub_term_w)
    term_f = gensim.matutils.Sparse2Corpus(term_f, documents_columns=False)
    term_w = gensim.matutils.Sparse2Corpus(term_w, documents_columns=False)
    # print(term_f)
    # print(term_w)
    
    model = LdaModel(term_matrix, num_topics=topics, id2word=idword)
    
    topics_f = model.get_document_topics(term_f, per_word_topics=False)
    topics_w = model.get_document_topics(term_w, per_word_topics=False)
    # model_w = LdaModel(term_w, num_topics=topics, id2word=idword)
    # model_f = LdaModel(term_f, num_topics=topics, id2word=idword)
    
    # print(model_w.print_topics(num_topics = 5, num_words = 5))
    # print(model_f.print_topics(num_topics = 5, num_words = 5))
    # .print_topics(num_topics = 5, num_words = 5)
    # print(model.inference(term_f))
    # print(model.inference(term_w))
    # print(topics_f.print_topics(num_topics = 5, num_word = 3))
    # print(topics_w.print_topics(num_topics = 5, num_word = 3))
    
    
    # new_topics_f = model[term_f]
    # new_topics_w = model[term_w]
    
    # print(new_topics_f)
    # print(new_topics_w)
    print("Top 3 topics in folklore:", model.top_topics(term_f, topn=3))
    print("Top 3 topics in witchcraft:", model.top_topics(term_w, topn=3))
    # model_f = LdaModel(term_f, num_topics=topics, id2word=idword)
    # model_w = LdaModel(term_w, num_topics=topics, id2word=idword)
    
    
    
    # for topic in new_topics_f:
    #     print(topic)
        
    # for topic in new_topics_w:
    #     print(topic)
    
    # gensim.interface..save_corpus("lda_folklore.csv", corpus, id2word=None, metadata=False)
    
    '''
    [(0, '0.001*"immobile" + 0.001*"abbot" + 0.001*"discussions" + 0.000*"yet" + 0.000*"ways"'), 
    (5, '0.008*"ways" + 0.005*"conversion" + 0.004*"years" + 0.003*"sunday" + 0.002*"hand"'), 
    (3, '0.002*"son" + 0.001*"abbot" + 0.001*"god" + 0.001*"action" + 0.001*"new"'), 
    (4, '0.130*"cuffed" + 0.009*"deserter" + 0.006*"discouraged" + 0.006*"discussed" + 0.004*"disperse"'), 
    (6, '0.018*"beds" + 0.017*"anathema" + 0.016*"carried" + 0.014*"cessation" + 0.013*"beyond"')]
    '''
    
    '''
    Top 4 topics in folklore: 
    [([(0.09277924, 'latters'), (0.08714014, 'leave'), (0.07821521, 'languid')], 0.0), 
    ([(0.34385446, 'master'), (0.13414359, 'marched'), (0.020273622, 'may')], 0.0), 
    ([(0.06658595, 'morning'), (0.052259546, 'immobile'), (0.047169186, 'laid')], 0.0), 
    ([(0.05779511, 'gone'), (0.056666892, 'druid'), (0.051265836, 'gospel')], 0.0),
    
     
    ([(0.10379037, 'give'), (0.09178663, 'disperse'), (0.04669322, 'discussions')], 0.0), 
    ([(0.009826971, 'beds'), (0.0060068895, 'abbot'), (0.0031643626, 'action')], -0.3155921803863891), 
    ([(0.0015392016, 'action'), (0.0011931945, 'abbot'), (0.00075496687, 'abide')], -0.8205901326376152), 
    ([(0.091258146, 'abbot'), (0.05108207, 'action'), (0.03943526, 'admitted')], -0.936429046302898), 
    ([(0.015332338, 'window'), (0.008779066, 'wise'), (0.007741367, 'ways')], -1.0392680060814878), 
    ([(0.022478137, 'abide'), (0.016562646, 'anathema'), (0.016421227, 'carried')], -16.632451021409654)]
    
    Top 4 topics in witchcraft: 
    [([(0.091258146, 'abbot'), (0.05108207, 'action'), (0.03943526, 'admitted')], 0.0), 
    ([(0.0015392016, 'action'), (0.0011931945, 'abbot'), (0.00075496687, 'abide')], 0.0), 
    ([(0.09277924, 'latters'), (0.08714014, 'leave'), (0.07821521, 'languid')], 0.0), 
    ([(0.34385446, 'master'), (0.13414359, 'marched'), (0.020273622, 'may')], 0.0),
    
     
    ([(0.06658595, 'morning'), (0.052259546, 'immobile'), (0.047169186, 'laid')], 0.0), 
    ([(0.05779511, 'gone'), (0.056666892, 'druid'), (0.051265836, 'gospel')], 0.0), 
    ([(0.009826971, 'beds'), (0.0060068895, 'abbot'), (0.0031643626, 'action')], 0.0), 
    ([(0.022478137, 'abide'), (0.016562646, 'anathema'), (0.016421227, 'carried')], 0.0), 
    ([(0.015332338, 'window'), (0.008779066, 'wise'), (0.007741367, 'ways')], 0.0), 
    ([(0.10379037, 'give'), (0.09178663, 'disperse'), (0.04669322, 'discussions')], 0.0)]
    '''
    return model

def main():
    # if len(sys.argv) < 3:
    #     exit("Please input text file and at least one preprocessing command.")
        
    parser = argparse.ArgumentParser(
        prog='normalize_text',
        description='This program will take in a txt file and list of desired preproccessing commands and output tokens.',
        epilog='Please give at least one of the following preprocessing procedures: lowercasing (--lower or -l), stemming (--stem or -s), stopword removal (--word or -w), or number and symbol removal (--number)'
    )
    parser.add_argument('foldername')
    parser.add_argument('-l', '--lower', action='store_true')
    parser.add_argument('-s', '--stem', action='store_true')
    parser.add_argument('-w', '--word', action='store_true')
    parser.add_argument('-n', '--number', action='store_true')
    parser.add_argument('-p', '--preprocess', action='store_true')
    
    args = parser.parse_args()
    
    corpus = {}
    # all_files = glob.glob('/Users/Naga/Desktop/Python/Data/*/y.txt')
    file_path = args.foldername + "*.txt"
    # print(file_path)
    all_files = glob.glob(file_path)
    # print(all_folders)
    # for folder in all_folders:
    #     all_files.append(glob.glob(args.foldername + folder + "/*.txt"))
    
    # print(all_files)
    for file in all_files:
        # print(file)
        # all_files.append(file)
        corpus[file] = {}
    
    if args.preprocess:
        dir = 'processed_txt'
        if os.path.exists(dir):
            shutil.rmtree(dir)
        os.makedirs(dir)
    
        # print(corpus)
        for doc in corpus:
            corpus[doc]["location"] = process_text(args, doc)
    else:
        for doc in corpus:
            corpus[doc]["location"] = "processed_txt/" + doc.split("/")[-1][:-3] + "json"
            
    
    
    # print(corpus)
    
    vocab = set()
    word_index = {}
    print(create_matrix(corpus, vocab, word_index))
    
if __name__ == "__main__":
    main()