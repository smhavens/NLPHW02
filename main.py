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
    sparse_categories = scipy.sparse.csr_matrix(category_matrix)
    sparse_word = scipy.sparse.csr_matrix(word_doc_matrix)
    sparse_word.eliminate_zeros()
    sparse_categories.eliminate_zeros()
    word_matrix = np.delete(word_doc_matrix, (0), axis=1)
    
    class_prob = bayes(category_matrix, word_doc_matrix, num_w, num_f, word_index)
    
    return lda(word_matrix, class_prob, 10)
                

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
    print(class_prob)
    top_10_f = np.sort(class_prob[:, 2])  
    top_10_w = np.sort(class_prob[:, 1])
    
    final_prob = np.delete(class_prob, (0), axis=1)
    print(top_10_f)
    print(top_10_w)        
    
    # Calculate log-likelihood-ratio and give top 10 words per class
    
    # Need corpus, id2word
    
    return class_prob

def lda(term_matrix, idword, topics):
    return LdaModel(corpus = term_matrix, num_topics = topics)

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