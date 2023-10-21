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

def create_matrix(corpus, vocab:set):
    category_matrix = np.empty([1,3], dtype = object)
    category_matrix[0][0] = "word"
    category_matrix[0][1] = "witch"
    category_matrix[0][2] = "folk"
    # category_matrix = np.array([["word", 1, 2], []])
    word_doc_matrix = np.empty(len(corpus), dtype = object)
    # word_doc_matrix = np.array(["word", ])
    print(word_doc_matrix)
    print(category_matrix)
    count = 0
    for doc in corpus:
        # np.append(word_doc_matrix, [])
        # print(word_doc_matrix)
        word_doc_matrix = np.append(word_doc_matrix[0], doc)
        category_index = 1
        if doc.split("/")[1][0] == "w":
            category_index = 1
        elif doc.split("/")[1][0] == "f":
            category_index = 2
        with open(corpus[doc]["location"], 'r') as file:
            data = json.load(file)
            for i in data:
                if i not in vocab:
                    vocab.add(i)
                    # category_matrix = np.append(category_matrix, i)
                    temp = np.empty(3, dtype = object)
                    temp[0] = i
                    if category_index == 1:
                        temp[1] = data[i]
                        temp[2] = 0
                        category_matrix = np.vstack([category_matrix, temp])
                    elif category_index == 2:
                        temp[1] = 0
                        temp[2] = data[i]
                        category_matrix = np.vstack([category_matrix, temp])
                else:
                    location = 0
                    # print(category_matrix[:,0])
                    if i == "word":
                        # location = np.char.find(category_matrix[:, 0], i, start = 1)
                        location = np.where(category_matrix[:,0] == i)
                    else:
                        # location = np.char.find(category_matrix[:, 0], i)
                        location = np.where(category_matrix[:,0] == i)
                    # print(category_matrix)
                        location = location[0][0]
                    # print("location:", location)
                    # print(category_index)
                    # print(category_matrix[location])
                    
                    # if location == 16:
                    #     print("RAW:", category_matrix[16])
                    #     print("BASE:", category_matrix[location])
                    #     print(category_matrix[location][category_index])
                    #     print(type(category_matrix[location][category_index]))
                    #     print(type(data[i]))
                        category_matrix[location][category_index] += data[i]
                    # if category_index == 1:
                        
                    # elif category_index == 2:
                    #     category_matrix = np.vstack([category_matrix, [i, 0, data[i]]])
    
    print(category_matrix)
    print(word_doc_matrix)
                

def bayes(categories, term_matrix):
    return categories

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
    create_matrix(corpus, vocab)
    
if __name__ == "__main__":
    main()