import sys
import glob
import os
import argparse
import nltk
import scipy.sparse
import numpy as np
import token_count

'''CATEGORIES PLANNED
Folktales, World War I, Witchcraft
Current Count of documents
---------
FOLKTALES
---------
Book of Myths: 34
Russian Folktales: 51
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
    
    args = parser.parse_args()
    
    corpus = {}
    # all_files = glob.glob('/Users/Naga/Desktop/Python/Data/*/y.txt')
    file_path = args.foldername + "*.txt"
    print(file_path)
    all_files = glob.glob(file_path)
    # print(all_folders)
    # for folder in all_folders:
    #     all_files.append(glob.glob(args.foldername + folder + "/*.txt"))
    
    print(all_files)
    for file in all_files:
        # print(file)
        # all_files.append(file)
        corpus[file] = {}
    
    print(corpus)
    
if __name__ == "__main__":
    main()