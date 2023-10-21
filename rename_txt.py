import sys
import glob
import os
import argparse
import nltk
import scipy.sparse
import numpy as np
import token_count
import shutil
import re

def main():
    # # if len(sys.argv) < 3:
    # #     exit("Please input text file and at least one preprocessing command.")
        
    # parser = argparse.ArgumentParser(
    #     prog='normalize_text',
    #     description='This program will take in a txt file and list of desired preproccessing commands and output tokens.',
    #     epilog='Please give at least one of the following preprocessing procedures: lowercasing (--lower or -l), stemming (--stem or -s), stopword removal (--word or -w), or number and symbol removal (--number)'
    # )
    # parser.add_argument('filename')
    # parser.add_argument('keyword')
    # parser.add_argument('-n', '--newline', action='store_true')
    
    # args = parser.parse_args()
    
    file_path = "texts/*.txt"
    all_files = glob.glob(file_path)
    
    for file in all_files:
        # print(file)
        file_root = file.split('/')[0]
        file_text = file.split('/')[1]
        if file_text[:5] == "irish":
            print("folk")
            os.rename(file, file_root + "/f_" + file_text)
        elif file_text[:5] == "amber" or file_text[:5] == "witch":
            print("witch")
            os.rename(file, file_root + "/w_" + file_text)
        else:
            print("None")
    
    
if __name__ == "__main__":
    main()