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
    # if len(sys.argv) < 3:
    #     exit("Please input text file and at least one preprocessing command.")
        
    parser = argparse.ArgumentParser(
        prog='normalize_text',
        description='This program will take in a txt file and list of desired preproccessing commands and output tokens.',
        epilog='Please give at least one of the following preprocessing procedures: lowercasing (--lower or -l), stemming (--stem or -s), stopword removal (--word or -w), or number and symbol removal (--number)'
    )
    parser.add_argument('filename')
    parser.add_argument('keyword')
    parser.add_argument('-n', '--newline', action='store_true')
    
    args = parser.parse_args()
    
    with open(args.filename, 'r') as file:
        contents = file.read()
        count = 0
        if not args.newline:
            for entry in contents.split(args.keyword)[1:]:
                # print(count)
                temp_name = "texts/" + args.filename.split("/")[-1].split('.')[0] + str(count) + ".txt"
                with open(temp_name, 'w') as w:
                    w.write(entry)
                count += 1
        else:
            print("newlines")
            for entry in re.split(r'\n\n\n\n\n', contents)[1:]:
                temp_name = "texts/" + args.filename.split("/")[-1].split('.')[0] + str(count) + ".txt"
                with open(temp_name, 'w') as w:
                    w.write(entry)
                count += 1
        # do something with entry 
    
    
if __name__ == "__main__":
    main()