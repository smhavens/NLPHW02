# NLPHW02

## How to Run
Required Files (main.py, token_count.py)
Required Downloads: gensim, scipy
```
pip install --upgrade gensim
python -m pip install scipy
```

This program is made to work with two categories (witchcraft and folklore) with the included text folder.

To run, download main.py, token_count.py, and the text/ folder. Then, run the command
```
python3 main.py "texts/"
```
with any of the given extentions:
1. (-l, --lower) to have the preprocessing lowercase everything
2. (-s, --stem) to have the preprocessing stem everything
3. (-w, --word) to have the preprocessing remove stopwords
4. (-n, --number) to have preprocessing remove all numbers and symbols
5. (-p, --preprocess) to have the program run preprocessing (must do if there is no processed_txt folder downloaded)

The program will use a folder named processed_txt filled with json tokenizations of all documents in texts/ and then it will print out
1. The top 10 most common witchcraft and folklore words based on naive bayes calculations using the count method with smoothing
2. The LdaModel information, including number of terms, number of topics, decay, and chunksize

## Report
### Dataset
For my dataset I collected multiple documents from Project Gutenberg in the witchcraft and folklore categories and then used a mix of manual and automated (see split_chapter.py and rename_txt.py) to split the documents into their subsections.

| Category | Number of Documents | Avg # of Tokens per Document |
| ----- | ----- | ----- |
| Witchcraft| 106 | 2245.99 |
| Folklore | 129 | 1000.22 |

### Methodology

### Results and Analysis

### Discussion
