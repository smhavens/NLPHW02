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

I thought these categories were interesting due to them both being tied to religion and cultural history, especially at potential similarities between the two categories.

### Methodology
For preprocessing, I made use of all options (lowercasing, stemming, stop word removal, and removal of numbers and symbol) because it gave the most interesting results for the most common words. With lowercasing and stemming, it allowed the counts to show off truly common words and without stopwords to dilute the results. The use of number and sybmol removal was mostly to account for document formatting, which often included symbols or numbers that would be difficult to remove only partially, so I decided for best results was to sacrifice all numbers and symbols to focus entirely on words.
For topic modeling, I made use of sparse matrixes and the gensim library to do LDA modeling of the entire document-term matrix and then by categories.

### Results and Analysis
#### Top 10 words
| Category |  |  | | | | | | | | |
| ----- | ----- | ----- | ----- | ----- | ----- | ----- | ----- | ----- | ----- | ----- |
| Witchcraft| 'elizabeth'| 'ane'| 'à' | 'james' | 'goodwife' | 'witchcraft' | 'john' | 'england' | 'item' | 'que' |
| Folklore | 'fionn'| 'ivan'| 'mongan'| 'baba'| 'mac'| 'russian'| 'yaga'| 'moujik'| 'deirdrê'| 'roland' |

A table for the top 10 most common words for each category. This was done by using my own "bayes" function, where I took a category-term matrix and processed their naive bayes value term by term until returning the sorted top 10 scores for both Witchcraft and Folklore.

These were most interesting because of just how common names were used, with the exception of 'goodwife', 'witchcraft', 'england', 'item', 'que', and 'russian'. The witchcraft words can hint at the time period they were most well known for because of the names Elizabeth, James, and John and the focus on women with the term 'goodwife'. Folklore can also show which cultures were used in the documents, as 'fionn', 'mongan', 'mac', 'deirdre', and 'roland' all point towards Irish folklore while 'yaga', 'baba', 'ivan', and 'russian' obviously allude to Russian folklore. It can help point out bias in source texts while also illustrating how pivotal certain figures were in those source texts (baba yaga, for example).

#### Top 4 topics in Witchcraft
| Topic ID |  |  | |
| ----- | ----- | ----- | ----- |
| 0 | abbot | action | admitted |
| 1 | action | abbot | abide |
| 2 | latters | leave | languid |
| 3 | master | marched | may |

The table above and below show the top 4 topics for their given category. I used the function
```
model.top_topics(term_f, topn=3)
```
To generate topics of length 3, using an LDA model built off the entire set of documents but then focusing on the subset of folklore or witchcraft. As can be seen, there is overlap, but it seems to most likely be from the model itself rather than entirely accurate to the documents. Regardless, we can see some of the major common connections: abbot and admitted seem heavily related to the persecution of witches while we have 'druid' and 'gospel' relating to religion heavy terms in folklore.

#### Top 4 topics in Folklore
| Topic ID |  |  | |
| ----- | ----- | ----- | ----- |
| 0 | latters | leave | languid |
| 1 | may | marched | may |
| 2 | morning | immobile | laid |
| 3 | gone | druid | gospel |

### Discussion

#### What I've Learned From My Dataset
I learned from my dataset the importance of names to contextualize documents and categories. Historical figures or legendary characters from folklore all were major words that regularly appeared in the datasets for their category. This method also can show the discrepency of the quantity of documents pertaining to certain sub-categories, as Irish and Russian folklore appear to have much greater weight compared to the general myths subcategory.

#### What Lessons I Learned Completing this Assignment 
I learned the importance of reading documentation and examples, as when using gensim I struggled to learn how to use the lda model and its functions due to a severe lack of examples. I also learned about the difficulty of processing a large quantity of documents, as both finding relevant documents and then splitting them up into useable information took a surprising amount of time.
