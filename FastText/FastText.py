

import pandas as pd
import numpy as np
import string
import pickle
import itertools
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from collections import Counter
from gensim.models.fasttext import FastText
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
import logging
import time
from utils import *


logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

def most_similar_words(model, words, nwords = 30):
    
    word_collection = {}
    list_words = []
    
    for word in words:    
        word_collection[word] = model.wv.most_similar(word, topn = nwords)
        list_words.append(word)
        for wrd in word_collection[word]:
            list_words.append(wrd[0])
    
    return word_collection , list_words


def associated_comments(tokenized_corpus, word_dicts):
    
    comment_index = []
    for key, words in word_dicts.items():
        for word in words: 
            index = [idx for idx,comment in enumerate(tokenized_corpus) 
                          if key in comment if word[0] in comment]
            comment_index.append([key,word[0],index])
    
    return comment_index
            

def grouping_words(comment_indexes, survey, nwords = 20):
    
        
    df = []
    for comment_index in comment_indexes:
        keyword, similar_word, indexes = comment_index
        for idx in indexes:
            area, role, gender, comment = survey.loc[idx]
            df.append([idx,keyword, similar_word, area, role, gender, comment])

    df = pd.DataFrame(df)
    grouping = df.groupby([0,1,3,4,5,6])[2].apply(','.join)
    grouping = grouping.str.split(",")
    grouping = grouping.reset_index()
    grouping["length"] = [len(x) for x in grouping[2]]
    grouping = grouping.sort_values(by = "length", ascending = False)
    
    return grouping, df


def doc2vec_train(tokens, **kwargs):
    
    # Doc2Vec
    tagged_corpus = [TaggedDocument(tokenized_doc, [idx]) 
                     for idx, tokenized_doc in enumerate(tokens)]
        
    doc2vec_model = Doc2Vec(vector_size=100, 
                            min_count=5, 
                            epochs=200, 
                            hs = 0, 
                            negative = 25, 
                            window = 40)
    
    doc2vec_model.build_vocab(tagged_corpus)
    doc2vec_model.train(tagged_corpus, total_examples=doc2vec_model.corpus_count, 
                epochs=doc2vec_model.epochs)
    
    return doc2vec_model, tagged_corpus
    

def main():
    
    df_survey = pickle.load(open("survey.pkl","rb"))
    
    # Data Cleaning
    lemma = WordNetLemmatizer()
    en_stop = stopwords.words('english')
    exclude_sent = ["see above"]
    processed_comments = df_survey["Comment"].apply(lambda x: clean_text(x,lemma, en_stop, exclude_sent = exclude_sent))
    comments = processed_comments.dropna()
    df_survey = df_survey.loc[comments.index.tolist()].reset_index(drop = True)
    
    # Tokenization
    word_tokenized_corpus = [word_tokenize(sent) for sent in comments]
    tagged_corpus = [TaggedDocument(tokenized_doc, [idx]) 
                     for idx, tokenized_doc in enumerate(word_tokenized_corpus)]
    
    # Doc2Vec Training
    # t = time.time()
    # doc2vec_model = doc2vec_train(word_tokenized_corpus)
    # elapsed = time.time() - t
    # print(elapsed)
    
    # Saving model
    # doc2vec_model.save("doc2vec")
    
    # loading model
    doc2vec_model = Doc2Vec.load("doc2vec")
        
    # Doc2Vec Testing
    
    # Similar comments
    inferred_vector = doc2vec_model.infer_vector(tagged_corpus[823].words)
    sim_comments_index = doc2vec_model.docvecs.most_similar([inferred_vector],topn = 30)
    closest_comments = [[idx, df_survey["Comment"].iloc[idx]] for idx, cosine_sim in sim_comments_index]
    closest_comments = pd.DataFrame(closest_comments)    
    
    #Similar words
    similarity = doc2vec_model.wv.most_similar("leadership")
    print(similarity)
    
    #frequency analysis between word embeddings
    male_freq, female_freq, word_dict, discarded_words = frequency_calculation(word_tokenized_corpus, 
                                                              doc2vec_model, 
                                                              "leadership", 
                                                              "woman", 
                                                              threshold = 0.3,
                                                              diff = 0.1)
    
    wordlist = ["issue", "harassment","lack", "abuse", "diversity"]

    similarity_words, associated_words = most_similar_words(doc2vec_model,wordlist, nwords = 20)
    comment_indexes = associated_comments(word_tokenized_corpus, similarity_words)
    grouping, doc2vec_associations = grouping_words(comment_indexes, df_survey, nwords = 20)
    
    doc2vec_words = [word[0] for word in similarity]  
    Wordcloud_parameters = {"height": 800, "width": 600, "max_words": 20}
    WordCloud_generator(' '.join(doc2vec_words), **Wordcloud_parameters)
   
    ### FAST TEXT
    
    """
    embedding_size = 100
    window_size = 10
    min_word = 5
    down_sampling = 1e-3
    
    t = time.time()
    
    # Fast Text Training
    ft_model = FastText(word_tokenized_corpus, size=embedding_size,
                        window=window_size, min_count=min_word,
                        sample=down_sampling, sg=1, iter=100)    
    
    elapsed = time.time() - t
    print(elapsed)
    
    #Saving model
    pickle.dump(ft_model, open("ft_model.pkl","wb"))
    """
    
    # Loading model
    ft_model = pickle.load(open("ft_model.pkl","rb"))
    
    #Similar words
    ft_similarity = ft_model.wv.most_similar("leadership")
    print(ft_similarity)
    
    wordlist = ["issue", "harassment","lack", "abuse", "diversity"]
    #list_of_words_test = ["woman", "gender", "girl", "women"]
    #list_of_words_test = ["balance","struggle","needs"]
    #list_of_words_test = ["leadership","lack","ceiling"]
    
    #Relations between words
    similarity_words, associated_words = most_similar_words(ft_model,wordlist, nwords = 20)
    comment_indexes = associated_comments(word_tokenized_corpus, similarity_words)
    grouping, ft_associations = grouping_words(comment_indexes, df_survey, nwords = 20)
    
    
    #frequency analysis between word embeddings
    ft_w1_freq, ft_w2_freq, ft_word_dict, ft_discarded_words = frequency_calculation(word_tokenized_corpus, 
                                                              ft_model, 
                                                              "leadership", 
                                                              "woman", 
                                                              threshold = 0.3,
                                                              diff = 0.1)
    
    #WordCloud
    text = associated_words 
    Wordcloud_parameters = {"height": 800, "width": 600, "max_words": 20}
    WordCloud_generator(' '.join(text), **Wordcloud_parameters)
    
    #tsnescatterplot(ft_model, "culture", ["man"])
    
    return doc2vec_associations, ft_associations


if __name__ == "__main__":
    fasttext = main()


