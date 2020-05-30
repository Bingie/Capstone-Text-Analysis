

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import pickle
import itertools
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
from collections import Counter
from gensim.models.fasttext import FastText
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
import logging
import time
import random 
from utils import *


logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)


def cleaning_comments(survey):
    
    # Data Cleaning
    lemma = WordNetLemmatizer()
    en_stop = stopwords.words('english')
    exclude_sent = ["see above"]
    processed_comments = survey["Comment"].apply(lambda x: clean_text(x,lemma, en_stop, exclude_sent = exclude_sent))
    comments = processed_comments.dropna()
    df_survey = survey.loc[comments.index.tolist()].reset_index(drop = True)
    
    return comments, df_survey


def TF_IDF(comments):
    
    vectorizer = TfidfVectorizer(ngram_range = (1, 2))
    tf_idf = vectorizer.fit_transform(comments)
    
    avg_tfidf = tf_idf.mean(axis = 0)
    words_tfidf = vectorizer.get_feature_names()
    
    tf_idf_dict = {}
    for idx, word in enumerate(words_tfidf):
        tf_idf_dict[word] = 1/float(avg_tfidf[:,idx])
        
    return tf_idf_dict
    

def most_similar_words(model, words, nwords = 30):
    
    word_sim_collection = {}
    word_collection = {}
    
    for word in words:    
        word_sim_collection[word] = model.wv.most_similar(word, topn = nwords)
        
        list_words = []
        for tuple_collection in word_sim_collection[word]:
            list_words.append(tuple_collection[0])
        
        word_collection[word] = list_words
        
    return word_collection, word_sim_collection


def sorted_similar_words(similar_words, criteria):
    
    sorted_words = {}
    
    for term, list_words in similar_words.items():
        
        dict_words = {}
        for word in list_words:
            try:
                dict_words[word] = criteria[word]
            except KeyError:
                print("word \"{}\" not in the dictionary".format(word))
                continue
        
        sorted_words[term] = dict_words

    return sorted_words


def associated_comments(tokenized_corpus, word_dicts):
    
    comment_index = []
    for key, words in word_dicts.items():
        words.append(key)
        for word in words: 
            index = [idx for idx,comment in enumerate(tokenized_corpus) 
                          if key in comment if word in comment]
            comment_index.append([key,word,index])
    
    return comment_index
            

def combine_comments(comment_indexes, survey, selected_columns):
    
    df = []
    ### TODO Catch key error, column not found + text 
    for comment_index in comment_indexes:
        keyword, similar_words, indexes = comment_index
        comment = survey.loc[indexes,selected_columns]
        comment["Similar words"] = similar_words
        df.append(comment)
    
    df = pd.concat(df)
    
    return df


def grouping_similar_words(comment_indexes, survey, selected_columns = ["Comment"]):
    
    new_columns = selected_columns.copy()
    
    if "Comment" not in new_columns:
        new_columns.append("Comment")
    
    combined_comments = combine_comments(comment_indexes, survey, new_columns)
    combined_comments = combined_comments.reset_index()
    combined_comments["index"] = combined_comments["index"].apply(lambda x: str(x))
    
    new_columns.append("index")
    
    grouping = combined_comments.groupby(new_columns)["Similar words"].apply(','.join)
    grouping = grouping.str.split(",")
    grouping = grouping.reset_index()

    grouping["length"] = [len(x) for x in grouping["Similar words"]]
    grouping = grouping[grouping["length"] > 1] 
    grouping = grouping.sort_values(by = "length", ascending = False)
    
    return grouping, combined_comments


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
    
    return doc2vec_model


def word_embeddings(model, words):
    
    list_words = [word for word in words if word in model.wv.vocab]
    
    embeddings = {}
    
    for word in list_words:
        embeddings[word] = model.wv[list_words]
      
    return embeddings


def plot_frequency_comments(grouped_comments, selected_columns):
        
    ncols = 2 
    nrows = int(np.ceil(len(selected_columns)/ncols))
    
    fig, axes = plt.subplots(ncols = ncols, nrows = nrows)
    
    for column, ax in zip(selected_columns,axes.flatten()):
        count_group = grouped_comments.groupby(column).count()
        group = count_group.loc[:,"Comment"]
        
        title = "Frequency of column {}".format(column)
        group.plot(kind = "bar", color = "blue", title = title, ax = ax, rot = 30)
    

def data_company2_relations():
    #selected_dict["environment"] = ["friendly","positive", "safe", "fine"]
    #selected_dict["culture"] =  ["toxic", "abuse","pervasive", "bro", "frat"]
    #selected_dict["erg"] = ["bonus","grassroots","reward","sponsorship","official","wlf"]
    #selected_dict["leadership"] = ["dominated","behavior","style","skill","aware",
    #"bias"]
    return None


def doc2vec_test(model, tokenized_corpus, comments):
    
    ### TODO!!!!
    
    tagged_corpus = [TaggedDocument(tokenized_doc, [idx]) 
                     for idx, tokenized_doc in enumerate(word_tokenized_corpus)]
    
    inferred_vector = model.infer_vector(tagged_corpus[3].words)
    sim_comments_index = model.docvecs.most_similar([inferred_vector],topn = 30)
    closest_comments = [[idx, comments.iloc[idx]] for idx, cosine_sim in sim_comments_index]
    closest_comments = pd.DataFrame(closest_comments)    
    
    #Similar words
    similarity = model.wv.most_similar("erg")
    print(doc2vec_model.wv.similarity("erg", "know"))
    print(doc2vec_model.wv.similarity("erg", "what"))
    print(similarity)
    
    #frequency analysis between word embeddings
    male_freq, female_freq, word_dict, discarded_words = frequency_calculation(word_tokenized_corpus, 
                                                              doc2vec_model, 
                                                              "leadership", 
                                                              "woman", 
                                                              threshold = 0.3,
                                                              diff = 0.1)
    
    wordlist = ["remote", "home", "flexible"]

    similarity_words, associated_words = most_similar_words(doc2vec_model,wordlist, nwords = 20)
    comment_indexes = associated_comments(word_tokenized_corpus, similarity_words)
    grouping, doc2vec_associations = grouping_words(comment_indexes, df_survey, nwords = 20)
    
    doc2vec_words = [word[0] for word in similarity]  
    Wordcloud_parameters = {"height": 800, "width": 600, "max_words": 20}
    WordCloud_generator(' '.join(doc2vec_words), **Wordcloud_parameters)


def main(survey):
    selected_words = ["issue", "harassment", "lack", "environment", "culture", "flexibility", "leadership", "ceiling", "woman", "positive"]
    for i in range(5):
        main_run(survey, random.sample(selected_words, 2)) # pick 2 words randomly


def main_run(survey, words):
    
    #Data Cleaning
    comments, survey = cleaning_comments(survey)
            
    # Tokenization
    word_tokenized_corpus = [word_tokenize(sent) for sent in comments]
    
    # TD-IDF
    tf_idf_dict = TF_IDF(comments)
        
    """
    # Doc2Vec Training

    t = time.time()
    doc2vec_model = doc2vec_train(word_tokenized_corpus)
    elapsed = time.time() - t
    print(elapsed)
    
    # Saving model
    doc2vec_model.save("doc2vec")
    
    # loading model
    #doc2vec_model = Doc2Vec.load("doc2vec")
    
    """
    
           
    ### FAST TEXT    
    embedding_size = 100
    window_size = 10
    min_word = 5
    down_sampling = 1e-3
    
    t = time.time()
    
    # Fast Text Training
    ft_model = FastText(word_tokenized_corpus, size=embedding_size,
                        window=window_size, min_count=min_word,
                        sample=down_sampling, sg=1, iter=150, hs = 1)    
    
    elapsed = time.time() - t
    print(elapsed)
     
    #Saving model
    pickle.dump(ft_model, open("ft_model.pkl","wb"))
    
    
    # Loading model
    #ft_model = pickle.load(open("ft_model.pkl","rb"))
     
    
    selected_words = words
    max_words = 20
    
    #Relations between words
    similar_words, _ = most_similar_words(ft_model, selected_words, 
                                             nwords = max_words)
    
    sorted_words = sorted_similar_words(similar_words, tf_idf_dict)

    ##WordCloud
    for word in selected_words:
        Wordcloud_parameters = {"height": 800, "width": 600, "max_words": max_words}
        title = "Top {0} words associated with term: {1}".format(max_words, word)
        WordCloud_generator(sorted_words[word], title, **Wordcloud_parameters)
    
    selected_columns = ["Location", "Supervisor",  
                        "Area", "Gender"]
    
    comment_indexes = associated_comments(word_tokenized_corpus, similar_words)
    grouped_comments, ft_associations = grouping_similar_words(comment_indexes, survey,
                                                       selected_columns = selected_columns)
    
    #grouped_comments = grouped_comments.loc[grouped_comments['Leadership advancing women.'] <= 5]
    
    plot_frequency_comments(grouped_comments, selected_columns)

    
    if words.length == 1:
        file_name = words[0]
    else:
        file_name = ",".join(words)

    grouped_comments.to_excel(file_name +".xlsx")
    
    return grouped_comments


    #frequency analysis between word embeddings
    ft_w1_freq, ft_w2_freq, ft_word_dict, ft_discarded_words = frequency_calculation(word_tokenized_corpus, 
                                                              ft_model, 
                                                              "leadership", 
                                                              "woman", 
                                                              threshold = 0.3,
                                                              diff = 0.1)
    

    

    #tsnescatterplot(ft_model, "culture", ["man"])
    A = word_embeddings(ft_model, ["woman","female","girl","women"])
    X = word_embeddings(ft_model, ["problem","issue","sexual","harassment"])
    B = word_embeddings(ft_model, ["man","male","boy","bro"])
    Y = word_embeddings(ft_model, ["success","leadership","senior","executive"])
    
    
    
    
    return grouping


if __name__ == "__main__":
    
    filepath = "Cleaned Spreadsheet Directory Here!!!"
    df_survey = pd.read_excel(filepath)
    fasttext = main(df_survey)

