""" 

Word Embeddings Association Test

"""

import numpy as np
from itertools import permutations


def cosine_similarity(vec1, vec2):
    """
    Calculates cosine similarity between vectors

    Parameters
    ----------
    vec1 : Array
        DESCRIPTION. Word embeddings of set A
    vec2 : Array
        DESCRIPTION. Word embeddings of set B

    Returns
    -------
    cos_sim : Float
        DESCRIPTION. cosine similarity metric

    """
    
    #Vector Normalization
    vec1 = vec1 / np.linalg.norm(vec1, axis = -1, keepdims = True) 
    vec2 = vec2 / np.linalg.norm(vec2, axis = -1, keepdims = True)
    
    cos_sim = np.tensordot(vec1,vec2, axes = (-1,-1))
    
    return cos_sim

def word_association(W, A, B):
    
    mean_A = np.mean(cosine_similarity(W,A), axis = -1)
    mean_B = np.mean(cosine_similarity(W,B), axis = -1)
    
    diff = mean_A - mean_B
    
    return diff
    
def test_statistic(X, Y, A, B):
    
    return np.sum(word_association(X, A, B)) - np.sum(word_association(Y, A, B))

def permutation_pairs(X, Y):
    """
    

    Parameters
    ----------
    X : TYPE
        DESCRIPTION.
    Y : TYPE
        DESCRIPTION.

    Returns
    -------
    pairs : TYPE
        DESCRIPTION.

    """
    
    X = set(X)
    Y = set(Y)
    
    X_Y = X.union(Y)
    
    perms_1 = permutations(X_Y, len(X))
    perms_2 = [perm for perm in permutations(X_Y, len(X))]
        
    pairs = []
    test = []
    for e1 in perms_1:
        for e2 in perms_2:
            if e1 != e2:
                if any(l1 == l2 for l1 in e1 for l2 in e2):
                    continue
                else:
                    pairs.append([e1,e2])
   
    return pairs


def word_vectors(dict_word, words):
    
    assert len(words) > 0
    
    word_vec = 0
    
    for i, word in enumerate(words):
        if i == 0:
            word_vec = dict_word[word]
        else:    
            np.append(word_vec, dict_word[word])
    
    return word_vec


def word_arrays(X,Y,A,B):
    
    X_vec = word_vectors(X, X.keys())
    Y_vec = word_vectors(Y, Y.keys())
    A_vec = word_vectors(A, A.keys())
    B_vec = word_vectors(B, B.keys())
    
    return [X_vec, Y_vec, A_vec, B_vec]

def p_value(X,Y,A,B):
              
    [X_vec, Y_vec, A_vec, B_vec] = word_arrays(X,Y,A,B)
    
    S = test_statistic(X_vec, Y_vec, A_vec, B_vec)
    
    pairs = permutation_pairs(X.keys(), Y.keys())
    
    Target = {**X,**Y}
    
    partition_test_statistic = []
    
    for pair in pairs:
        partition_1 = word_vectors(Target, pair[0])
        partition_2 = word_vectors(Target, pair[1])
        partition_test_statistic.append(test_statistic(partition_1, 
                                                       partition_2, 
                                                       A_vec, B_vec))
    
    return np.sum(partition_test_statistic > S) / len(partition_test_statistic)
    
def effect_size(X, Y, A, B):
    """
    

    Parameters
    ----------
    X : TYPE
        DESCRIPTION.
    Y : TYPE
        DESCRIPTION.
    A : TYPE
        DESCRIPTION.
    B : TYPE
        DESCRIPTION.

    Returns
    -------
    TYPE
        DESCRIPTION.

    """
    
    [X_vec, Y_vec, A_vec, B_vec] = word_arrays(X,Y,A,B)
    
    x_association = word_association(X, A, B)
    y_association = word_association(Y, A, B)

    mean = np.mean(x_association, axis=-1) - np.mean(y_association, axis=-1)
    std = np.std(np.concatenate((x_association, y_association), axis=0))

    return mean / std