# 1. Import all necessary libraries

import nltk
from nltk.corpus import stopwords
from nltk.cluster.util import cosine_distance
import numpy as np
import networkx as nx
 
# ******************************************************************************************************************************    

# 2. Generate clean sentences

def read_article(file_name):
    file = open(file_name, "r")
    filedata = file.readlines()
    article = filedata[0].split(". ")
    sentences = []
#     print(article[0])
    for sentence in article:
#          print(sentence)
        sentences.append(sentence.replace("[^a-zA-Z]", " ").split(" "))
    sentences.pop() 
    return sentences

# ******************************************************************************************************************************

# 3. Similarity matrix

def sentence_similarity(sent1, sent2, stopwords=None):
    if stopwords is None:
        stopwords = []
 
    sent1=[w.lower() for w in sent1]
    sent2=[w.lower() for w in sent2]
#     print(sent1)
 
    all_words = list(set(sent1 + sent2))
#     print(all_words)
    vector1 = [0]*len(all_words)
    vector2 = [0]*len(all_words)
#     print(vector2)

    # build the vector for the first sentence
    for w in sent1:
        if w in stopwords:
            continue
        vector1[all_words.index(w)] += 1
#         print(vector1)
 
    # build the vector for the second sentence
    for w in sent2:
        if w in stopwords:
            continue
        vector2[all_words.index(w)] += 1
        
    return  1-cosine_distance(vector1, vector2)
 
    
# cosine_distance(u, v)
# Returns the cosine of the angle between vectors v and u. This is equal to u.v / |u||v|.
# nltk.cluster.cosine_distance(u, v)[source]
# Returns 1 minus the cosine of the angle between vectors v and u. This is equal to 1 - (u.v / |u||v|).
    
 # ******************************************************************************************************************************   
 
# 4. Similarity matrix

def build_similarity_matrix(sentences, stop_words):
    # Create an empty similarity matrix
    similarity_matrix = np.zeros((len(sentences), len(sentences)))
#     print(similarity_matrix)
 
    for idx1 in range(len(sentences)):
        for idx2 in range(len(sentences)):
            if idx1==idx2: #ignore if both are same sentences
                continue 
            similarity_matrix[idx1][idx2] = sentence_similarity(sentences[idx1], sentences[idx2], stop_words)
#             print(idx1)

    return similarity_matrix



# ******************************************************************************************************************************

# 4. Generate Summary Method
def generate_summary(file_name, top_n=5):
#     nltk.download("stopwords")
    stop_words = stopwords.words('english')
    summarize_text = []

    # Step 1 - Read text anc split it
    sentences =  read_article(file_name)
#     print(len(sentences))
    
#     print(sentences[0])

    # Step 2 - Generate Similary Martix across sentences
    sentence_similarity_martix = build_similarity_matrix(sentences, stop_words)
#     print((sentence_similarity_martix )) 
    
    
#     Step 3 - Rank sentences in similarity martix
    sentence_similarity_graph = nx.from_numpy_array(sentence_similarity_martix)
#       Returns a graph from a 2D NumPy array.
#      The 2D NumPy array is interpreted as an adjacency matrix for the graph.


#     print(sentence_similarity_graph)
    scores = nx.pagerank(sentence_similarity_graph)
#     PageRank computes a ranking of the nodes in the graph G based on the structure of the incoming links.
#     It was originally designed as an algorithm to rank web pages.
    
    
#     print(scores)
#     print(sentence_similarity_martix)

#     Step 4 - Sort the rank and pick top sentences
    ranked_sentence = sorted(((scores[i],s) for i,s in enumerate(sentences)), reverse=True)    
#     print("Indexes of top ranked_sentence order are ", ranked_sentence)    

    for i in range(top_n):
       summarize_text.append(" ".join(ranked_sentence[i][1]))

    # Step 5 - Offcourse, output the summarize texr
    print("Summarize Text: \n", ". ".join(summarize_text))
# ******************************************************************************************************************************

# let's begin
generate_summary( "summary.txt", 2)

# ******************************************************************************************************************************