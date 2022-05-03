import boto3
import pandas as pd
import swifter
import os
import re
import pandas as pd
import s3fs
import sys
import pprint
import textstat
from sklearn import cluster
from collections import defaultdict
import en_core_web_lg
nlp = en_core_web_lg.load()
import spacy
from spacy import displacy
import json
import pandas as pd
import numpy as np
import nltk
from nltk.stem import PorterStemmer, WordNetLemmatizer
nltk.download('vader_lexicon')
import matplotlib.pyplot as plt
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('vader_lexicon')
nltk.download('punkt')
from nltk.corpus import stopwords
from nltk.probability import FreqDist
from nltk.tokenize import word_tokenize
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import time
import json



#CLEANING

def clean_data(df):

    pd.options.mode.chained_assignment = None

    print("******Cleaning Started*****")

    print(f'Shape of df before cleaning : {df.shape}')
    #df['review_date'] = pd.to_datetime(df['review_date'])
    df = df[df['review'].notna()]
    df['review'] = df['review'].str.replace("<br />", " ")
    df['review'] = df['review'].str.replace("\[?\[.+?\]?\]", " ")
    df['review'] = df['review'].str.replace("\/{3,}", " ")
    df['review'] = df['review'].str.replace("\&\#.+\&\#\d+?;", " ")
    df['review'] = df['review'].str.replace("\d+\&\#\d+?;", " ")
    df['review'] = df['review'].str.replace("\&\#\d+?;", " ")

    #facial expressions
    df['review'] = df['review'].str.replace("\:\|", "")
    df['review'] = df['review'].str.replace("\:\)", "")
    df['review'] = df['review'].str.replace("\:\(", "")
    df['review'] = df['review'].str.replace("\:\/", "")

    df['review'] = df['review'].str.replace("["
            u"\U0001F600-\U0001F64F"  # emoticons
            u"\U0001F300-\U0001F5FF"  # symbols & pictographs
            u"\U0001F680-\U0001F6FF"  # transport & map symbols
            u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
                              "]+", "")
    #replace multiple spaces with single space
    df['review'] = df['review'].str.replace("\s{2,}", " ")
    #replace multiple fullstops by one
    df['review'] = df['review'].str.replace('\.+', ".")

    df['review'] = df['review'].str.lower()
    print(f'Shape of df after cleaning : {df.shape}')
    print("******Cleaning Ended*****")


    return(df)


# ******************************************************************
# RULES TO EXTRACT ASPECTS
def apply7_extraction(row,nlp,sid):
    game_pronouns=["it", "this", "they", "product"]

    badsymbols=['#','*']
    stprmv=''
    word_tokens = word_tokenize(row)

    
    swords=stopwords.words('english')
    not_stopwords_list=['not']
    final_stopwords_list = set([word for word in swords if word not in not_stopwords_list])
    filtered_sentence = [w for w in word_tokens if not w in final_stopwords_list]

    for w in filtered_sentence:
        if w!='#' or w!='*':
            stprmv+=' '+w
    text=stprmv
    e=''
    d=nlp(text)
#         Lemmatising
    for token in d:
        token=token.lemma_
        e=e+token+' '

    doc=nlp(e)
#print("--- SPACY : Doc loaded ---")

    rule1_pairs = []
    rule2_pairs = []
    rule3_pairs = []
    rule4_pairs = []
    rule5_pairs = []
    rule6_pairs = []
    rule7_pairs = []

    for token in doc:
        A = "999999"
        M = "999999"
        if token.dep_ == "amod" and not token.is_stop:
            M = token.text
            A = token.head.text

            # add adverbial modifier of adjective (e.g. 'most comfortable headphones')
            M_children = token.children
            for child_m in M_children:
                if(child_m.dep_ == "advmod"):
                    M_hash = child_m.text
                    M = M_hash + " " + M
                    break

            # negation in adjective, the "no" keyword is a 'det' of the noun (e.g. no interesting characters)
            A_children = token.head.children
            for child_a in A_children:
                if(child_a.dep_ == "det" and child_a.text == 'no'):
                    neg_prefix = 'not'
                    M = neg_prefix + " " + M
                    break

        if(A != "999999" and M != "999999"):
            if A in game_pronouns or A=='-PRON-' :
                A = "Game"
            dict1 = {"noun" : A, "adj" : M, "rule" : 1, "polarity" : sid.polarity_scores(token.text)['compound']}
            rule1_pairs.append(dict1)

        # print("--- SPACY : Rule 1 Done ---")


        children = token.children
        A = "999999"
        M = "999999"
        add_neg_pfx = False
        for child in children :
            if(child.dep_ == "nsubj" and not child.is_stop):
                A = child.text
                # check_spelling(child.text)

            if((child.dep_ == "dobj" and child.pos_ == "ADJ") and not child.is_stop):
                M = child.text
                #check_spelling(child.text)

            if(child.dep_ == "neg"):
                neg_prefix = child.text
                add_neg_pfx = True

        if (add_neg_pfx and M != "999999"):
            M = neg_prefix + " " + M

        if(A != "999999" and M != "999999"):
            if A in game_pronouns or A=='-PRON-':
                A = "Game"
            dict2 = {"noun" : A, "adj" : M, "rule" : 2, "polarity" : sid.polarity_scores(token.text)['compound']}
            rule2_pairs.append(dict2)




        # print("--- SPACY : Rule 2 Done ---")


        children = token.children
        A = "999999"
        M = "999999"
        add_neg_pfx = False
        for child in children :
            if(child.dep_ == "nsubj" and not child.is_stop):
                A = child.text
                # check_spelling(child.text)

            if(child.dep_ == "acomp" and not child.is_stop):
                M = child.text

            # example - 'this could have been better' -> (this, not better)
            if(child.dep_ == "aux" and child.tag_ == "MD"):
                neg_prefix = "not"
                add_neg_pfx = True

            if(child.dep_ == "neg"):
                neg_prefix = child.text
                add_neg_pfx = True

        if (add_neg_pfx and M != "999999"):
            M = neg_prefix + " " + M
                #check_spelling(child.text)

        if(A != "999999" and M != "999999"):
            if A in game_pronouns or A=='-PRON-':
                A = "Game"
            dict3 = {"noun" : A, "adj" : M, "rule" : 3, "polarity" : sid.polarity_scores(token.text)['compound']}
            rule3_pairs.append(dict3)
            #rule3_pairs.append((A, M, sid.polarity_scores(M)['compound'],3))
    # print("--- SPACY : Rule 3 Done ---")



        children = token.children
        A = "999999"
        M = "999999"
        add_neg_pfx = False
        for child in children :
            if((child.dep_ == "nsubjpass" or child.dep_ == "nsubj") and not child.is_stop):
                A = child.text
                # check_spelling(child.text)

            if(child.dep_ == "advmod" and not child.is_stop):
                M = child.text
                M_children = child.children
                for child_m in M_children:
                    if(child_m.dep_ == "advmod"):
                        M_hash = child_m.text
                        M = M_hash + " " + child.text
                        break
                #check_spelling(child.text)

            if(child.dep_ == "neg"):
                neg_prefix = child.text
                add_neg_pfx = True

        if (add_neg_pfx and M != "999999"):
            M = neg_prefix + " " + M

        if(A != "999999" and M != "999999"):
            if A in game_pronouns or A=='-PRON-':
                A = "Game"
            dict4 = {"noun" : A, "adj" : M, "rule" : 4, "polarity" : sid.polarity_scores(token.text)['compound']}
            rule4_pairs.append(dict4)
            #rule4_pairs.append((A, M,sid.polarity_scores(M)['compound'],4)) # )

    # print("--- SPACY : Rule 4 Done ---")




        children = token.children
        A = "999999"
        buf_var = "999999"
        for child in children :
            if(child.dep_ == "nsubj" and not child.is_stop):
                A = child.text
                # check_spelling(child.text)

            if(child.dep_ == "cop" and not child.is_stop):
                buf_var = child.text
                #check_spelling(child.text)

        if(A != "999999" and buf_var != "999999"):
            if A in game_pronouns or A=='-PRON-':
                A = "Game"
            dict5 = {"noun" : A, "adj" : token.text, "rule" : 5, "polarity" : sid.polarity_scores(token.text)['compound']}
            rule5_pairs.append(dict5)
            #rule5_pairs.append((A, token.text,sid.polarity_scores(token.text)['compound'],5))

    # print("--- SPACY : Rule 5 Done ---")



        children = token.children
        A = "999999"
        M = "999999"
        if(token.pos_ == "INTJ" and not token.is_stop):
            for child in children :
                if(child.dep_ == "nsubj" and not child.is_stop):
                    A = child.text
                    M = token.text
                    # check_spelling(child.text)

        if(A != "999999" and M != "999999"):
            if A in game_pronouns or A=='-PRON-':
                A = "Game"
            dict6 = {"noun" : A, "adj" : M, "rule" : 6, "polarity" : sid.polarity_scores(M)['compound']}
            rule6_pairs.append(dict6)

            #rule6_pairs.append((A, M,sid.polarity_scores(M)['compound'],6))

    # print("--- SPACY : Rule 6 Done ---")


        children = token.children
        A = "999999"
        M = "999999"
        add_neg_pfx = False
        for child in children :
            if(child.dep_ == "nsubj" and not child.is_stop):
                A = child.text
                # check_spelling(child.text)

            if((child.dep_ == "attr") and not child.is_stop):
                M = child.text
                #check_spelling(child.text)

            if(child.dep_ == "neg"):
                neg_prefix = child.text
                add_neg_pfx = True

        if (add_neg_pfx and M != "999999"):
            M = neg_prefix + " " + M

        if(A != "999999" and M != "999999"):
            if A in game_pronouns or A=='-PRON-':
                A = "Game"
            dict7 = {"noun" : A, "adj" : M, "rule" : 7, "polarity" : sid.polarity_scores(M)['compound']}
            rule7_pairs.append(dict7)
            #rule7_pairs.append((A, M,sid.polarity_scores(M)['compound'],7))



    #print("--- SPACY : Rules Done ---")


    aspects = []

    aspects = rule1_pairs + rule2_pairs + rule3_pairs +rule4_pairs +rule5_pairs + rule6_pairs + rule7_pairs


    dic = {"aspect_pairs" : aspects}
    # i+=1
    return dic
# ******************************************************************


#num_partitions = 10 #number of partitions to split dataframe
#num_cores = 4
sid=SentimentIntensityAnalyzer()
game_pronouns = ['it','this','they','these']
session = boto3.Session( 
         aws_access_key_id='<YOUR_ACCESS_ID>', 
         aws_secret_access_key='<YOUR_KEY>')





# #Then use the session to get the resource
s3 = session.resource('s3')

my_bucket = s3.Bucket('diva-11')
urls="s3://diva-11/"
record_count=0
keys =[]
output_name=[]
for key in my_bucket.objects.all():
  keys.append("s3://diva-11/"+key.key)
  #gamereviews=11
  dat=""
  dat+=key.key
  dat=dat[12:-4]
  output_name.append('Result_'+dat)


count=0
for i in keys:
    print()
    print("------------------------------------------------------------------")
    print("Running for file : ",i)
    t1=time.time()
    print()
    reviews_decomp=[]
    df=pd.read_csv(i)
    cleaned_df=clean_data(df)
    # record_count+=df.shape[0]
    aspect_list = cleaned_df['review'].swifter.apply(lambda row: apply7_extraction(row,nlp,sid))
    # aspect_list = parallelize_dataframe(cleaned_df, multiply_columns)
    aspects_tuples=[]
    for review in (aspect_list.items()):
    
        aspect_pairs = review[1]
        for number, pairs in enumerate(aspect_pairs['aspect_pairs']):
            # print(pairs)
            aspects_tuples.append(pairs)

    noun=[]
    adj=[]
    aspects=[]
    dictionaryBag=[]
    for i in range(0,len(aspects_tuples)):
        noun.append(aspects_tuples[i]['noun'])
        adj.append(aspects_tuples[i]['adj'])
        aspects.append(aspects_tuples[i]['noun'])
        dictionaryBag.append(aspects_tuples[i]['noun'])
        dictionaryBag.append(aspects_tuples[i]['adj'])
    unique_aspects = list(set(aspects))

    print(len(unique_aspects))

    aspects_map = defaultdict(int)
    for asp in aspects:
        aspects_map[asp] += 1


    asp_vectors = []
    for aspect in unique_aspects:
        #print(aspect)
        token = nlp(aspect)
        asp_vectors.append(token.vector)

    # print("\n\n Aspect Vectors",asp_vectors)

    NUM_CLUSTERS = 8
    if len(unique_aspects) <= NUM_CLUSTERS:
        NUM_CLUSTERS=len(unique_aspects)
        
        print(list(range(len(unique_aspects))))

    # print("Running k-means clustering...")
    n_clusters = NUM_CLUSTERS
    kmeans = cluster.KMeans(n_clusters=n_clusters)
    kmeans.fit(asp_vectors)
    labels = kmeans.labels_

    
   #TIME TAKING
    asp_to_cluster_map = dict(zip(unique_aspects,labels))
    cluster_id_to_name_map = defaultdict()
    cluster_to_asp_map = defaultdict()
    freq_map={}
    for i in range(NUM_CLUSTERS):
        cluster_nouns = [k for k,v in asp_to_cluster_map.items() if v == i]
    #     print(cluster_nouns)
        freq_map = {k:v for k,v in aspects_map.items() if k in cluster_nouns}
        freq_map = sorted(freq_map.items(), key = lambda x: x[1], reverse = True)
    #     print(freq_map)
        cluster_id_to_name_map[i] = freq_map[0][0]
        cluster_to_asp_map[i] = cluster_nouns #see clusters better


    #BAG OF ASPECTS
    dictionaryBag={}
    k=''

    for i in range(0,len(aspects_tuples)):
        
        v=[]
        k=aspects_tuples[i]['noun']
        v.append(aspects_tuples[i]['adj'])
        #v.append(aspects_tuples[i]['polarity'])
        for j in range(1,len(aspects_tuples)):
            if(k==aspects_tuples[j]['noun']):
                v.append(aspects_tuples[j]['adj'])
        dictionaryBag[k]=v

    #PRINT FEATURE'S ADJECTIVES
    features=[]
    for i in cluster_id_to_name_map:
        # print("Feature=",cluster_id_to_name_map[i])
        features.append(cluster_id_to_name_map[i])

    f=open(output_name[count]+".txt",'w+')
    f.write(json.dumps(cluster_id_to_name_map))


    #CALUCLATING MEAN OF POLARITIES['compund'] OF EACH ASPECT
    finalFeaturePol=[]
    for i in cluster_id_to_name_map:
        s=[]
        # print("Feature=",cluster_id_to_name_map[i])
        for j in range(0,len(dictionaryBag[cluster_id_to_name_map[i]])):
            s.append(sid.polarity_scores(dictionaryBag[cluster_id_to_name_map[i]][j])['compound'])
        xf=sum(s)/len(s)
        finalFeaturePol.append(xf)

    print("\n\nASPECTS:")
    for i in range(0,len(features)):
        print(features[i],"\t",finalFeaturePol[i],"\n")
    print("Unique Aspects",len(unique_aspects))
    print("Total Aspects Extracted",len(aspects_tuples))
    

    finalDF=pd.DataFrame({"Aspects":features,"SentimentScore":finalFeaturePol})
    
    finalDF.to_csv(output_name[count]+'.csv')
    if count<len(output_name):
        count+=1
    t2=time.time()
    print("Completed ",output_name[i])
    print("Time Taken:",t2-t1)
    print("------------------------------------------------------------------")