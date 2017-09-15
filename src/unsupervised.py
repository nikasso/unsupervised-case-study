# Galvanize DSI
# Unsupervised
# Week 6, Fri, 7/28/17
# Case Study

from __future__ import division
import numpy as np
import pandas as pd
import json
from bs4 import BeautifulSoup
import requests
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.decomposition import NMF


def parse_one_report(html):
    soup = BeautifulSoup(html,'html.parser')
    # title = [p.text.split(':')[0] for p in soup.find_all('span',class_='field')]
    ps = [p for p in soup.find_all('p')]
    body =  [p.text.split(':') for p in ps]
    title = [p.text for p in soup.find_all('span',class_='field')]
    content = {}
    for b in body:
        content[b[0]] = b[-1]
    if len(title) < 2:
        t  = 'No title available'
    else:
        t = title[1]
    content['Title'] = t
    return content

def parse_all_report(htmls):
    contents = []
    for html in htmls:
        contents.append(parse_one_report(html))
    return contents

def parse_column(des,contents):
    col = []
    for content in contents:
            if des in content.keys():
                col.append(content[des])
            else:
                col.append(None)
    return col


def convert_to_df(contents):
    observations = parse_column('OBSERVED',contents)
    year = parse_column('YEAR',contents)
    season = parse_column('SEASON',contents)
    county = parse_column('COUNTY',contents)
    state = parse_column('STATE',contents)
    title = parse_column('Title',contents)

    df = pd.DataFrame({'State':state,'County':county,'Year':year,'Season':season,'Title':title,'Obs':observations})
    return df

def most_common(H, feat_names):
    '''
    TF-IDF & MF
    '''
    most_common_idx = np.argsort(H, axis=1)[:,:-11:-1]
    # print "Shape of most_common_idx: {}".format(most_common_idx.shape)
    print "\n\nLatent Topics and their most common words: "
    print "==========================================="
    for idx, topic_idx in enumerate(most_common_idx):
        print "\nTopic {}: {}".format(idx+1, np.array(feat_names)[topic_idx])
        print '-'*50
    print '='*80

def most_common_words(doc_term_matrix, feat_names, top_num=10):
    mean_word_freq = np.sum(doc_term_matrix, axis=0)/len(doc_term_matrix)
    most_freq_words_idx = np.argsort(mean_word_freq)[:-11:-1]
    most_freq_words = []
    for idx in most_freq_words_idx:
        most_freq_words.append(feat_names[idx])
    return "Words with top average TF-IDF across corpus: ", most_freq_words

def strong_latent(W, report_titles, num_topics=5):
    '''
    TF-IDF & MF
    '''
    max_idx = np.argsort(W, axis=0)
    max_idx = max_idx[-num_topics:,:]
    max_idx = np.flip(max_idx,axis=0)
    print "\n\nLatent Topics and the reports that contributed most to them: "
    print "============================================================="
    for topic_idx, reports in enumerate(max_idx):
        print "\nLatent {}: Report Title: {}"\
        .format(topic_idx+1,report_titles[reports].encode('ascii','ignore'))
        for report in reports:
            
        print '-'*50
    print '='*80

def fit_nmf(doc_term_matrix, feat_names):
    '''
    Takes in TF-IDF matrix and vocab list
    Decomposes matrix to W & H
    Returns matrices W & H
    '''
    nmf = NMF(n_components = 5,random_state=42)
    W = nmf.fit_transform(doc_term_matrix)
    H = nmf.components_
    most_common(H, feat_names)
    strong_latent(W,report_titles)
    print '+'* 80
    return W,H

if __name__ == '__main__':
    reports = []
    with open('data/bigfoot_data.json') as f:
        for i in f:
            reports.append(json.loads(i))
    #
    urls = [rep['url'] for rep in reports]
    htmls = [rep['html'] for rep in reports]


    contents = parse_all_report(htmls)
    clean_contents = [content for content in contents if 'OBSERVED' in content.keys()]

    df = convert_to_df(clean_contents)

    ### TF-IDF and Matrix Facotrization
    report_titles = df.Title

    np.random.seed(42)
    obs = df.Obs

    wc_vect = CountVectorizer(max_features=1000)
    wc_matrix = wc_vect.fit_transform(obs).toarray()

    vect = TfidfVectorizer(max_features=1000, stop_words='english')
    doc_term_matrix = vect.fit_transform(obs).toarray()
    feat_names = vect.get_feature_names()
    fit_nmf(doc_term_matrix, feat_names)
    most_freq_words = most_common_words(doc_term_matrix, feat_names)
    print most_freq_words
