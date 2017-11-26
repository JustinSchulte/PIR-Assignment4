
# coding: utf-8

# In[1]:


import csv
from collections import defaultdict
import math
import numpy as np

from whoosh.index import create_in
from whoosh.fields import *
from whoosh import scoring
from whoosh.qparser import QueryParser


# Read the TSV file, removing newlines:

# In[2]:


def read_file(file_path, delimiter='\t'):
    with open(file_path, 'r', encoding='utf-8') as csvfile:
        reader = csv.reader(csvfile, delimiter=delimiter, quotechar='|', quoting=csv.QUOTE_MINIMAL)
        return [(doc_id, rel, content.replace('\n', ' ')) for doc_id, rel, content in reader]
doc_list = read_file('collection.tsv')
print('read {} docs'.format(len(doc_list)))


# Create the index:

# In[3]:


schema = Schema(id=ID(stored=True), content=TEXT)
ix = create_in('cw_index', schema)
writer = ix.writer()

for doc in doc_list:
    writer.add_document(id=doc[0], content=doc[2])
writer.commit()


# We define a helper function for searching:

# In[4]:


def search_index(ix, query_str, ranking_fn, limit=None):
    result_list = []
    with ix.searcher(weighting=ranking_fn) as searcher:
        query = QueryParser('content', ix.schema).parse(query_str)
        results = searcher.search(query, limit=limit)
        for result in results:
            result_list.append(result['id'])
        return result_list


# We read the csv file and return a 2-dimensional dictionary:
# 
# `all_qrels[query][doc_id] = score`

# In[5]:


def read_qrels(file_path, delimiter=' '):
    with open(file_path, 'r', encoding='utf-8') as csvfile:
        reader = csv.reader(csvfile, delimiter=delimiter)
        result = defaultdict(dict)
        for query, doc_id, score in reader:
            result[query][doc_id] = int(score)
    return result
all_qrels = read_qrels('q5.web.qrels.txt')


# We define the scoring functions:

# In[6]:


def tf_idf(searcher, fieldname, text, matcher):
    return matcher.value_as('frequency') * searcher.get_parent().idf(fieldname, text)
tf_idf_weighting = scoring.FunctionWeighting(tf_idf)

def pos(searcher, fieldname, text, matcher):
    return matcher.value_as('frequency') * matcher.value_as('positions')[0]
pos_weighting = scoring.FunctionWeighting(pos)


#TODO Assignment
def BM25(searcher, fieldname, text, matcher):
    k1 = 1.5
    b = 0.75
    idf = searcher.get_parent().idf(fieldname, text)
    tf = matcher.value_as('frequency')
    dl = searcher.doc_field_length(matcher.id(), fieldname, 1)
    avdl = searcher.get_parent().avg_field_length(fieldname)
	
    numerator = idf * (k1 + 1) * tf
    denumerator = k1 * ((1-b) + b * (dl/avdl)) + tf
    result = numerator/denumerator
    return result
bm25_weighting = scoring.FunctionWeighting(BM25)

def jenilek(searcher, fieldname, text, matcher):
    tf = matcher.value_as('frequency')
    dl = searcher.doc_field_length(matcher.id(), fieldname, 1)
    tf_all = searcher.frequency(fieldname, text)
    dl_all = searcher.get_parent().field_length(fieldname)
    
    n = 0.5
    PD = tf/dl
    PC = tf_all/dl_all
    result = n * PD + (1-n) * PC
    result = np.log(result) #use the logarithm, because it will be summed up
    return result
jenilek_weighting = scoring.FunctionWeighting(jenilek)


# Now search for all queries using both scoring functions. The query words are separated by underscores in the file, so we have to convert them.

# In[7]:


results_tf_idf = {}
results_pos = {}
results_tfidf_sys = {}
results_bm25 = {}
results_jenilek = {}
for query in all_qrels:
    query_new = query.replace('_', ' ')
    results_tf_idf[query] = search_index(ix, query_new, tf_idf_weighting)
    results_pos[query] = search_index(ix, query_new, pos_weighting)
    results_tfidf_sys[query] = search_index(ix, query_new, scoring.TF_IDF)
    results_bm25[query] = search_index(ix, query_new, bm25_weighting)
    results_jenilek[query] = search_index(ix, query_new, jenilek_weighting)


# Implement $\text{P}@k$.
# 
# We count the number of documents in our top-$k$ results (`doc_list[:k]`) that have a query relevance larger than $0$ (true positives). We divide by the total number of retrieved items ($k$).

# In[8]:


def precision(doc_list, qrels, k):
    true_pos = len([doc_id for doc_id in doc_list[:k] if qrels.get(doc_id, 0) > 0])
    return true_pos / k


# Implement recall.
# 
# We count the number of documents in our results (`doc_list`) that have a query relevance larger than $0$ (true positives). We divide by the total number of relevant items.

# In[9]:


def recall(doc_list, qrels):
    true_pos = len([doc_id for doc_id in doc_list if qrels.get(doc_id, 0) > 0])
    total_relevant = len([rel for rel in qrels.values() if rel > 0])
    return true_pos / total_relevant


# Implement $\text{NDCG}@k$.
# 
# We calculate the DCG using the first $k$ documents by dividing their relevances by the log of the position. Since we have negative relevances in the qrel file, we take the maximum of $0$ and the relevance.
# 
# The ideal DCG is given by the $k$ highest relevances divided by the logs of their positions. We sort the qrel values and use the top-$k$ ones.
# 
# We get the normalized DCG by dividing the DCG by the ideal DCG.

# In[10]:


def ndcg(doc_list, qrels, k):
    dcg = 0
    qrels_sorted = sorted(qrels.values(), reverse=True)
    idcg = 0
    for i in range(1, k + 1):
        rel = max(0, qrels.get(doc_list[i - 1], 0))
        dcg += rel / math.log2(i + 1)
        idcg += max(0, qrels_sorted[i - 1]) / math.log2(i + 1)

    return dcg / idcg


# Implement $\text{MAP}@k$.
# 
# For every query, we calculate the average precision, that is, we calculate $P@i$ for all $0 < i \leq k$ where the $i$'th document was relevant and take the average.
# 
# The MAP is the mean of all average precision values, i.e. across all queries.

# In[11]:


def avg_prec(doc_list, qrels, k):
    total = 0
    num = 0
    for i in range(1, k + 1):
        if qrels.get(doc_list[i - 1], 0) > 0:
            total += precision(doc_list, qrels, i)
            num += 1
    if num == 0:
        return 0
    return total / num

def mean_avg_prec(doc_lists, all_qrels, k):
    total = 0
    for key in doc_lists:
        total += avg_prec(doc_lists[key], all_qrels[key], k)
    return total / len(doc_lists)


# Report the results:

# In[12]:


k = 10
for query in all_qrels:
    print('QUERY: {}\n'
          '{} documents\n'.format(query, len(results_pos[query])))
    
    p_at_k = precision(results_tf_idf[query], all_qrels[query], k)
    r = recall(results_tf_idf[query], all_qrels[query])
    ndcg_at_k = ndcg(results_tf_idf[query], all_qrels[query], k)
    print('tf-idf scoring:\n'
          'P@{} = {}\n'
          'R = {}\n'
          'NDCG@{} = {}\n'.format(k, p_at_k, r, k, ndcg_at_k))
    
    p_at_k = precision(results_pos[query], all_qrels[query], k)
    r = recall(results_pos[query], all_qrels[query])
    ndcg_at_k = ndcg(results_pos[query], all_qrels[query], k)
    print('Term position scoring:\n'
          'P@{} = {}\n'
          'R = {}\n'
          'NDCG@{} = {}\n'.format(k, p_at_k, r, k, ndcg_at_k))
    
    p_at_k = precision(results_tfidf_sys[query], all_qrels[query], k)
    r = recall(results_tfidf_sys[query], all_qrels[query])
    ndcg_at_k = ndcg(results_tfidf_sys[query], all_qrels[query], k)
    print('Whoosh tf-idf scoring:\n'
          'P@{} = {}\n'
          'R = {}\n'
          'NDCG@{} = {}\n'.format(k, p_at_k, r, k, ndcg_at_k))
    
    p_at_k = precision(results_bm25[query], all_qrels[query], k)
    r = recall(results_bm25[query], all_qrels[query])
    ndcg_at_k = ndcg(results_bm25[query], all_qrels[query], k)
    print('BM25 scoring:\n'
          'P@{} = {}\n'
          'R = {}\n'
          'NDCG@{} = {}\n'.format(k, p_at_k, r, k, ndcg_at_k))
    
    p_at_k = precision(results_jenilek[query], all_qrels[query], k)
    r = recall(results_jenilek[query], all_qrels[query])
    ndcg_at_k = ndcg(results_jenilek[query], all_qrels[query], k)
    print('Jenilek Mercer scoring:\n'
          'P@{} = {}\n'
          'R = {}\n'
          'NDCG@{} = {}'.format(k, p_at_k, r, k, ndcg_at_k))

    print('\n========================================\n')

map_at_k = mean_avg_prec(results_tf_idf, all_qrels, k)
print('tf-idf scoring:\n'
      'MAP@{} = {}'.format(k, map_at_k))
map_at_k = mean_avg_prec(results_pos, all_qrels, k)
print('Term position scoring:\n'
      'MAP@{} = {}'.format(k, map_at_k))
map_at_k = mean_avg_prec(results_bm25, all_qrels, k)
print('BM25 scoring:\n'
      'MAP@{} = {}'.format(k, map_at_k))
map_at_k = mean_avg_prec(results_jenilek, all_qrels, k)
print('Jenilek Mercer scoring:\n'
      'MAP@{} = {}'.format(k, map_at_k))

