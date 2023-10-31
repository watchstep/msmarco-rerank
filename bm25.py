import os
import random
import json
import argparse
import multiprocessing

import numpy as np
import pandas as pd

import torch

import pyserini
from pyserini.search import SimpleSearcher
from pyserini.dsearch import SimpleDenseSearcher
from pyserini.search.lucene import LuceneSearcher
from typing import List

class Indexer:
    def __init__(self, jsonl_path, index_path):
        self.jsonl_path = jsonl_path
        self.index_path = index_path # indexes/lucene-index-msmarco-passage
    
    def build_sparse_index(self):
        execute_code = os.system('python -m pyserini.index.lucene ' + 
                                 '--collection JsonCollection ' +
                                 f'--input {self.jsonl_path} ' +
                                 f'--index {self.index_path} ' +
                                 '--generator DefaultLuceneDocumentGenerator ' +
                                 '--threads 1 --storeRaw')
        if execute_code != 0:
            raise Exception('Indexing Failed!')
        else:
            print('Indexing Success!')
    
    def build_dense_index(self):
        pass 

class BM25Retriever:
    def __init__(self, index_path, k, k1=1.5, b=0.75):
        self.searcher = LuceneSearcher(index_path)
        self.searcher.set_bm25(k1=k1, b=b)
        self.k = k
            
    def _get_results(self, qid, hits:List):
        results = []
        
        for i, hit in enumerate(hits):
            docid = hit.docid
            content = json.loads(hits[i].raw)['contents']
            bm25_score = hit.score
            result = {'rank': i+1,
                      'qid': qid,
                      'docid': docid, 
                      'score': bm25_score,
                      'content': content}
            results.append(result)
            
        return results
    
    def _save_json(self, results:List[dict], file_name:str):
        json_path = os.path.join('./retrieved/', f'{file_name}.json')
        json_file = open(json_path, 'w', encoding='utf-8', newline='\n')
        for result in results:
            json_file.write(json.dumps(result) + '\n')
        
        json_file.close()
    
    def search(self, qid, query_text:str):
        search_results = {}
        hits = self.searcher.search(query_text, k=self.k,)
        search_results['query'] = query_text
        search_results['hits']  = self._get_results(qid, hits)
        
        return search_results
    
    def batch_search(self, qids:List[str], query_texts: List[str], is_save:bool):
        query_dict = dict(zip(qids, query_texts))
        batch_hits = self.searcher.batch_search(query_texts, qids, k=self.k, threads=multiprocessing.cpu_count())
        bsearch_results = []
        bsearch_items = {}

        for qid, hits in batch_hits.items():
            bsearch_items['query'] = query_dict[qid]
            bsearch_items['hits'] = self._get_results(qid, hits)
            bsearch_results.append(bsearch_items)
            bsearch_items = {}
            
        if is_save:
            self._save_json(bsearch_results)
       
        return bsearch_results
    
# if not os.path.exists(index_path):
#             indexer = Indexer()
#             self.build_sparse_index(jsonl_path, index_path)   