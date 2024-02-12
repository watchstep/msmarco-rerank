import os, sys
import argparse
import json


# Setup ElasticSearch
# %%bash
# sudo -H -u daemon elasticsearch-7.9.2/bin/elasticsearch&
# ps -ef | grep elasticsearch
# curl -X GET "localhost:9200/"

p_prompt = f"""
Passage: <P> Given the provided passage, generate 3 similar passages on related topic: <T>
"""

q_prompt = """
<P> Review the given passages and answer a specific and detailed query. {'Query: Your query here.'}”
"""

parser = argparse.ArgumentParser(description = "Prompting with ExLlamaV2")
parser.add_argument("--dataset", type = str, default ='nq')
parser.add_argument("--seed", type = int, default = 2023)
parser.add_argument("--topk", type=int, default=10)
parser.add_argument("--model_dir", type=str, default="/root/Mistral-7B-instruct-exl2")
parser.add_argument("--p_prompt", type=str, default=p_prompt)
parser.add_argument("--q_prompt", type=str, default=q_prompt)

args = parser.parse_args([])

from beir import util
from beir.datasets.data_loader import GenericDataLoader

url = f"https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/{args.dataset}.zip"
data_path = util.download_and_unzip(url, './data/')

corpus, queries, qrels = GenericDataLoader(data_folder=data_path).load(split="test")

from beir.retrieval.search.lexical import BM25Search as BM25
from beir.retrieval.evaluation import EvaluateRetrieval

hostname = "localhost"
index_name = "nq"
initialize = True

model = BM25(index_name=index_name, hostname=hostname, initialize=initialize)
retriever = EvaluateRetrieval(model)

def save_json(results, file_name:str):
    json_path = f'./output/{args.dataset}_{file_name}.json'
    with open(json_path, 'w') as f:
        json.dump(results, f, indent=4)

def read_json(json_path:str):
    with open(json_path, 'r') as f:
        data = f.read()
        output = json.loads(data)
    return output

results = read_json('/root/workspace/output/nq_bm25_es_retrieved.json')
output = read_json('/root/workspace/output/nq_eval_output.json')

from exllamav2 import(
    ExLlamaV2,
    ExLlamaV2Config,
    ExLlamaV2Cache,
    ExLlamaV2Tokenizer,
)

from exllamav2.generator import (
    ExLlamaV2BaseGenerator,
    ExLlamaV2Sampler
)

# from beir.reranking import Rerank
from typing import List, Tuple
import torch
import torch.nn.functional as F

# https://github.com/beir-cellar/beir/blob/main/beir/reranking/rerank.py
from typing import Dict, List

class Rerank:
    def __init__(self, model, batch_size: int = 128, **kwargs):
        self.model = model
        self.batch_size = batch_size
        self.rerank_results = {}
    
    def rerank(self,
               corpus: Dict[str, Dict[str, str]],
               queries: Dict[str, str],
               results: Dict[str, Dict[str, float]],
               top_k: int) -> Dict[str, Dict[str, float]]:
        
        sentence_pairs, pair_ids = [], []
        
        for query_id in results:
            if len(results[query_id]) > top_k:
                for (doc_id, _) in sorted(results[query_id].items(), key=lambda item: item[1], reverse=True)[:top_k]:
                    pair_ids.append([query_id, doc_id])
                    corpus_text = {'title': corpus[doc_id].get("title", "").strip(), 'text': corpus[doc_id].get("text", "").strip()}
                    sentence_pairs.append([queries[query_id], corpus_text])
                else:
                    for doc_id in results[query_id]:
                        pair_ids.append([query_id, doc_id])
                        corpus_text = {'title': corpus[doc_id].get("title", "").strip(), 'text': corpus[doc_id].get("text", "").strip()}
                        # corpus_text = (corpus[doc_id].get("title", "") + " " + corpus[doc_id].get("text", "")).strip()
                        sentence_pairs.append([queries[query_id], corpus_text])
        
        #### Starting to Rerank using cross-attention
        rerank_scores = [float(score) for score in self.model.predict(sentence_pairs, batch_size=self.batch_size)]
        
        #### Reranking results
        self.rerank_results = {query_id: {} for query_id in results}
        for pair, score in zip(pair_ids, rerank_scores):
            query_id, doc_id = pair[0], pair[1]
            self.rerank_results[query_id][doc_id] = score
        
        return self.rerank_results
    

class ExLlamaV2Reranker:
    def __init__(self, p_prompt=args.p_prompt, q_prompt=args.q_prompt, model_dir=args.model_dir, seed=args.seed, max_new_tokens=400, **kwargs):
        self.p_prompt = p_prompt
        self.q_prompt = q_prompt
        self.seed = seed
        self.max_new_tokens = max_new_tokens
        
        self.config = ExLlamaV2Config()
        self.config.model_dir = model_dir
        self.config.prepare()
        
        self.model = ExLlamaV2(self.config)
        if not self.model.loaded:
            self.cache = ExLlamaV2Cache(self.model, lazy = True)
            self.model.load_autosplit(self.cache)
            
        self.tokenizer = ExLlamaV2Tokenizer(self.config)
        self.generator = ExLlamaV2BaseGenerator(self.model, self.cache, self.tokenizer)
        self.generator.warmup()
        
        self.settings = ExLlamaV2Sampler.Settings()
        self.settings.temperature = 0.6
        self.settings.top_k = 50
        self.settings.top_p = 0.9
        self.settings.token_repetition_penalty = 1.15
        self.settings.disallow_tokens(self.tokenizer, [self.tokenizer.eos_token_id])
    
    def predict(self, sentences: List[Tuple[str,str]], batch_size: int, **kwags) -> List[float]:
        scores = []
        for query, psg in sentences:
            self.p_prompt = self.p_prompt.replace('<P>', psg['text']).replace('<T>', psg['title']).strip()
            generated = self.generator.generate_simple(self.p_prompt, self.settings, self.max_new_tokens, seed=self.seed)
            self.q_prompt = self.q_prompt.replace('<P>', generated).strip()
            with torch.inference_mode():
                input_ids = self.tokenizer.encode(self.q_prompt)
                # input_ids = input_ids.shape[-1]
                # self.cache.current_seq_len = 0
                logits = self.model.forward(input_ids[:, -1:], self.cache)
                logits = logits[:, :-1, :]
                logits = logits.float() + 1e-10
                log_probs = F.log_softmax(logits, dim=-1)
            scores.append(log_probs)
        
        assert len(scores) == len(sentences)
        return scores
    
reranker = Rerank(ExLlamaV2Reranker(), batch_size=128)
rerank_results = reranker.rerank(corpus, queries, results, top_k=100)

# save output
output['reranking'] = EvaluateRetrieval.evaluate(qrels, rerank_results, retriever.k_values)

with open(f'/content/{args.dataset}_eval_ouput.json', 'w') as f:
    json.dump(output, f, indent=4)