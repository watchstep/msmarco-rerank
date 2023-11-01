import os
import json
from typing import List

def read_json(json_path:str):
    with open(json_path, 'r') as f:
        lines = f.readlines()
        result = [json.loads(line) for line in lines]
    return result

def save_json(results:List[dict], file_name:str):
    json_path = os.path.join('./retrieved/', f'{file_name}.json')
    json_file = open(json_path, 'w', encoding='utf-8', newline='\n')
    for result in results:
        json_file.write(json.dumps(result) + '\n')
        
    json_file.close()

def read_top1000(top1000_path):
    result = dict()
    with open(top1000_path, 'r') as file:
        for i, line in enumerate(file):
            line = line.strip().split('\t')
            result[int(line[0])] = int(line[1])
    return result

def get_result(results, qids, dids, scores):
    for qid, did, score in zip(qids, dids, scores):
        if qid not in results:
            results[qid] = {}
        
        results[qid][did] = float(score)