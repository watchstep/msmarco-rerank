import os
import wget
import requests
import tarfile

def get_msmarco_passage_collectionandqueries():
    # https://microsoft.github.io/msmarco/Datasets
    msmarco_passage_path = './collections/msmarco-passage/'
    msmarco_url = 'https://msmarco.blob.core.windows.net/msmarcoranking/collectionandqueries.tar.gz'
    
    if not os.path.exists(msmarco_passage_path):
        os.mkdir(msmarco_passage_path)
        
    response = requests.get(msmarco_url, stream=True)
    file = tarfile.open(fileobj=response.raw, mode='r|gz')
    file.extractall(path=msmarco_passage_path)

def get_msmarco_passage_qrels_dev():
    qrels_dev_url = 'https://msmarco.blob.core.windows.net/msmarcoranking/qrels.dev.tsv'
    wget.download(qrels_dev_url, out='./collections/msmarco-passage/')


def get_msmarco_passage_top1000_tr():
    top1000_tr_url = 'https://msmarco.blob.core.windows.net/msmarcoranking/top1000.train.tar.gz'
    response = requests.get(top1000_tr_url, stream=True)
    file = tarfile.open(fileobj=response.raw, mode='r|gz')
    file.extractall(path='./collections/msmarco-passage')

def get_msmarco_passage_top1000_dev():
    top1000_dev_url = 'https://msmarco.blob.core.windows.net/msmarcoranking/top1000.dev.tar.gz'
    response = requests.get(top1000_dev_url, stream=True)
    file = tarfile.open(fileobj=response.raw, mode='r|gz')
    file.extractall(path='./collections/msmarco-passage')