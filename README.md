# Reranking with LLaMA2 

```bash
rerank
├─ .conda
├─ .gitignore
├─ anserini-tools
├─ collections
│  └─ msmarco-passage
│     └─ collection_jsonl
├─ indexes
├─ hf_ms_marco_dataset.ipynb
└─ rerank.ipynb
```

```bash
(base) conda create -n rerank python=3.8 # pyserini 패키지 설치때문에 3.8 환경
(base) conda activate rerank

(rerank) conda install wget
(rerank) conda install -c conda-forge openjdk=11
(rerank) conda install -c conda-forge maven
(rerank) conda install -c conda-forge lightgbm
(rerank) conda install -c conda-forge faiss-cpu
(rerank) conda install pytorch torchvision torchaudio -c pytorch
(rerank) pip install --no-binary :all: nmslib
(rerank) pip install -e .

(rerank) pip install pyserini==0.18.0 pygaggle # pip show pyserini 버전 꼭 확인하기
(rerank) pip install transformers accelerate sentencepiece huggingface peft bitsandbytes datasets trl
```
