# Domain-Specific AI Assistant Enhancement Project

## Overview
This project focuses on enhancing the quality of Large Language Models (LLMs) for domain-specific AI assistants.
The implementation leverages UniXcoder for traditional code search capabilities and works with the CodeSearchNet dataset, with plans to expand to custom repositories.

## Team Members
- Carter Cutsforth (cvcuts@iastate.edu)
- Jacob Duba (jduba@iastate.edu)
- Keenan Jacobs (kcjacobs@iastate.edu)
- Conor O'Shea (coshea@iastate.edu)
- Diego Perez (joceo@iastate.edu)

## Getting Started

### Prerequisites
- UV package installer ([Installation guide](https://github.com/astral-sh/uv))

### Installation
1. Create and activate UV environment:
   ```bash
   uv venv
   # If using bash/zsh
   source .venv/bin/activate
   # If using fish
   source .venv/bin/activate.fish
   ```
2. Install dependencies:
   ```bash
   make install
   ```

### Running
Run the script that creates the data:
```bash
make create
```
Run the code search script:
```bash
make search
```

## License
*License information to be added*

## Contact
For questions or collaboration inquiries, please contact any team member listed above.


## Benchmarking
Note: This should be done in CodeSearch

## Data Download

#### 1. AdvTest dataset

```bash
mkdir dataset && cd dataset
wget https://github.com/microsoft/CodeXGLUE/raw/main/Text-Code/NL-code-search-Adv/dataset.zip
unzip dataset.zip && rm -r dataset.zip && mv dataset AdvTest && cd AdvTest
wget https://zenodo.org/record/7857872/files/python.zip
unzip python.zip && python preprocess.py && rm -r python && rm -r *.pkl && rm python.zip
cd ../..
```

#### 2. CosQA dataset

```bash
cd dataset
mkdir cosqa && cd cosqa
wget https://github.com/Jun-jie-Huang/CoCLR/raw/main/data/search/code_idx_map.txt
wget https://github.com/Jun-jie-Huang/CoCLR/raw/main/data/search/cosqa-retrieval-dev-500.json
wget https://github.com/Jun-jie-Huang/CoCLR/raw/main/data/search/cosqa-retrieval-test-500.json
wget https://github.com/Jun-jie-Huang/CoCLR/raw/main/data/search/cosqa-retrieval-train-19604.json
cd ../..
```

#### 3. CSN dataset

```bash
cd dataset
wget https://github.com/microsoft/CodeBERT/raw/master/GraphCodeBERT/codesearch/dataset.zip
unzip dataset.zip && rm -r dataset.zip && mv dataset CSN && cd CSN
bash run.sh 
cd ../..
```


## Dependency 

- pip install torch
- pip install transformers

## Zero-Shot Setting

We first provide scripts for zero-shot code search. The similarity between code and nl we use is cosine distance of hidden states of UniXcoder.

#### 1. AdvTest dataset

```bash
python run.py \
    --output_dir saved_models/AdvTest \
    --model_name_or_path microsoft/unixcoder-base  \
    --do_zero_shot \
    --do_test \
    --test_data_file dataset/AdvTest/test.jsonl \
    --codebase_file dataset/AdvTest/test.jsonl \
    --num_train_epochs 2 \
    --code_length 256 \
    --nl_length 128 \
    --train_batch_size 64 \
    --eval_batch_size 64 \
    --learning_rate 2e-5 \
    --seed 123456
```

#### 2. CosQA dataset

```bash
python run.py \
    --output_dir saved_models/cosqa \
    --model_name_or_path microsoft/unixcoder-base  \
    --do_zero_shot \
    --do_test \
    --test_data_file dataset/cosqa/cosqa-retrieval-test-500.json \
    --codebase_file dataset/cosqa/code_idx_map.txt \
    --num_train_epochs 10 \
    --code_length 256 \
    --nl_length 128 \
    --train_batch_size 64 \
    --eval_batch_size 64 \
    --learning_rate 2e-5 \
    --seed 123456
```

#### 3. CSN dataset

```bash
lang=python
python run.py \
    --output_dir saved_models/CSN/$lang \
    --model_name_or_path microsoft/unixcoder-base  \
    --do_zero_shot \
    --do_test \
    --test_data_file dataset/CSN/$lang/test.jsonl \
    --codebase_file dataset/CSN/$lang/codebase.jsonl \
    --num_train_epochs 10 \
    --code_length 256 \
    --nl_length 128 \
    --train_batch_size 64 \
    --eval_batch_size 64 \
    --learning_rate 2e-5 \
    --seed 123456
```

---
*This project is part of COM S 4020 at Iowa State University*
