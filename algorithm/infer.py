import os
import pickle
import numpy as np
from src.index import Indexer
import torch
import argparse
from src.text_embedding import TextEmbeddingModel
import random

def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.
    np.random.seed(seed)  # Numpy module.
    random.seed(seed)  # Python random module.

def load_pkl(path):
    with open(path, 'rb') as f:
        return pickle.load(f)

def infer(opt):
    model = TextEmbeddingModel(opt.model_name).cuda()
    state_dict = torch.load(opt.model_path, map_location=model.model.device)
    new_state_dict={}
    for key in state_dict.keys():
        if key.startswith('model.'):
            new_state_dict[key[6:]]=state_dict[key]
    model.load_state_dict(state_dict)
    tokenizer=model.tokenizer

    index = Indexer(opt.embedding_dim)
    index.deserialize_from(opt.database_path)
    label_dict=load_pkl(os.path.join(opt.database_path,'label_dict.pkl'))
    is_mixed_dict=load_pkl(os.path.join(opt.database_path,'is_mixed_dict.pkl'))
    
    text = opt.text
    encoded_text = tokenizer.batch_encode_plus(
                        [text],
                        return_tensors="pt",
                        max_length=512,
                        padding="max_length",
                        truncation=True,
                    )
    encoded_text = {k: v.cuda() for k, v in encoded_text.items()}
    embeddings = model(encoded_text).cpu().detach().numpy()
    top_ids_and_scores = index.search_knn(embeddings, opt.K)
    pred = []
    for i, (ids, scores) in enumerate(top_ids_and_scores):
        print(f"Top {opt.K} results for text:")
        cnt = {(1,0):0,(0,10^3):0,(1,1):0}
        for j, (id, score) in enumerate(zip(ids, scores)):
            print(f"{j+1}. ID {id} Label {label_dict[int(id)]} Is_mixed {is_mixed_dict[int(id)]} Score {score}")
            cnt[(label_dict[int(id)], is_mixed_dict[int(id)])]+=1
        final = max(cnt, key=cnt.get)
        pred.append(final)
        if final==(1,0):
            print("Human")
        elif final==(0,10^3):
            print("AI")
        else:
            print("Mixed")
    return pred
        
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--embedding_dim', type=int, default=768)
    parser.add_argument('--database_path', type=str, default="database", help="Path to the index file")

    parser.add_argument("--model_path", type=str, default="/home/heyongxin/detect-LLM-text/DAT/pth/unseen_model/model_best_gpt35.pth",\
                         help="Path to the embedding model checkpoint")
    parser.add_argument('--model_name', type=str, default="princeton-nlp/unsup-simcse-roberta-base", help="Model name")

    parser.add_argument('--K', type=int, default=5, help="Search [1,K] nearest neighbors,choose the best K")
    parser.add_argument('--pooling', type=str, default="average", help="Pooling method, average or cls")
    parser.add_argument('--text', type=str, default="I really want someone to change my view on this, since everyone I know are frowning on me for thinking this way. My argument is, that just with my single vote wouldn't have any effect in the result and thus, it's not worth voting at all But if you don't vote then your opinion doesn't count")
    parser.add_argument('--seed', type=int, default=0)

    opt = parser.parse_args()
    set_seed(opt.seed)
    infer(opt)
