import os
import pickle
import random
import faiss
from src.index import Indexer
import torch
import torch.nn.functional as F
import numpy as np
from torch.utils.data import DataLoader
from lightning import Fabric
from tqdm import tqdm
import argparse
from src.text_embedding import TextEmbeddingModel
from utils.load_dataset import load_dataset, TextDataset, load_outdomain_dataset

def load_pkl(path):
    with open(path, 'rb') as f:
        return pickle.load(f)

def infer(passages_dataloder,fabric,tokenizer,model,ood=False):
    if fabric.global_rank == 0 :
        passages_dataloder=tqdm(passages_dataloder,total=len(passages_dataloder))
        if ood:
            allids, allembeddings,alllabels,all_is_mixed= [],[],[],[]
        else:
            allids, allembeddings,alllabels,all_is_mixed,all_write_model= [],[],[],[],[]
    model.model.eval()
    with torch.no_grad():
        for batch in passages_dataloder:
            if ood:
                ids, text, label, is_mixed = batch
                encoded_batch = tokenizer.batch_encode_plus(
                            text,
                            return_tensors="pt",
                            max_length=512,
                            padding="max_length",
                            # padding=True,
                            truncation=True,
                        )
                encoded_batch = {k: v.cuda() for k, v in encoded_batch.items()}
                # output = model(**encoded_batch).last_hidden_state
                # embeddings = pooling(output, encoded_batch)  
                # print(encoded_batch)
                embeddings = model(encoded_batch)
                # print(encoded_batch['input_ids'].shape)
                embeddings = fabric.all_gather(embeddings).view(-1, embeddings.size(1))
                label = fabric.all_gather(label).view(-1)
                ids = fabric.all_gather(ids).view(-1)
                is_mixed = fabric.all_gather(is_mixed).view(-1)
                if fabric.global_rank == 0 :
                    allembeddings.append(embeddings.cpu())
                    allids.extend(ids.cpu().tolist())
                    alllabels.extend(label.cpu().tolist())
                    all_is_mixed.extend(is_mixed.cpu().tolist())
            else:
                ids, text, label, is_mixed, write_model = batch
                encoded_batch = tokenizer.batch_encode_plus(
                            text,
                            return_tensors="pt",
                            max_length=512,
                            padding="max_length",
                            # padding=True,
                            truncation=True,
                        )
                encoded_batch = {k: v.cuda() for k, v in encoded_batch.items()}
                # output = model(**encoded_batch).last_hidden_state
                # embeddings = pooling(output, encoded_batch)  
                # print(encoded_batch)
                embeddings = model(encoded_batch)
                # print(encoded_batch['input_ids'].shape)
                embeddings = fabric.all_gather(embeddings).view(-1, embeddings.size(1))
                label = fabric.all_gather(label).view(-1)
                ids = fabric.all_gather(ids).view(-1)
                is_mixed = fabric.all_gather(is_mixed).view(-1)
                write_model = fabric.all_gather(write_model).view(-1)
                if fabric.global_rank == 0 :
                    allembeddings.append(embeddings.cpu())
                    allids.extend(ids.cpu().tolist())
                    alllabels.extend(label.cpu().tolist())
                    all_is_mixed.extend(is_mixed.cpu().tolist())
                    all_write_model.extend(write_model.cpu().tolist())
    if fabric.global_rank == 0 :
        allembeddings = torch.cat(allembeddings, dim=0)
        epsilon = 1e-6
        if ood:
            emb_dict,label_dict,is_mixed_dict={},{},{}
            allembeddings= F.normalize(allembeddings,dim=-1)
            for i in range(len(allids)):
                emb_dict[allids[i]]=allembeddings[i]
                label_dict[allids[i]]=alllabels[i]
                is_mixed_dict[allids[i]]=all_is_mixed[i]
            allids,allembeddings,alllabels,all_is_mixed=[],[],[],[]
            for key in emb_dict:
                allids.append(key)
                allembeddings.append(emb_dict[key])
                alllabels.append(label_dict[key])
                all_is_mixed.append(is_mixed_dict[key])
            allembeddings = torch.stack(allembeddings, dim=0)
            return allids,allembeddings.numpy(),alllabels,all_is_mixed
        else:
            emb_dict,label_dict,is_mixed_dict,write_model_dict={},{},{},{}
            allembeddings= F.normalize(allembeddings,dim=-1)
            for i in range(len(allids)):
                emb_dict[allids[i]]=allembeddings[i]
                label_dict[allids[i]]=alllabels[i]
                is_mixed_dict[allids[i]]=all_is_mixed[i]
                write_model_dict[allids[i]]=all_write_model[i]
            allids,allembeddings,alllabels,all_is_mixed,all_write_model=[],[],[],[],[]
            for key in emb_dict:
                allids.append(key)
                allembeddings.append(emb_dict[key])
                alllabels.append(label_dict[key])
                all_is_mixed.append(is_mixed_dict[key])
                all_write_model.append(write_model_dict[key])
            allembeddings = torch.stack(allembeddings, dim=0)
            return allids, allembeddings.numpy(),alllabels,all_is_mixed,all_write_model
    else:
        if ood:
            return [],[],[],[]
        return [],[],[],[],[]

def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.
    np.random.seed(seed)  # Numpy module.
    random.seed(seed)  # Python random module.

def test(opt):
    if opt.device_num>1:
        fabric = Fabric(accelerator="cuda",devices=opt.device_num,strategy='ddp')
    else:
        fabric = Fabric(accelerator="cuda",devices=opt.device_num)
    fabric.launch()
    model = TextEmbeddingModel(opt.model_name).cuda()
    state_dict = torch.load(opt.model_path, map_location=model.model.device)
    new_state_dict={}
    for key in state_dict.keys():
        if key.startswith('model.'):
            new_state_dict[key[6:]]=state_dict[key]
    model.load_state_dict(state_dict)
    tokenizer=model.tokenizer
    database = load_dataset(opt.dataset_name,opt.database_path)[opt.database_name]  
    passage_dataset = TextDataset(database,need_ids=True)
    print(len(passage_dataset))

    passages_dataloder = DataLoader(passage_dataset, batch_size=opt.batch_size, num_workers=opt.num_workers, pin_memory=True)
    passages_dataloder=fabric.setup_dataloaders(passages_dataloder)
    model=fabric.setup(model)

    train_ids, train_embeddings,train_labels, train_is_mixed, train_write_model = infer(passages_dataloder,fabric,tokenizer,model)
    fabric.barrier()

    if fabric.global_rank == 0:
        index = Indexer(opt.embedding_dim)
        index.index_data(train_ids, train_embeddings)
        label_dict={}
        is_mixed_dict={}
        write_model_dict={}
        for i in range(len(train_ids)):
            label_dict[train_ids[i]]=train_labels[i]
            is_mixed_dict[train_ids[i]]=train_is_mixed[i]
            write_model_dict[train_ids[i]]=train_write_model[i]

        if not os.path.exists(opt.save_path):
            os.makedirs(opt.save_path)
        index.serialize(opt.save_path)
        #save label_dict using pickle
        with open(os.path.join(opt.save_path, 'label_dict.pkl'), 'wb') as f:
            pickle.dump(label_dict, f)
        #save is_mixed_dict using pickle
        with open(os.path.join(opt.save_path, 'is_mixed_dict.pkl'), 'wb') as f:
            pickle.dump(is_mixed_dict, f)
        #save write_model_dict using pickle
        with open(os.path.join(opt.save_path, 'write_model_dict.pkl'), 'wb') as f:
            pickle.dump(write_model_dict, f)

def add_to_existed_index(opt):
    if opt.device_num>1:
        fabric = Fabric(accelerator="cuda",devices=opt.device_num,strategy='ddp')
    else:
        fabric = Fabric(accelerator="cuda",devices=opt.device_num)
    fabric.launch()
    model = TextEmbeddingModel(opt.model_name).cuda()
    state_dict = torch.load(opt.model_path, map_location=model.model.device)
    new_state_dict={}
    for key in state_dict.keys():
        if key.startswith('model.'):
            new_state_dict[key[6:]]=state_dict[key]
    model.load_state_dict(state_dict)
    tokenizer=model.tokenizer

    if opt.ood:
        database = load_outdomain_dataset(opt.database_path)[opt.database_name]
    else:
        database = load_dataset(opt.dataset_name,opt.database_path)[opt.database_name]

    passage_dataset = TextDataset(database,need_ids=True,out_domain=opt.ood)
    print(len(passage_dataset))

    passages_dataloder = DataLoader(passage_dataset, batch_size=opt.batch_size, num_workers=opt.num_workers, pin_memory=True)
    passages_dataloder=fabric.setup_dataloaders(passages_dataloder)
    model=fabric.setup(model)

    if opt.ood:
        train_ids, train_embeddings,train_labels, train_is_mixed = infer(passages_dataloder,fabric,tokenizer,model,ood=True)
    else:
        train_ids, train_embeddings,train_labels, train_is_mixed, train_write_model = infer(passages_dataloder,fabric,tokenizer,model)
    fabric.barrier()

    if fabric.global_rank == 0:
        new_index = Indexer(opt.embedding_dim)
        new_index.index_data(train_ids, train_embeddings)

        old_index = Indexer(opt.embedding_dim)
        old_index.deserialize_from(opt.existed_index_path)
        old_ids = old_index.index_id_to_db_id

        # Ensure both indexes are of type IndexFlatIP
        # assert isinstance(new_index.index, faiss.IndexFlatIP)
        # assert isinstance(old_index.index, faiss.IndexFlatIP)

        # Ensure both indexes have the same dimensionality
        assert new_index.index.d == old_index.index.d

        # Extract vectors from old_index.index
        vectors = old_index.index.reconstruct_n(0, old_index.index.ntotal)

        # Add vectors to new_index.index
        new_index.index_data(old_ids, vectors)

        if not os.path.exists(opt.new_save_path):
            os.makedirs(opt.new_save_path)
        new_index.serialize(opt.new_save_path)

        if opt.ood:
            label_dict=load_pkl(os.path.join(opt.existed_index_path, 'label_dict.pkl'))
            is_mixed_dict=load_pkl(os.path.join(opt.existed_index_path, 'is_mixed_dict.pkl'))
            for i in range(len(train_ids)):
                label_dict[train_ids[i]]=train_labels[i]
                is_mixed_dict[train_ids[i]]=train_is_mixed[i]
            #save label_dict using pickle
            with open(os.path.join(opt.new_save_path, 'label_dict.pkl'), 'wb') as f:
                pickle.dump(label_dict, f)
            #save is_mixed_dict using pickle
            with open(os.path.join(opt.new_save_path, 'is_mixed_dict.pkl'), 'wb') as f:
                pickle.dump(is_mixed_dict, f)

        else:
            label_dict=load_pkl(os.path.join(opt.existed_index_path, 'label_dict.pkl'))
            is_mixed_dict=load_pkl(os.path.join(opt.existed_index_path, 'is_mixed_dict.pkl'))
            write_model_dict=load_pkl(os.path.join(opt.existed_index_path, 'write_model_dict.pkl'))
            for i in range(len(train_ids)):
                label_dict[train_ids[i]]=train_labels[i]
                is_mixed_dict[train_ids[i]]=train_is_mixed[i]
                write_model_dict[train_ids[i]]=train_write_model[i]
            #save label_dict using pickle
            with open(os.path.join(opt.new_save_path, 'label_dict.pkl'), 'wb') as f:
                pickle.dump(label_dict, f)
            #save is_mixed_dict using pickle
            with open(os.path.join(opt.new_save_path, 'is_mixed_dict.pkl'), 'wb') as f:
                pickle.dump(is_mixed_dict, f)
            #save write_model_dict using pickle
            with open(os.path.join(opt.new_save_path, 'write_model_dict.pkl'), 'wb') as f:
                pickle.dump(write_model_dict, f)





if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--device_num', type=int, default=1)
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--num_workers', type=int, default=8)
    parser.add_argument('--embedding_dim', type=int, default=768)

    parser.add_argument("--database_path", type=str, default="data/FALCONSet", help="Path to the data")
    parser.add_argument('--dataset_name', type=str, default='falconset', help="falconset, llmdetectaive, hart")
    parser.add_argument('--database_name', type=str, default='train', help="train,valid,test,test_ood")
    parser.add_argument("--model_path", type=str, default="runs/authscan_v6/model_best.pth",\
                         help="Path to the embedding model checkpoint")
    parser.add_argument('--model_name', type=str, default="FacebookAI/xlm-roberta-base", help="Model name")
    parser.add_argument("--save_path", type=str, default="/output", help="Path to save the database")
    parser.add_argument("--add_to_existed_index", type=int, default=0)
    parser.add_argument("--ood", type=int, default=0)
    parser.add_argument("--existed_index_path", type=str, default="/output", help="Path of existed index")
    parser.add_argument("--new_save_path", type=str, default="/new_db", help="Path to save the database")
    
    parser.add_argument('--seed', type=int, default=0)
    opt = parser.parse_args()
    set_seed(opt.seed)

    if not opt.add_to_existed_index:
        test(opt)
    else:
        add_to_existed_index(opt)
    