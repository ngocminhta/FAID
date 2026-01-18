import argparse
import os
import pickle
from collections import Counter

import numpy as np
import torch
from gen_database import infer, set_seed
from lightning import Fabric
from matplotlib import pyplot as plt
from src.index import Indexer
from src.text_embedding import TextEmbeddingModel
from torch.utils.data import DataLoader
from utils.load_dataset import TextDataset, load_dataset, load_outdomain_dataset
from utils.utils import compute_metrics


def softmax_weights(scores, temperature=1.0):
    scores = np.array(scores)
    scores = scores / temperature
    e_scores = np.exp(scores - np.max(scores))
    return e_scores / np.sum(e_scores)


def normalize_fuzzy_cnt(fuzzy_cnt):
    total = sum(fuzzy_cnt.values())
    if total == 0:
        return fuzzy_cnt
    for key in fuzzy_cnt:
        fuzzy_cnt[key] /= total
    return fuzzy_cnt


def class_type_boost(query_type, candidate_type):
    if query_type == candidate_type:
        return 1.3
    elif abs(query_type - candidate_type) == 1:
        return 1.1
    elif abs(query_type - candidate_type) == 2:
        return 0.9
    else:
        return 0.8


def load_pkl(path):
    with open(path, "rb") as f:
        return pickle.load(f)


def test_3_label(opt):
    if opt.device_num > 1:
        fabric = Fabric(accelerator="cuda", devices=opt.device_num, strategy="ddp")
    else:
        fabric = Fabric(accelerator="cuda", devices=opt.device_num)
    fabric.launch()
    model = TextEmbeddingModel(opt.model_name).cuda()
    state_dict = torch.load(opt.model_path, map_location=model.model.device)
    new_state_dict = {}
    for key in state_dict.keys():
        if key.startswith("model."):
            new_state_dict[key[6:]] = state_dict[key]
    model.load_state_dict(state_dict)
    tokenizer = model.tokenizer

    if opt.out_domain == 1:
        print("Using out domain dataset")
        test_database = load_outdomain_dataset(opt.out_domain_mode)[opt.test_dataset_name]
    else:
        test_database = load_dataset(opt.dataset_name, opt.test_dataset_path)[opt.test_dataset_name]

    test_dataset = TextDataset(test_database, need_ids=True, out_domain=opt.out_domain)

    test_dataloder = DataLoader(
        test_dataset, batch_size=opt.batch_size, num_workers=opt.num_workers, pin_memory=True
    )
    test_dataloder = fabric.setup_dataloaders(test_dataloder)
    model = fabric.setup(model)

    if opt.out_domain == 1:
        test_ids, test_embeddings, test_labels, test_is_mixed = infer(
            test_dataloder, fabric, tokenizer, model, ood=True
        )
    else:
        test_ids, test_embeddings, test_labels, test_is_mixed, test_write_model = infer(
            test_dataloder, fabric, tokenizer, model
        )
    fabric.barrier()

    if fabric.global_rank == 0:
        index = Indexer(opt.embedding_dim)
        index.deserialize_from(opt.database_path)

        label_dict = load_pkl(os.path.join(opt.database_path, "label_dict.pkl"))
        is_mixed_dict = load_pkl(os.path.join(opt.database_path, "is_mixed_dict.pkl"))

        test_labels = [(test_labels[i], test_is_mixed[i]) for i in range(len(test_labels))]

        # preds = [] # List[Tuple]
        preds = {i: [] for i in range(1, opt.max_K + 1)}
        if len(test_embeddings.shape) == 1:
            test_embeddings = test_embeddings.reshape(1, -1)
        top_ids_and_scores = index.search_knn(test_embeddings, opt.max_K)

        for i, (ids, scores) in enumerate(top_ids_and_scores):
            sorted_scores = np.argsort(scores)
            sorted_scores = sorted_scores[::-1]

            for k in range(1, opt.max_K + 1):
                ## REIMPLEMENT FUZZY KNN
                topk_ids = [ids[j] for j in sorted_scores[:k]]
                topk_scores = [scores[j] for j in sorted_scores[:k]]
                weights = softmax_weights(topk_scores, temperature=0.1)

                candidate_models = [is_mixed_dict[int(_id)] for _id in topk_ids]
                initial_pred = Counter(candidate_models).most_common(1)[0][0]

                fuzzy_cnt = {(1, 0): 0.0, (0, 10 ^ 3): 0.0, (1, 1): 0.0}
                for id, weight in zip(topk_ids, weights):
                    label = (label_dict[int(id)], is_mixed_dict[int(id)])
                    # boost = 1.3 if write_model_dict[int(id)] == test_write_model[i] else 1.0
                    boost = class_type_boost(is_mixed_dict[int(id)], initial_pred)
                    fuzzy_cnt[label] += weight * boost

                final = max(fuzzy_cnt, key=fuzzy_cnt.get)
                preds[k].append(final)

        K_values = list(range(1, opt.max_K + 1))
        accs = []
        precisions = []
        recalls = []
        f1_scores = []
        mses = []
        maes = []
        K_values = list(range(1, opt.max_K + 1))

        for k in range(1, opt.max_K + 1):
            acc, precision, recall, f1, mse, mae = compute_metrics(test_labels, preds[k], test_ids)
            print(
                f"K={k}, Acc:{acc:.5f}, Precision:{precision:.5f}, Recall:{recall:.5f}, F1:{f1:.5f}, MSE:{mse:.5f}, MAE:{mae:.5f}"
            )
            accs.append(acc)
            precisions.append(precision)
            recalls.append(recall)
            f1_scores.append(f1)
            mses.append(mse)
            maes.append(mae)

        fig, axs = plt.subplots(3, 3, figsize=(15, 15))

        axs[1, 0].plot(K_values, accs, marker="s", label="Accuracy")
        axs[1, 0].set_title("Accuracy")
        axs[1, 0].grid(True)

        axs[1, 1].plot(K_values, precisions, marker="p", label="Precision")
        axs[1, 1].set_title("Precision")
        axs[1, 1].grid(True)

        axs[1, 2].plot(K_values, recalls, marker="*", label="Recall")
        axs[1, 2].set_title("Recall")
        axs[1, 2].grid(True)

        axs[2, 0].plot(K_values, f1_scores, marker="D", label="F1 Score")
        axs[2, 0].set_title("F1 Score")
        axs[2, 0].grid(True)

        for i in range(2, 3):
            for j in range(1, 3):
                axs[i, j].axis("off")
        plt.tight_layout()
        plt.savefig("performance_metrics_subplot.png", dpi=300)
        max_ids = 0
        for i in range(1, opt.max_K):
            if accs[i] > accs[max_ids]:
                max_ids = i
        print(f"Find opt.max_K is {max_ids + 1}")
        print(
            f"Acc:{accs[max_ids]:.5f}, Precision:{precisions[max_ids]:.5f}, Recall:{recalls[max_ids]:.5f}, F1:{f1_scores[max_ids]:.5f}, MSE:{mse:.5f}, MAE:{mae:.5f}"
        )


def test_9_label(opt):
    if opt.device_num > 1:
        fabric = Fabric(accelerator="cuda", devices=opt.device_num, strategy="ddp")
    else:
        fabric = Fabric(accelerator="cuda", devices=opt.device_num)
    fabric.launch()
    model = TextEmbeddingModel(opt.model_name).cuda()
    state_dict = torch.load(opt.model_path, map_location=model.model.device)
    new_state_dict = {}
    for key in state_dict.keys():
        if key.startswith("model."):
            new_state_dict[key[6:]] = state_dict[key]
    model.load_state_dict(state_dict)
    tokenizer = model.tokenizer

    test_database = load_dataset(opt.dataset_name, opt.test_dataset_path)[opt.test_dataset_name]
    test_dataset = TextDataset(test_database, need_ids=True)

    test_dataloder = DataLoader(
        test_dataset, batch_size=opt.batch_size, num_workers=opt.num_workers, pin_memory=True
    )
    test_dataloder = fabric.setup_dataloaders(test_dataloder)
    model = fabric.setup(model)
    test_ids, test_embeddings, test_labels, test_is_mixed, test_write_model = infer(
        test_dataloder, fabric, tokenizer, model
    )
    fabric.barrier()

    if fabric.global_rank == 0:
        index = Indexer(opt.embedding_dim)
        index.deserialize_from(opt.database_path)

        label_dict = load_pkl(os.path.join(opt.database_path, "label_dict.pkl"))
        is_mixed_dict = load_pkl(os.path.join(opt.database_path, "is_mixed_dict.pkl"))
        write_model_dict = load_pkl(os.path.join(opt.database_path, "write_model_dict.pkl"))

        test_labels = [
            (test_labels[i], test_is_mixed[i], test_write_model[i]) for i in range(len(test_labels))
        ]

        preds = {i: [] for i in range(1, opt.max_K + 1)}
        if len(test_embeddings.shape) == 1:
            test_embeddings = test_embeddings.reshape(1, -1)
        top_ids_and_scores = index.search_knn(test_embeddings, opt.max_K)

        for i, (ids, scores) in enumerate(top_ids_and_scores):
            sorted_scores = np.argsort(scores)[::-1]

            for k in range(1, opt.max_K + 1):
                topk_ids = [ids[j] for j in sorted_scores[:k]]
                topk_scores = [scores[j] for j in sorted_scores[:k]]

                weights = softmax_weights(topk_scores, temperature=0.4)

                fuzzy_cnt = {
                    (1, 0, 0): 0.0,  # Human
                    (0, 10 ^ 3, 1): 0.0,
                    (0, 10 ^ 3, 2): 0.0,
                    (0, 10 ^ 3, 3): 0.0,
                    (0, 10 ^ 3, 4): 0.0,  # AI
                    (1, 1, 1): 0.0,
                    (1, 1, 2): 0.0,
                    (1, 1, 3): 0.0,
                    (1, 1, 4): 0.0,  # Human+AI
                }

                for id, weight in zip(topk_ids, weights):
                    label = (label_dict[int(id)], is_mixed_dict[int(id)], write_model_dict[int(id)])
                    boost = class_type_boost(test_write_model[i], write_model_dict[int(id)])
                    fuzzy_cnt[label] += weight * boost

                final = max(fuzzy_cnt, key=fuzzy_cnt.get)
                preds[k].append(final)

        K_values = list(range(1, opt.max_K + 1))
        accs = []
        precisions = []
        recalls = []
        f1_scores = []
        mses = []
        maes = []
        K_values = list(range(1, opt.max_K + 1))

        for k in range(1, opt.max_K + 1):
            acc, precision, recall, f1, mse, mae = compute_metrics(
                test_labels, preds[k], test_ids, full_labels=True
            )
            print(
                f"K={k}, Acc:{acc:.5f}, Precision:{precision:.5f}, Recall:{recall:.5f}, F1:{f1:.5f}, MSE:{mse:.5f}, MAE:{mae:.5f}"
            )
            accs.append(acc)
            precisions.append(precision)
            recalls.append(recall)
            f1_scores.append(f1)
            mses.append(mse)
            maes.append(mae)

        fig, axs = plt.subplots(3, 3, figsize=(15, 15))

        axs[1, 0].plot(K_values, accs, marker="s", label="Accuracy")
        axs[1, 0].set_title("Accuracy")
        axs[1, 0].grid(True)

        axs[1, 1].plot(K_values, precisions, marker="p", label="Precision")
        axs[1, 1].set_title("Precision")
        axs[1, 1].grid(True)

        axs[1, 2].plot(K_values, recalls, marker="*", label="Recall")
        axs[1, 2].set_title("Recall")
        axs[1, 2].grid(True)

        axs[2, 0].plot(K_values, f1_scores, marker="D", label="F1 Score")
        axs[2, 0].set_title("F1 Score")
        axs[2, 0].grid(True)

        for i in range(2, 3):
            for j in range(1, 3):
                axs[i, j].axis("off")
        plt.tight_layout()
        plt.savefig("performance_metrics_subplot.png", dpi=300)
        max_ids = 0
        for i in range(1, opt.max_K):
            if accs[i] > accs[max_ids]:
                max_ids = i
        print(f"Find opt.max_K is {max_ids + 1}")
        print(
            f"Acc:{accs[max_ids]:.5f}, Precision:{precisions[max_ids]:.5f}, Recall:{recalls[max_ids]:.5f}, F1:{f1_scores[max_ids]}:.5f, MSE:{mse:.5f}, MAE:{mae:.5f}"
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--device_num", type=int, default=1)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--embedding_dim", type=int, default=768)
    parser.add_argument(
        "--database_path", type=str, default="database", help="Path to the index file"
    )

    parser.add_argument(
        "--test_dataset_path", type=str, default="data/FAIDSetv2", help="Path to the data"
    )
    parser.add_argument(
        "--dataset_name",
        type=str,
        default="faidsetv2",
        help="faidset, llmdetectaive, hart, faidsetv2",
    )
    parser.add_argument(
        "--test_dataset_name", type=str, default="test", help="train,valid,test,test_ood"
    )
    parser.add_argument(
        "--out_domain_mode",
        type=str,
        default="",
        help="unseen_domain, unseen_generator, unseen_domain + unseen_generator",
    )

    parser.add_argument("--model_path", type=str, help="Path to the embedding model checkpoint")
    parser.add_argument(
        "--model_name",
        type=str,
        default="ZurichNLP/unsup-simcse-xlm-roberta-base",
        help="Model name",
    )
    parser.add_argument(
        "--full_labels", type=int, default=1, help="Conduct evaluation on full labels or not"
    )
    parser.add_argument(
        "--out_domain", type=int, default=0, help="Whether to use out domain dataset"
    )
    parser.add_argument(
        "--max_K", type=int, default=51, help="Search [1,K] nearest neighbors,choose the best K"
    )

    parser.add_argument(
        "--fuzzy_parameter", type=int, default=2, help="Values to make fuzzy decision"
    )
    parser.add_argument("--seed", type=int, default=0)
    opt = parser.parse_args()
    set_seed(opt.seed)
    if opt.full_labels:
        print("Test mode: Full labels")
        test_9_label(opt)
    else:
        print("Test mode: 3 labels")
        test_3_label(opt)
