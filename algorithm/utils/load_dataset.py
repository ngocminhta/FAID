import hashlib
import json
import os
import random

from torch.utils.data import Dataset


def stable_long_hash(input_string):
    hash_object = hashlib.sha256(input_string.encode())
    hex_digest = hash_object.hexdigest()
    int_hash = int(hex_digest, 16)
    long_long_hash = int_hash & ((1 << 63) - 1)
    return long_long_hash


model_map_authscan = {
    "gpt-4o-mini-text": 1,
    "gemini-2.0-text": 2,
    "deepseek-text": 3,
    "llama-text": 4,
}

model_map_llmdetectaive = {"gemma-text": 1, "mixtral-text": 2, "llama3-text": 3}

model_map_hart = {"claude-text": 1, "gemini-text": 2, "gpt-text": 3}

model_map_faidsetv2 = {"gpt-text": 1, "gemini-text": 2, "deepseek-text": 3, "llama-text": 4}


def load_dataset(dataset_name, path=None):
    dataset = {"train": [], "valid": [], "test": []}
    if dataset_name == "falconset":
        model_map = model_map_authscan
    elif dataset_name == "llmdetectaive":
        model_map = model_map_llmdetectaive
    elif dataset_name == "hart":
        model_map = model_map_hart
    elif dataset_name == "faidsetv2":
        model_map = model_map_faidsetv2
    folder = os.listdir(path)
    # print(folder)
    for sub in folder:
        sub_path = os.path.join(path, sub)
        files = os.listdir(sub_path)
        for file in files:
            if not file.endswith(".jsonl"):
                continue
            file_path = os.path.join(sub_path, file)
            key_name = file.split(".")[0]

            assert key_name in dataset.keys(), f"{key_name} is not in dataset.keys()"
            with open(file_path, "r") as f:
                data = [json.loads(line) for line in f]
            for i in range(len(data)):
                dct = {}
                dct["text"] = data[i]["text"]
                if sub == "human-text":
                    dct["label"] = "human"
                    dct["label_detailed"] = "human"
                    dct["index"] = (1, 0, 0)
                elif sub.startswith("human---"):
                    dct["label"] = "human+AI"
                    model = sub.split("---")[1]
                    dct["label_detailed"] = model
                    dct["index"] = (1, 1, model_map[model])
                else:
                    dct["label"] = "AI"
                    dct["label_detailed"] = sub
                    dct["index"] = (0, 10 ^ 3, model_map[sub])
                dataset[key_name].append(dct)
    return dataset


def load_outdomain_dataset(path):
    dataset = {"valid": [], "test": []}
    folder = os.listdir(path)
    for sub in folder:
        sub_path = os.path.join(path, sub)
        files = os.listdir(sub_path)
        for file in files:
            if not file.endswith(".jsonl"):
                continue
            file_path = os.path.join(sub_path, file)
            key_name = file.split(".")[0]
            assert key_name in dataset.keys(), f"{key_name} is not in dataset.keys()"
            with open(file_path, "r", encoding="utf-8") as f:
                data = [json.loads(line) for line in f]
            for i in range(len(data)):
                dct = {}
                dct["text"] = data[i]["text"]
                if sub == "human-text":
                    dct["label"] = "human"
                    dct["label_detailed"] = "human"
                    dct["index"] = (1, 0)
                elif sub.startswith("human---"):
                    dct["label"] = "human+AI"
                    model = sub.split("---")[1]
                    dct["label_detailed"] = model
                    dct["index"] = (1, 1)
                else:
                    dct["label"] = "AI"
                    dct["label_detailed"] = sub
                    dct["index"] = (0, 10 ^ 3)
                dataset[key_name].append(dct)
    return dataset


def load_dataset_conditional_lang(path=None, language="vi", seed=42):
    dataset = {"train": [], "val": [], "test": []}
    combined_data = []

    random.seed(seed)  # for reproducibility
    folder = os.listdir(path)
    print("Subfolders:", folder)

    for sub in folder:
        sub_path = os.path.join(path, sub)
        if not os.path.isdir(sub_path):
            continue
        files = os.listdir(sub_path)

        for file in files:
            if not file.endswith(".jsonl") or language not in file:
                continue

            file_path = os.path.join(sub_path, file)
            with open(file_path, "r", encoding="utf-8") as f:
                data = [json.loads(line) for line in f]

            for entry in data:
                if "content" not in entry:
                    print("Key does not exist!")
                    continue

                dct = {}
                dct["text"] = entry["content"]

                if sub == "human":
                    dct["label"] = "human"
                    dct["label_detailed"] = "human"
                    dct["index"] = (1, 0, 0)
                elif sub == "human+AI":
                    model = entry["label_detailed"].split("+")[1]
                    dct["label"] = "human+AI"
                    dct["label_detailed"] = model
                    dct["index"] = (1, 1, model_map[model])
                else:
                    dct["label"] = "AI"
                    dct["label_detailed"] = entry["label_detailed"]
                    dct["index"] = (0, 10**3, model_map[entry["label_detailed"]])

                combined_data.append(dct)

    random.shuffle(combined_data)
    total = len(combined_data)
    train_end = int(total * 0.9)
    val_end = train_end + int(total * 0.05)

    dataset["train"] = combined_data[:train_end]
    dataset["val"] = combined_data[train_end:val_end]
    dataset["test"] = combined_data[val_end:]

    print(
        f"Total: {total} | Train: {len(dataset['train'])} | Val: {len(dataset['val'])} | Test: {len(dataset['test'])}"
    )
    return dataset


## only authscan
def load_monolingual_dataset(path=None, language="vi"):
    dataset = {"train": [], "val": [], "test": []}

    folder = os.listdir(path)
    print("Subfolders:", folder)

    for sub in folder:
        sub_path = os.path.join(path, sub)
        if not os.path.isdir(sub_path):
            continue
        files = os.listdir(sub_path)

        for file in files:
            if not file.endswith(".jsonl") or language not in file:
                continue

            file_path = os.path.join(sub_path, file)
            with open(file_path, "r", encoding="utf-8") as f:
                data = [json.loads(line) for line in f]

            for entry in data:
                if "text" not in entry:
                    print("Key does not exist!")
                    continue

                dct = {}
                dct["text"] = entry["text"]

                if sub == "human-text":
                    dct["label"] = "human"
                    dct["label_detailed"] = "human"
                    dct["index"] = (1, 0, 0)
                elif sub.startswith("human---"):
                    dct["label"] = "human+AI"
                    model = sub.split("---")[1]
                    dct["label_detailed"] = model
                    dct["index"] = (1, 1, model_map_authscan[model])
                else:
                    dct["label"] = "AI"
                    dct["label_detailed"] = sub
                    dct["index"] = (0, 10 ^ 3, model_map_authscan[sub])

                dataset["test"].append(dct)

    print(
        f"Train: {len(dataset['train'])} | Val: {len(dataset['val'])} | Test: {len(dataset['test'])}"
    )
    return dataset


def load_per_class_dataset(path=None, label=None):
    """class in [human, AI, hybrid]"""
    dataset = {"train": [], "val": [], "test": []}

    folder = os.listdir(path)
    print("Subfolders:", folder)

    if label == "human":
        subs = ["human-text"]
    elif label == "AI":
        subs = ["deepseek-text", "gemini-2.0-text", "llama-text", "gpt-4o-mini-text"]
    else:
        subs = [f for f in folder if f.startswith("human---")]

    for sub in subs:
        sub_path = os.path.join(path, sub)
        if not os.path.isdir(sub_path):
            continue
        files = os.listdir(sub_path)

        for file in files:
            if not file.endswith(".jsonl"):
                continue

            file_path = os.path.join(sub_path, file)
            with open(file_path, "r", encoding="utf-8") as f:
                data = [json.loads(line) for line in f]

            for entry in data:
                if "text" not in entry:
                    print("Key does not exist!")
                    continue

                dct = {}
                dct["text"] = entry["text"]
                dct["label"] = label

                if label == "human":
                    dct["label_detailed"] = "human"
                    dct["index"] = (1, 0, 0)
                elif label == "AI":
                    dct["label_detailed"] = sub
                    dct["index"] = (0, 10 ^ 3, model_map_authscan[sub])
                else:
                    dct["label_detailed"] = sub.split("---")[1]
                    dct["index"] = (1, 1, model_map_authscan[dct["label_detailed"]])

                dataset["test"].append(dct)

    print(
        f"Train: {len(dataset['train'])} | Val: {len(dataset['val'])} | Test: {len(dataset['test'])}"
    )
    return dataset


class TextDataset(Dataset):
    def __init__(self, dataset, need_ids=True, out_domain=0):
        self.dataset = dataset
        self.need_ids = need_ids
        self.out_domain = out_domain

    def get_class(self):
        return self.classes

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        text, label, label_detailed, index = self.dataset[idx].values()
        id = stable_long_hash(text)
        if self.out_domain:
            label, is_mixed = index
            if self.need_ids:
                return int(id), text, int(label), int(is_mixed)
            return text, int(label), int(is_mixed)
        else:
            label, is_mixed, write_model = index
            if self.need_ids:
                return int(id), text, int(label), int(is_mixed), int(write_model)
            return text, int(label), int(is_mixed), int(write_model)
