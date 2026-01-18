<div align="left" style="position: relative;">
<!-- <img src="https://img.icons8.com/?size=512&id=55494&format=png" align="right" width="30%" style="margin: -20px 0 0 20px;"> -->
<h1>FAID</h1>
<p align="left">
	<em>Fine-grained AI-generated Text Detection using Multi-task Auxiliary and Multi-level Contrastive Learning.</em>
</p>
<p align="left">
	<img src="https://img.shields.io/github/license/ngocminhta/FAID?style=flat-square&logo=opensourceinitiative&logoColor=white&color=00a3b9" alt="license">
	<img src="https://img.shields.io/github/last-commit/ngocminhta/FAID?style=flat-square&logo=git&logoColor=white&color=00a3b9" alt="last-commit">
	<img src="https://img.shields.io/github/languages/top/ngocminhta/FAID?style=flat-square&color=00a3b9" alt="repo-top-language">
	<img src="https://img.shields.io/github/languages/count/ngocminhta/FAID?style=flat-square&color=00a3b9" alt="repo-language-count">
</p>
<p align="left">Built with the tools and technologies:</p>
<p align="left">
	<img src="https://img.shields.io/badge/GNU%20Bash-4EAA25.svg?style=flat-square&logo=GNU-Bash&logoColor=white" alt="GNU%20Bash">
	<img src="https://img.shields.io/badge/Python-3776AB.svg?style=flat-square&logo=Python&logoColor=white" alt="Python">
</p>
</div>
<br clear="right">

> The dataset of this project, FAIDSet is explicitly available at HuggingFace dataset [ngocminhta/FAIDSet](https://huggingface.co/datasets/ngocminhta/FAIDSet)

## 🔗 Table of Contents

- [🔗 Table of Contents](#-table-of-contents)
- [📍 Overview](#-overview)
- [📁 Project Structure](#-project-structure)
  - [📂 Project Index](#-project-index)
- [🚀 Getting Started](#-getting-started)
  - [☑️ Prerequisites](#️-prerequisites)
  - [⚙️ Installation](#️-installation)
  - [🤖 Usage](#-usage)
  - [🧪 Testing](#-testing)
- [📌 News](#-news)
- [🎗 License](#-license)
- [🙌 Acknowledgments](#-acknowledgments)

---

## 📍 Overview

The FAID project revolutionizes the detection of deepfake content through advanced text analysis. By leveraging state-of-the-art machine learning techniques, it offers robust tools for generating, managing, and evaluating text embeddings to accurately classify content as human, AI-generated, or mixed. Ideal for tech companies and cybersecurity experts, FAID enhances digital trust and integrity across various media platforms.

---

## 📁 Project Structure

```sh
└── FAID/
    ├── README.md
    ├── algorithm
    │   ├── gen_database.py
    │   ├── infer.py
    │   ├── requirements.txt
    │   ├── src
    │   │   ├── index.py
    │   │   ├── simclr.py
    │   │   └── text_embedding.py
    │   ├── test_from_database.py
    │   ├── train_classifier.py
    │   └── utils
    │       ├── load_dataset.py
    │       └── utils.py
    └── data
        ├── FAIDSet
        ├── Unseen_Domain
        ├── Unseen_Domain_and_Unseen_Generator
        └── Unseen_Generator
```
---
## 🚀 Getting Started

### ☑️ Prerequisites

Before getting started with FAID, ensure your runtime environment meets the following requirements:

- **Programming Language:** Python
- **Package Manager:** Pip


### ⚙️ Installation

Install FAID using one of the following methods:

**Build from source:**

1. Clone the FAID repository:
```sh
❯ git clone https://github.com/ngocminhta/FAID
```

2. Navigate to the project directory:
```sh
❯ cd FAID
```

3. Install the project dependencies:


**Using `pip`** &nbsp; [<img align="center" src="https://img.shields.io/badge/Pip-3776AB.svg?style={badge_style}&logo=pypi&logoColor=white" />](https://pypi.org/project/pip/)

```sh
❯ pip install -r algorithm/requirements.txt
```




### 🤖 Usage
Run FAID using the following command:
**Using `pip`** &nbsp; [<img align="center" src="https://img.shields.io/badge/Pip-3776AB.svg?style={badge_style}&logo=pypi&logoColor=white" />](https://pypi.org/project/pip/)

To train the model
```sh
❯ python algorithm/train_classifier.py <your parameter goes here>
```
To generate the vector database after training
```sh
❯ python algorithm/gen_database.py <your parameter goes here>
```

### 🧪 Testing
Run the test suite using the following command:
**Using `pip`** &nbsp; [<img align="center" src="https://img.shields.io/badge/Pip-3776AB.svg?style={badge_style}&logo=pypi&logoColor=white" />](https://pypi.org/project/pip/)

```sh
❯ python algorithm/test_from_database.py <your parameter goes here>
```


---
## 📌 News

**[2026.01.04]** Our research paper is accepted to EACL 2026 Main Conference!

**[2025.05.20]** Our research paper now publicly accessible on arXiv.

**[2025.05.06]** Our project is publicly accessible.

---

## 🎗 License

This project is protected under the [MIT](LICENSE.md) License.

---

## 🙌 Acknowledgments

This research is carried on at:

- BKAI Research Center, Hanoi University of Science and Technology.
- Natural Language Processing Department, Mohamed bin Zayed University of Artificial Intelligence.

---

## 🔬 Citation

```
@misc{ta2025faidfinegrainedaigeneratedtext,
      title={FAID: Fine-grained AI-generated Text Detection using Multi-task Auxiliary and Multi-level Contrastive Learning}, 
      author={Minh Ngoc Ta and Dong Cao Van and Duc-Anh Hoang and Minh Le-Anh and Truong Nguyen and My Anh Tran Nguyen and Yuxia Wang and Preslav Nakov and Sang Dinh},
      year={2025},
      eprint={2505.14271},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/2505.14271}, 
}
```
