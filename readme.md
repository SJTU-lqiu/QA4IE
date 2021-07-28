# Pytorch Version QA4IE Code (Journal Version)
Here is the original implementation for the following series of publications on QA4IE.

- ISWC 2018: [QA4IE: A Question Answering based Framework for Information Extraction](https://link.springer.com/chapter/10.1007/978-3-030-00671-6_12)
- IEEE Access 2020: [QA4IE: A Question Answering based System for Document-Level General Information Extraction](https://ieeexplore.ieee.org/abstract/document/8972460)
- SIGIR DEMO 2020: [QuAChIE: Question Answering based Chinese Information Extraction System](https://dl.acm.org/doi/abs/10.1145/3397271.3401411)

This branch mantains the code for the journal version QA4IE. The ISWC conference version is maintained in [iswc](https://github.com/SJTU-lqiu/QA4IE/tree/iswc) branch. The chinese benchmark and the corresponding code will also be released soon.

## Requirements
- torch = 1.7.1
- wandb
## Pretrained Models

Download preprocessed data and pretrained models [here](https://drive.google.com/drive/folders/18_d2ASDWopcseEmrNaAcqaegX8ErJqlH?usp=sharing).

Uncompress them in ```./data``` and ```./out```, respectively.

## Training & Evaluation Scripts
Training and evaluation scripts are provided in ```./scripts```. Note that you need to execute the ```dump_SS.sh``` script before training QA module, and execute the ```dump_QA.sh``` before training the AT module.

## IE-setting Evaluation
Evaluate in IE-setting with different types of scorer:
- mean: average probability of answer sequence
- prod: product of answer sequence probabilities
- AT: use the output of the AT module as the score

```python3 eval_ie.py --scorer <mean|prod|AT>```

## Issues on Experimental Results
- Note that the results obtained by running the code in this repo will be slightly better than the results reported in the paper. The main reasons are the usage of a more proper optimizer, a larger batch size, and a learning rate scheduler in the new implementation.

## Cite Us

```
@inproceedings{qiu2018qa4ie,
  title={QA4IE: A question answering based framework for information extraction},
  author={Qiu, Lin and Zhou, Hao and Qu, Yanru and Zhang, Weinan and Li, Suoheng and Rong, Shu and Ru, Dongyu and Qian, Lihua and Tu, Kewei and Yu, Yong},
  booktitle={International Semantic Web Conference},
  pages={198--216},
  year={2018},
  organization={Springer}
}

@article{qiu2020qa4ie,
  title={Qa4ie: A question answering based system for document-level general information extraction},
  author={Qiu, Lin and Ru, Dongyu and Long, Quanyu and Zhang, Weinan and Yu, Yong},
  journal={IEEE Access},
  volume={8},
  pages={29677--29689},
  year={2020},
  publisher={IEEE}
}

@inproceedings{ru2020quachie,
  title={QuAChIE: Question Answering based Chinese Information Extraction System},
  author={Ru, Dongyu and Wang, Zhenghui and Qiu, Lin and Zhou, Hao and Li, Lei and Zhang, Weinan and Yu, Yong},
  booktitle={Proceedings of the 43rd International ACM SIGIR Conference on Research and Development in Information Retrieval},
  pages={2177--2180},
  year={2020}
}
```