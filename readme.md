# Pytorch Version QA4IE Code (Journal Version)
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
```