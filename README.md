# Learning Attentive Pairwise Interaction for Fine-Grained Classification (API-Net)
Peiqin Zhuang, Yali Wang, Yu Qiao
# Introduction
In order to effectively identify contrastive clues among highly-confused categories, we propose a simple but effective Attentive Pairwise Interaction Network (API-Net), which can progressively recognize a pair of fine-grained images by interaction. We aim at learning a mutual vector first to capture semantic differences in the input pair and then comparing this mutual vector with individual vectors to highlight their semantic differences respectively. Besides, we also introduce a score-ranking regularization to promote the priorities of these features. For more details, please refer to [our paper](https://www.aaai.org/Papers/AAAI/2020GB/AAAI-ZhuangP.2505.pdf).
# Framework
![Framework](/Framework.png)
# Dependencies
* Python 2.7
* Pytorch 0.4.1
* torchvision 0.2.0
# Citing
Please kindly cite the following paper, if you find this code helpful in your work.
```
@article{zhuang2020learning,
  title={Learning Attentive Pairwise Interaction for Fine-Grained Classification},
  author={Zhuang, Peiqin and Wang, Yali and Qiao, Yu},
  journal={arXiv preprint arXiv:2002.10191},
  year={2020}
}
```
# Acknowledgement
Some of the codes are borrowed from [siamese-triplet](https://github.com/adambielski/siamese-triplet) and [triplet-reid-pytorch](https://github.com/CoinCheung/triplet-reid-pytorch). Many thanks to them.

