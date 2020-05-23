# Learning Attentive Pairwise Interaction for Fine-Grained Classification (API-Net)
Peiqin Zhuang, Yali Wang, Yu Qiao
# Introduction
This repo contains the source code of API-Net. In order to effectively identify contrastive clues among highly-confused categories, we propose a simple but effective Attentive Pairwise Interaction Network (API-Net), which can progressively recognize a pair of fine-grained images by interaction. We aim at learning a mutual vector first to capture semantic differences in the input pair and then comparing this mutual vector with individual vectors to highlight their semantic differences respectively. Besides, we also introduce a score-ranking regularization to priorities of these features. For more details, please refer to [our paper](https://www.aaai.org/Papers/AAAI/2020GB/AAAI-ZhuangP.2505.pdf).
# Framework
![Framework](https://github.com/PeiqinZhuang/API-Net/blob/master/Framework.png)

