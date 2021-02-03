# DST-STAR

This is the Pytorch implementation of our work: **Slot Self-Attentive Dialogue State Tracking. Fanghua Ye, Jarana Manotumruksa, Qiang Zhang, Shenghui Li, Emine Yilmaz. WWW 2021.** [[paper](https://arxiv.org/abs/2101.09374)]

## Abstract
An indispensable component in task-oriented dialogue systems is the dialogue state tracker, which keeps track of users' intentions in the course of conversation. The typical approach towards this goal is to fill in multiple pre-defined slots that are essential to complete the task. Although various dialogue state tracking methods have been proposed in recent years, most of them predict the value of each slot separately and fail to consider the correlations among slots. In this paper, we propose a slot self-attention mechanism that can learn the slot correlations automatically. Specifically, a slot-token attention is first utilized to obtain slot-specific features from the dialogue context. Then a stacked slot self-attention is applied on these features to learn the correlations among slots. We conduct comprehensive experiments on two multi-domain task-oriented dialogue datasets, including MultiWOZ 2.0 and MultiWOZ 2.1. The experimental results demonstrate that our approach achieves state-of-the-art performance on both datasets, verifying the necessity and effectiveness of taking slot correlations into consideration.

## Model 

<p align="center">
  <img src="models/STAR_framework.png" width="70%" />
</p>

<p align="center"> The model architecture of STAR</p>

## Usage
### Data Preprocessing

```console
❱❱❱ python3 create_data.py
❱❱❱ python3 preprocess_data.py
❱❱❱ python3 convert_by_turn.py
```
