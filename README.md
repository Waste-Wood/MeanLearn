# MeanLearn
## 1. Structure

```
# folders
clustering: stores the clusters of OOD benchmarks
data: stores the training and test data of AbsR, as well as the OOD bechmark data
lora: stores the trained lora weights
output: stores the ppl-based inference results across different methods and benchmarks

# scripts
abstract_performance_absr.py: the script of computing AbsAcc on AbsR test set
abstract_performance_clustering.py: the script of computing AbsAcc on OOD benchmarks based on the clusters in "clustering" folder
performance_bbh.py: the script of computing vanilla accuracy on BBH benchmark across different methods
performance_classification.py: the script of computing vanilla accuracy on multiple-choice question-answering benchmarks
reason_ppl.py: the script of ppl-based evaluation
reason_ppl.sh: the shell script of ppl-based evaluation
tools.py: some supporting functions
train.py: the training code of MeanLearn
train.sh: the shell script to train MeanLearn
```



## 2. Training

You can edit the necessary arguments in train.sh to start training.

```shell
sh train.sh
```



## 3. Inference & Evaluation

We utilize ppl-based evaluation due to limited computing resources. In `reason_ppl.sh`, we provide an exmple of conducting inference on AGIEval based on MeanLearn and Orca-2.

```shell
sh reason_ppl.sh
```



After the inference stage, we can use the following scripts to obtain the evaluation results of vanilla accuracy and AbsAcc:

* obtaining vanilla accuracy on BBH

```shell
python performance_bbh.py --input_dir THE_PATH_TO_BBH_PREDICTION_FOLDER
```

* obtaining vanilla accuracy on AbsR, AGIEval, Comm., RACE, ARC, MMLU

```shell
python performance_classification.py --input_dir THE_PATH_TO_PREDICTION_FOLDER
```

* obtaining AbsAcc on AbsR

```shell
python abstract_performance_absr.py --input_dir THE_PATH_TO_PREDICTION_FOLDER
```

* obtaining AbsAcc on other benchmarks based in clustering results

```shell
python abstract_performance_clustering.py --input_dir THE_PATH_TO_PREDICTION_FOLDER
```



## 4. Q&A

* How to minimize the two different distributions mentioned in Equations 2 and 3?

We use two kinds of examples, one is <X, Y>, and the other is <X,r, Y>. The two kinds of examples are fed into LLMs to model Equations 2 and 3 autoregressively, we do not apply other tricks to minimize the two different distributions mentioned in Equations 2 and 3. Since LLMs can learn it well just by autoregressively training on both <X, Y> and <X,r, Y>, according to Occam's razor, we do not use a more complicate version.

Furthermore, if you want to achieve better results, you can consider the following methods:
(1) applying a KL loss for the two different distributions mentioned in Equations 2 and 3
(2) adopting conditional-VAE to achieve this imitation process (can refer to paper: Modeling event background for if-then commonsense reasoning using context-aware variational autoencoder)


