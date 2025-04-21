# GDP-HMM_AAPMChallenge


The method used here is similar to the method in the paper 'Automated High Quality RT Planning at Scale'.
We use 'mednextv1_large' as our model. And we add 'Metric 1' to the loss function.


## Model file
The pre-trained model file is in huggingface. 
https://huggingface.co/datasets/justinzhdm/GDP-HMM_AAPM/tree/main/checkpoint/train.pth

We used the model ensemble method in inference phase. So the model file is formed by merging four checkpionts from a single training process.


## How to run

- train: python3 train.py
- test:  python3 inference.py
