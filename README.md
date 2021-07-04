## Deep Recurrent Model for Individualized Prediction of Alzheimer’s Disease Progression
<p align="center"><img width="80%" src="Files/Multi_final_cog.png" /></p>

This repository contains code to train __Deep Recurrent AD__.

## Usage
For training:

`python main.py --dataset='Zero' --data_path=PATH --kfold=5 --impute_weigh=0.1 --reg_weight=0.5 --label_weight=0.5 --gamma=5.0 --cognitive_score=True`
