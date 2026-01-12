# End-to-end-chest-cancer-problem

Dataset source: https://www.kaggle.com/datasets/mohamedhanyyy/chest-ctscan-images/code?datasetId=839140&sortBy=dateRun&tab=profile&excludeNonAccessedDatasources=false

**Although the project is still in development, the core deep-learning model is already fully implemented and can be accessed in this [repository](test_accuracy_95.ipynb)**


**State-of-the-Art Performance**

After extensive 2-phase training and optimization, the model achieved **94.6%** test accuracy. This is the highest known accuracy for this Chest Cancer dataset — outperforming all publicly available implementations to date.

## Core Deep-Learning Pipeline (Completed) ##
The following major components of the deep-learning system are fully finished:<br>
-Transfer Learning using VGG16<br>
-Two-phase training (head training → fine-tuning)<br>
-Custom classifier head architecture<br>
-High-performance modular training pipeline<br>
-Evaluation notebook with detailed analysis<br>

## Upcoming Features (In Development) ##
The broader ML system and deployment stack are actively being built:<br>
-Web Application Interface<br>
-Full Deployment Pipeline on AWS<br>
-Automated CI/CD deployment workflow<br>
