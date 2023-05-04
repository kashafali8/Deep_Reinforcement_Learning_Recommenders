# Deep Reinforcement Learning Recommender Systems
> #### Kashaf Ali, Pragya Raghuvanshi, Pooja Kabber | Spring '23 | AIPI 590 Take Home Challenge


## About the Project

**Motivation:**

The motivation for this project stems from the importance of personalized product recommendations in E-commerce. With the increasing amount of data generated by online shopping platforms, it has become more challenging to provide relevant recommendations to users, especially for cold items or new users with little purchase history. Inaccurate recommendations may result in a poor user experience and reduced engagement with the platform. Therefore, developing effective and efficient product recommendation strategies that can adapt to different types of user sessions and incorporate item and user features is critical. Additionally, the use of Deep RL recommenders in this project will allow for the exploration of more sophisticated recommendation techniques and potentially improve the accuracy of the recommendations. By comparing the performance of different recommenders, this project aims to provide insights into which approaches are most effective and efficient for personalized product recommendations in E-commerce. Ultimately, the project's outcome could help improve user engagement and satisfaction, increase sales, and benefit online shopping platforms and consumers alike.

**Project Overview:** 

This project aims to implement different session (contextual, sequential) based product recommendation recommenders for the E-commerce use case, and will incorporate item features for cold items. The goal would be to compare the performance of these recommenders on two different dataset, namely Retail Rocket and Diginetica, through two offline evaluation metrics: NDCG@k and HR@k. We will be comparing the performance of a SASRec model to a hRNN (hierarchical RNN) model that also accounts for item features. In this project, we aim to develop effective and efficient product recommendation strategies for E-commerce that can adapt to different types of user sessions and leverage available features for improved performance.

## Datasets

**Retail Rocket:**

**Diginetica Dataset:**

Diginetica is a company that specializes in developing personalized recommendation systems for E-commerce. They have provided a dataset, called the Diginetica Challenge dataset, which is available on the CodaLab platform. The dataset includes anonymized user interactions with items on an E-commerce website. It consists of two parts: a training set with over 4 million user interactions and a test set with around 1 million interactions. The interactions include views, clicks, and purchases, and cover a variety of item types, including books, movies, and music. In addition to the interaction data, the dataset also includes metadata on each item, such as titles, categories, and descriptions. The goal of the Diginetica Challenge is to develop a personalized recommendation system that can accurately predict user interactions on the test set based on the training data. The challenge provides an opportunity for researchers and data scientists to explore and develop state-of-the-art recommendation algorithms for E-commerce applications. For the purpose of our project, we primarily leverage the view and purchase logs, and also the product_category dataset to build item features.

> Download [here](https://competitions.codalab.org/competitions/11161)

## Methodology

We have created Google Colab Notebooks for running the model both with and without item features for each datasets (Retail Rocket and Diginetica). They contain all the code and corresponding descriptions to reproduce the results.

All the required datasets have been stored in the Google Drive (linked here), and the Google Colab notebooks automatically clones our GitHub repository and conducts preprocessing on the datasets before executing modeling and evaluation scripts.

**Data Preprocessing:**

We conducted the following data preprocessing for both our datasets prior to running them through the model:

1. Retail Rocket:

2. Diginetica: we used the train-item-views.csv for views and train-purchases.csv datasets for purchases for our model, and product_cat dataset for product category information for creating our one-hot-encodings for item features. Our SNQN model also needs data on user interactions such as clicks or views, as well as purchases or adding items to the cart. Although both types of interactions are positive, we consider clicks as negative signals and purchases as positive signals, and only positive signals are used to affect the user state in the model. We also used publicly available code by Xin Xin et al. (including the replay_buffer.py and pop.py files for preprocessing the data). We also preprocessed our datasets by assigning is_buy flags to indicate whether a user had made a purchase, and we excluded users with fewer than three interactions from the analysis.

**Item Feature Selection & One Hot Encoding:**

To prepare the Retail Rocket and Diginetica datasets for analysis, we performed feature selection to identify the most relevant properties, and then one-hot-encoded all the features for each item. Specifically, we decided to focus on the 500 most frequent properties because they provided sufficient coverage for our analysis without compromising computational efficiency. We included a script for creating the item features matrix in the item_features_K.py file.

**Steps to run the code:**

Jupyter notebook at in the DRL_Recommenders directory contains code for both datasets. Below are the steps to reproduce the code:

1. Launch `Driver Notebook` in a Google Colab instance with GPU and mount the relevant google drive directory
2. Install and load the required python libraries
3. Clone the git repository containing all source code and files for running the model.
4. Run the preprocessing files to generate a replay buffer from the source code: replay_buffer.py and pop.py
5. Run the final cell to being model training and evaluation (both with and without item features)

## Evaluation Metrics & Results

We used two evaluation metrics: Normalized Discounted Cumulative Gain (NDCG@k) and Hit Ratio (HR@k). To ensure consistency in the comparison of results across different DRL models, we report the performance at the highest number of training epochs that we were able to complete, which was approximately 15 epochs.

1. NDCG@k is a metric used to evaluate the effectiveness of a recommendation list based on how the top-k items in the list are ranked, with higher ranked items receiving higher scores.
2. HR@k is a metric that determines if a recommended item is in the top-k positions of the list produced by the model, as compared to the ground-truth item.

**1. Results for Retail Rocket Dataset**


**2. Results for Diginetica Dataset**
|SNQN_SAS-Rec Model Results without Item Features|SNQN_SAS-Rec Model Results with Item Features|
|--|--|
|<table><tr><th>NDCG@10</th><th>HR@10</th></tr><tr><td>0.0198</td><td>0.0377</td></tr></table>|<table><tr><th>NDCG@10</th><th>HR@10</th><th>MRR@10</th><th>MAP@10</th></tr><tr><td>0.0014</td><td>0.0034</td><td>0.0026</td><td>0.0026</td></tr></table>|





