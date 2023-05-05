# Deep Reinforcement Learning Recommender Systems
> #### Kashaf Ali, Pragya Raghuvanshi, Pooja Kabber | Spring '23 | AIPI 590 Take Home Challenge


## About the Project

**Motivation:**

The motivation for this project stems from the importance of personalized product recommendations in E-commerce. With the increasing amount of data generated by online shopping platforms, it has become more challenging to provide relevant recommendations to users, especially for cold items or new users with little purchase history. Inaccurate recommendations may result in a poor user experience and reduced engagement with the platform. Therefore, developing effective and efficient product recommendation strategies that can adapt to different types of user sessions and incorporate item and user features is critical. Additionally, the use of Deep RL recommenders in this project will allow for the exploration of more sophisticated recommendation techniques and potentially improve the accuracy of the recommendations. By comparing the performance of different recommenders, this project aims to provide insights into which approaches are most effective and efficient for personalized product recommendations in E-commerce. Ultimately, the project's outcome could help improve user engagement and satisfaction, increase sales, and benefit online shopping platforms and consumers alike.

**Project Overview:** 

This project aims to implement different session (contextual, sequential) based product recommendation recommenders for the E-commerce use case, and will incorporate item features for cold items. The goal would be to compare the performance of these recommenders on two different dataset, namely Retail Rocket and Diginetica, through two offline evaluation metrics: NDCG@k and HR@k. We will be comparing the performance of a vanilla SNQN-SASRec model (without item features) with a SNQN-SASRec model that also accounts for item features. In this project, we aim to develop effective and efficient product recommendation strategies for E-commerce that can adapt to different types of user sessions and leverage available features for improved performance.

## Datasets

**1. Retail Rocket:**

Retail Rocket is an organization that creates customized product suggestions for online shopping websites and performs customer segmentation based on various factors, including user interests. The dataset was obtained from an actual e-commerce website and consisted of unprocessed data, meaning that it was not altered in any way. However, all data points were hashed to protect confidentiality. In this project, only the behavior data file (`events.csv`) was used, which contains a timestamped log of various events, such as clicks, add-to-cart actions, and transactions. These events represent different interactions made by visitors to the e-commerce website over a 4.5-month period, totaling 2,756,101 events produced by 1,407,580 distinct visitors. Additionally, the `item_features_x.csv` dataset was used to extract features of each item, which includes information about their properties and respective values.

Downlooad [here](https://www.kaggle.com/datasets/retailrocket/ecommerce-dataset)

**2. Diginetica Dataset:**

Diginetica is a company that specializes in developing personalized recommendation systems for E-commerce. They have provided a dataset, called the Diginetica Challenge dataset, which is available on the CodaLab platform. The dataset includes anonymized user interactions with items on an E-commerce website. It consists of two parts: a training set with over 4 million user interactions and a test set with around 1 million interactions. The interactions include views, clicks, and purchases, and cover a variety of item types, including books, movies, and music. In addition to the interaction data, the dataset also includes metadata on each item, such as titles, categories, and descriptions. The goal of the Diginetica Challenge is to develop a personalized recommendation system that can accurately predict user interactions on the test set based on the training data. The challenge provides an opportunity for researchers and data scientists to explore and develop state-of-the-art recommendation algorithms for E-commerce applications. For the purpose of our project, we primarily leverage the view and purchase logs, and also the product_category dataset to build item features.

> Download [here](https://competitions.codalab.org/competitions/11161)

## Methodology

We have created Google Colab Notebooks for running the model both with and without item features for each dataset (Retail Rocket and Diginetica). They contain all the code and corresponding descriptions to reproduce the results.

All the required datasets have been stored in the Google Drive (linked [here](https://drive.google.com/drive/folders/1EOWOzpnCGcaXGhO2oe7NUUMK2Q2Wji1a?usp=sharing)), and the Google Colab notebooks automatically clones our GitHub repository and conducts preprocessing on the datasets before executing modeling and evaluation scripts.

**Data Preprocessing:**

We conducted the following data preprocessing for both our datasets prior to running them through the model:

1. Retail Rocket: The preprocessing steps include removing transactions, removing users and items with few interactions, and transforming item IDs, session IDs, and behavior types to integers using LabelEncoder. The training, validation, and test sets are split using a 80-10-10 split. The item popularity is calculated and stored in a dictionary, and the replay buffer is generated from the training set by padding or trimming a user's interaction history to a maximum length.

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

We used the SNQN_SASRec Model both with and without item features. The SAS-Rec algorithm allows the model to capture both sequential and user-item features. Overall, the SNQN_SAS-Rec model has been shown to outperform other state-of-the-art recommendation models in terms of recommendation accuracy and computational efficiency.

We used two evaluation metrics: Normalized Discounted Cumulative Gain (NDCG@k) and Hit Ratio (HR@k). To ensure consistency in the comparison of results across different DRL models, we report the performance at the highest number of training epochs that we were able to complete, which was approximately 15 epochs.

1. NDCG@k is a metric used to evaluate the effectiveness of a recommendation list based on how the top-k items in the list are ranked, with higher ranked items receiving higher scores.
2. HR@k is a metric that determines if a recommended item is in the top-k positions of the list produced by the model, as compared to the ground-truth item.

**1. Results for Retail Rocket Dataset**

**Purchases**

| Model                             | HR@5     | NDCG@5   | HR@10    | NDCG@10  | HR@15    | NDCG@15 | HR@20   | NDCG@20   |
|-----------------------------------|----------|----------|----------|----------|----------|---------|---------|------------|
| SNQN-SASRec without item Features | 0.003586 | 0.002537 | 0.006275 | 0.003390 |0.007709| 0.003763 | 0.009143 |0.004102| 
| SNQN-SASRec with item Features    |0.005020 | 0.003653| 0.008247 | 0.004669| 0.009860 | 0.005093| 0.010577 | 0.005263 |

**Clicks**

| Model                             | HR@5     | NDCG@5   | HR@10    | NDCG@10  | HR@15    | NDCG@15 | HR@20   | NDCG@20   |
|-----------------------------------|----------|----------|----------|----------|----------|---------|---------|------------|
| SNQN-SASRec without item Features |0.001068 | 0.000695|   0.001752 | 0.000918 |0.002384| 0.001085 | 0.002820 | 0.001187| 
| SNQN-SASRec with item Features    | 0.001376 |0.001023 |0.002119| 0.001262 |  0.002743 |0.001428 |  0.003299| 0.001560 | 


From the above results we can see that there is approximately 25% increase in the Purchases for HR and approximately 30% increase in the NDCG mertics when we considered item features as compared to when we don't consider item features. We can also see that for clicks too, there is an approximate increase of 20% for HR and 30 %for NDCG, after taking item features in consideration.


**2. Results for Diginetica Dataset**

**Purchases**

| Model                             | HR@5     | NDCG@5   | HR@10    | NDCG@10  | HR@15    | NDCG@15 | HR@20   | NDCG@20   |
|-----------------------------------|----------|----------|----------|----------|----------|---------|---------|------------|
| SNQN-SASRec without item Features | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.001044 | 0.000246 | 
| SNQN-SASRec with item Features    | 0.120042 | 0.081328 | 0.160752 | 0.094191 | 0.191023 | 0.102181 | 0.208768 | 0.106387 |

**Clicks**

| Model                             | HR@5     | NDCG@5   | HR@10    | NDCG@10  | HR@15    | NDCG@15 | HR@20   | NDCG@20   |
|-----------------------------------|----------|----------|----------|----------|----------|---------|---------|------------|
| SNQN-SASRec without item Features | 0.000108 | 0.000068 | 0.000378 | 0.000153 | 0.000649 | 0.000226 | 0.000784 | 0.000258 | 
| SNQN-SASRec with item Features    | 0.113090 | 0.081594 | 0.160067 | 0.096746 | 0.192205 | 0.105245 | 0.217099 | 0.111118 |



**3. Results for Retail Rocket Dataset for Non-RL Model**

**Purchases**

| Model                             | HR@5     | NDCG@5   | HR@10    | NDCG@10  | HR@15    | NDCG@15 | HR@20   | NDCG@20   |
|-----------------------------------|----------|----------|----------|----------|----------|---------|---------|------------|
| Non-RL without item Features | 0.000000 | 0.000000 | 0.000000 | 0.000000 |0.000000| 00.000000 |0.000000 |0.0000002| 
| Non-RL with item Features    |0.000000 | 0.000000| 0.000000 | 0.000000| 0.000179 | 0.000050|0.000179 | 0.000050 |

**Clicks**

| Model                             | HR@5     | NDCG@5   | HR@10    | NDCG@10  | HR@15    | NDCG@15 | HR@20   | NDCG@20   |
|-----------------------------------|----------|----------|----------|----------|----------|---------|---------|------------|
| Non-RL without item Features |0.000051 |0.000033|   0.000103 | 0.000050 |0.000137| 0.000059 | 0.000205 | 0.000075| 
| Non-RL with item Features    | 0.000017 | 0.000007 |0.000077| 0.000025 |  0.000111 |0.000034 |  0.000188| 0.000052 | 

When comparing the non RL with and without item features we can see an increase in the metrics for purchases but decrease in clicks after including item features. However , if we compare the metrics for  Non-RL with item Features  and RL with item features we can see that there is a significant increase in the metrics of about 50%.

## Limitation and Future Scope 

1. Exploring different item features or exploring other ways of incorporating item features into the model to make it more informative to improve the quality of the recommendations. 

2. Performing hyperparamneter tuning on different mdoel architecture to see if it improves the performance.

3. Incorporating other metrics like RRK, MRP and MAP can help us betetr evaluate the performance of the recommender system.



## Folder structure

```
Deep_Reinforcement_Learning_Recommenders
├─ DRL_Recommenders
│  ├─ Dataset_1_Retail_Rocket
│  │  ├─ RR_SA2C_Recommender.ipynb
│  │  └─ src
│  │     ├─ NextItNetModules_v2.py
│  │     ├─ SA2C_v2.py
│  │     ├─ SASRecModules_v2.py
│  │     ├─ gen_replay_buffer.py
│  │     └─ utility_v2.py
│  ├─ Dataset2_Diginetica
│  │  ├─ Diginetica_SNQN.ipynb
│  │  ├─ README.md
│  │  └─ src
│  │     ├─ NextItNetModules.py
│  │     ├─ SASRecModules.py
│  │     ├─ utility.py
│  │     ├─ pop_v1.py
│  │     ├─ replay_buffer_v1.py
│  │     ├─ preprocessing_splits.py
│  │     ├─ SNQN_V.py
│  │     ├─ SNQN_P.py
│  │     ├─ item_features_files.ipynb
│  │     ├─item_features_K.py 
│  │     ├─README.md 
│  └─ Source_Code_Implementation.ipynb
├─ README.md
```

## Contributions:
| Name                             | Reference    | Contribution   |
|-----------------------------------|----------|----------|
| Kashaf Ali| [Github Profile](https://github.com/kashafali8) | Diginetica Dataset Preprocessing, Executing SNQN-SASRec Models (with and without item features) and Source Code File Changes (SNQN model and item features)|
| Pragya Raghuvanshi| [Github Profile](https://github.com/pr-124) |  Preprocessing of RetailRocket Data,  Executing SNQN-SASRec Models (with and without item features) for RR data, Executing Non-RL Models (with and without item features) for RR data|
| Pooja Kabber | [Github Profile](https://github.com/poojakabber7)| Source Code Modification (SNQN model and item features) & Executing SNQN-SASRec Models (with and without item features) for RR data, Executing Non-RL Models (with and without item features) for RR data|

## References

1. Supervised Advantage Actor-Critic for Recommender Systems | X. Xin, A. Karatzoglou, I. Arapakis, and J. M. Jose, Proceedings of ACM Conference (Conference’17), 2021.

2. Xin, Xin, et al. "Self-supervised reinforcement learning for recommender systems." Proceedings of the 43rd International ACM SIGIR Conference on Research and Development in Information Retrieval. 2020.

