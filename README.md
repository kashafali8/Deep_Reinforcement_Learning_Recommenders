# Deep Reinforcement Learning Recommender Systems
> #### Kashaf Ali, Pragya Raghuvanshi, Pooja Kabber | Spring '23 | AIPI 590 Take Home Challenge


## About the Project

**Task:** 

Train different session (contextual, sequential) based product recommendation recommenders for E-commerce use case with item and/or user features inside for cold items/users, and compare the performance of the recommenders. 


**Requirements:**

In the deliverables and experiments, one of the recommenders needs to be a Deep RL recommender [DRL2] and at least two different datasets are used for
training/testing. Also, at least two offline evaluation metrics are used for benchmarking.

**Our Approach:**

We selected two e-commerce datasets for our project: H&M Dataset and dataset2. We assessed a Deep Reinforcement Learning model on each of these datasets and compared performance using two recommender evaluation metrics: metric1 and metric2.


## Datasets

**H&M Dataset:**

The H&M Personalized Fashion Recommendations dataset, available on Kaggle, contains information about customer transactions on the H&M website between January 2017 and December 2017. H&M operates in 53 online markets and with approximately 4,850 stores, offering an extensive selection of products for its customers to browse through. This dataset contains a wide range of information about customer interactions, customer demographic information, product text descriptions data and also product image data, making it a valuable resource for developing personalized fashion recommendation algorithms. For the purpose of this project, we only used the transaction dataset (transaction_train.csv), which contains details about customer purchases on each data, product prices and other information about the sales channel. It is important to note that duplicate rows in this dataset refers to multiple purchases on the same item.

