import pandas as pd
import numpy as np
import os
from sklearn.preprocessing import LabelEncoder
from utility import to_pickled_df


print(os.getcwd())

data_dir = "/Users/kashafali/Documents/Duke/Spring23/RL/Deep_Reinforcement_Learning_Recommenders/DRL_Recommenders/Dataset2_Diginetica/Data"
# read in data
views = pd.read_csv(os.path.join(data_dir, "train-item-views.csv"), sep=";")
purchases = pd.read_csv(os.path.join(data_dir, "train-purchases.csv"), sep=";")

# use users instead of sessions as there is more overlap between the two datasets
# keep required columns
cols = ["userId", "itemId", "eventdate"]
views = views[cols]
purchases = purchases[cols]

# drop nas and change to int
views = views.dropna().astype({"userId": int})
purchases = purchases.dropna().astype({"userId": int})

# change eventdate to sortable time
views["eventdate"] = pd.to_datetime(views["eventdate"])
purchases["eventdate"] = pd.to_datetime(purchases["eventdate"])

# rename columns
rename_cols = {"userId": "session_id", "itemId": "item_id", "eventdate": "timestamp"}
views = views.rename(columns=rename_cols)
purchases = purchases.rename(columns=rename_cols)


# outer merge with indicator
events = pd.merge(
    views,
    purchases,
    how="outer",
    on=["session_id", "item_id", "timestamp"],
    indicator=True,
)

# if _merge is left_only, then it was viewed not purchased, else it was purchased
events["is_buy"] = np.where(events["_merge"] == "left_only", 0, 1)
events.drop("_merge", axis=1, inplace=True)


######## transform to ids
item_encoder = LabelEncoder()
session_encoder = LabelEncoder()
events["item_id"] = item_encoder.fit_transform(events.item_id)
events["session_id"] = session_encoder.fit_transform(events.session_id)

##########sorted by user and timestamp
sorted_events = events.sort_values(by=["session_id", "timestamp"])

# Save processed data v_01 to pickle (optional)
#sorted_events.to_pickle(os.path.join(data_dir, 'Diginetica_processed_01.df'))


total_sessions = sorted_events.session_id.unique()
np.random.shuffle(total_sessions)

fractions = np.array([0.8, 0.1, 0.1])
# split into 3 parts
train_ids, val_ids, test_ids = np.array_split(
    total_sessions, (fractions[:-1].cumsum() * len(total_sessions)).astype(int)
)


train_sessions = sorted_events[sorted_events["session_id"].isin(train_ids)]
val_sessions = sorted_events[sorted_events["session_id"].isin(val_ids)]
test_sessions = sorted_events[sorted_events["session_id"].isin(test_ids)]

    to_pickled_df(data_directory, sampled_train=train_sessions)
    to_pickled_df(data_directory, sampled_val=val_sessions)
    to_pickled_df(data_directory, sampled_test=test_sessions)

if __name__ == "__main__":
    data_directory = "data"
    # sampled_buys=pd.read_pickle(os.path.join(data_directory, 'sampled_buys.df'))
    #
    # buy_sessions=sampled_buys.session_id.unique()
    sorted_events = pd.read_pickle(
        os.path.join(data_directory, "sorted_events.df")
    )  # NOTE:AD EDIT

    total_sessions = sorted_events.session_id.unique()
    np.random.shuffle(total_sessions)

    fractions = np.array([0.8, 0.1, 0.1])
    # split into 3 parts
    train_ids, val_ids, test_ids = np.array_split(
        total_sessions, (fractions[:-1].cumsum() * len(total_sessions)).astype(int)
    )

    train_sessions = sorted_events[sorted_events["session_id"].isin(train_ids)]
    val_sessions = sorted_events[sorted_events["session_id"].isin(val_ids)]
    test_sessions = sorted_events[sorted_events["session_id"].isin(test_ids)]

    to_pickled_df(data_directory, sampled_train=train_sessions)
    to_pickled_df(data_directory, sampled_val=val_sessions)
    to_pickled_df(data_directory, sampled_test=test_sessions)