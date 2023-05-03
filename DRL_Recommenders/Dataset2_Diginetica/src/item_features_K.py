import pandas as pd

import numpy as np


def create_feature_matrix(
    sorted_events, n_files=2, path_name="", one_hot_encode=True, top_features=500
):
    # for i in range(n_files):
    #     if i == 0:
    #         item_features = pd.read_csv(path_name + str(i + 1) + ".csv")
    #     else:
    #         item_features = pd.concat(
    #             [item_features, pd.read_csv(path_name + str(i + 1) + ".csv")],
    #             ignore_index=True,
    #         )

    item_features = pd.read_csv(
        "/Users/kashafali/Documents/Duke/Spring23/RL/Deep_Reinforcement_Learning_Recommenders/DRL_Recommenders/Dataset2_Diginetica/src/item_features/product_cat.csv"
    )

    item_features = item_features[
        item_features["itemId"].isin(sorted_events["item_id"].unique().tolist())
    ].drop_duplicates()
    # item_features["property_value"] = (
    #     item_features["property"].str.strip() + item_features["value"].str.strip()
    # )
    # item_features = item_features.drop(["timestamp"], axis=1).drop_duplicates()

    if one_hot_encode:
        one_hot_encoded = pd.DataFrame()
        itemids = []

        event_item_list = sorted_events.item_id.unique()
        event_item_list.sort()
        item_list = item_features["itemId"].unique()
        properties = (
            item_features["categoryId"].value_counts().head(top_features).index.tolist()
        )

        for item in event_item_list:
            if len(itemids) % 1000 == 0:
                print("hi")
            if item not in item_list:
                one_hot_encoded = pd.concat(
                    [one_hot_encoded, pd.DataFrame(np.zeros(len(properties))).T],
                    ignore_index=True,
                )
                itemids.append(item)
                continue

            item_properties = item_features[item_features["itemId"] == item][
                "categoryId"
            ].unique()
            one_hot_encoded = pd.concat(
                [
                    one_hot_encoded,
                    pd.DataFrame(
                        [1 if x in item_properties else 0 for x in properties]
                    ).T,
                ],
                ignore_index=True,
            )
            itemids.append(item)

        return one_hot_encoded, itemids

    else:
        return item_features, item_features["itemId"].unique().tolist()


# sorted_events = pd.read_pickle("sorted_events.df")
# one_hot_encoded, itemids = create_feature_matrix(
#     sorted_events, n_files=2, path_name="src/item_features/", one_hot_encode=True
# )
