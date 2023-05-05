"""Script to create item feature matrix."""
import pandas as pd
import numpy as np
import os


def create_features(
    sorted_events, n_files=2, path_name="", one_hot_encode=True, top_features=500
):
    """Create item feature matrix."""

    data_path = "Data"
    features_item = pd.read_csv(os.path.join(data_path, "product_cat.csv"))

    features_item = features_item[
        features_item["itemId"].isin(sorted_events["item_id"].unique().tolist())
    ].drop_duplicates()
    if one_hot_encode:
        one_hot_encoded = pd.DataFrame()
        itemids = []

        event_list = sorted_events.item_id.unique()
        event_list.sort()
        item_list = features_item["itemId"].unique()
        properties = (
            features_item["categoryId"].value_counts().head(top_features).index.tolist()
        )

        for i in event_list:
            if j not in item_list:
                one_hot_encoded = pd.concat(
                    [one_hot_encoded, pd.DataFrame(np.zeros(len(properties))).T],
                    ignore_index=True,
                )
                itemids.append(j)
                continue

            item_properties = features_item[features_item["itemId"] == i][
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
            itemids.append(i)

        return one_hot_encoded, itemids

    else:
        return features_item, features_item["itemId"].unique().tolist()


# sorted_events = pd.read_pickle("sorted_events.df")
# one_hot_encoded, itemids = create_features(
#     sorted_events, n_files=2, path_name="src/item_features/", one_hot_encode=True
# )
