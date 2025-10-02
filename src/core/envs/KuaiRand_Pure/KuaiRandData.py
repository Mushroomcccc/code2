import os
import sys
import pickle
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from scipy.sparse import csr_matrix

from src.core.envs.KuaiRand_Pure.preprocessing_kuairand import get_df_data

sys.path.extend([".", "./src", "./src/DeepCTR-Torch", "./src/tianshou"])
from src.core.envs.BaseData import BaseData, get_distance_mat

# ROOTPATH = os.path.dirname(__file__)
ROOTPATH = "data/KuaiRand_Pure"
DATAPATH = os.path.join(ROOTPATH, "data_raw")
PRODATAPATH = os.path.join(ROOTPATH, "data_processed")

for path in [PRODATAPATH]:
    if not os.path.exists(path):
        os.mkdir(path)


class KuaiRandData(BaseData):
    def __init__(self):
        super(KuaiRandData, self).__init__()
        self.train_data_path = "train_processed.csv"
        self.val_data_path = "test_processed.csv"
        
    def get_features(self, is_userinfo=False):
        user_features = ["user_id", 'user_active_degree', 'is_live_streamer', 'is_video_author',
                         'follow_user_num_range',
                         'fans_user_num_range', 'friend_user_num_range', 'register_days_range'] \
                        + [f'onehot_feat{x}' for x in range(18)]
        if not is_userinfo:
            user_features = ["user_id"]
        item_features = ["item_id"] + ["feat" + str(i) for i in range(1)] + ["duration_normed"]
        reward_features = ["is_click"]
        return user_features, item_features, reward_features
    
    def get_df(self, name, is_sort=True):
        filename = os.path.join(DATAPATH, name)
        df_data = get_df_data(filename,
                              usecols=['user_id', 'item_id', 'time_ms', 'is_like', 'is_click', 'long_view',
                                       'duration_normed', "watch_ratio_normed"])

        # df_data['watch_ratio'] = df_data["play_time_ms"] / df_data["duration_ms"]
        # df_data.loc[df_data['watch_ratio'].isin([np.inf, np.nan]), 'watch_ratio'] = 0
        # df_data.loc[df_data['watch_ratio'] > 5, 'watch_ratio'] = 5
        # df_data['duration_01'] = df_data['duration_ms'] / 1e5
        # df_data.rename(columns={"time_ms": "timestamp"}, inplace=True)
        # df_data["timestamp"] /= 1e3

        # load feature info
        list_feat, df_feat = KuaiRandData.load_category()
        df_data = df_data.join(df_feat, on=['item_id'], how="left")

        df_item = self.load_item_feat()

        # load user info
        df_user = self.load_user_feat()
        df_data = df_data.join(df_user, on=['user_id'], how="left")

        # get user sequences
        if is_sort:
            df_data.sort_values(["user_id", "time_ms"], inplace=True)
            df_data.reset_index(drop=True, inplace=True)

        return df_data, df_user, df_item, list_feat

    def get_domination(self):
        df_data, _, df_item, _ = self.get_df("train_processed.csv")
        feature_domination_path = os.path.join(PRODATAPATH, "feature_domination.pickle")

        if os.path.isfile(feature_domination_path):
            item_feat_domination = pickle.load(open(feature_domination_path, 'rb'))
        else:
            item_feat_domination = self.get_sorted_domination_features(
                df_data, df_item, is_multi_hot=True, yname="is_click", threshold=1)
            pickle.dump(item_feat_domination, open(feature_domination_path, 'wb'))
        return item_feat_domination
    
    def get_item_similarity(self):
        item_similarity_path = os.path.join(PRODATAPATH, "item_similarity.pickle")

        if os.path.isfile(item_similarity_path):
            item_similarity = pickle.load(open(item_similarity_path, 'rb'))
        else:
            yname = "is_click"
            mat = KuaiRandData.load_mat(yname)
            mat_distance = KuaiRandData.get_saved_distance_mat(yname, mat)
            item_similarity = 1 / (mat_distance + 1)
            pickle.dump(item_similarity, open(item_similarity_path, 'wb'))
        return item_similarity
        
    def get_item_popularity(self):
        item_popularity_path = os.path.join(PRODATAPATH, "item_popularity.pickle")

        if os.path.isfile(item_popularity_path):
            item_popularity = pickle.load(open(item_popularity_path, 'rb'))
        else:
            filename = os.path.join(DATAPATH, "train_processed.csv")
            df_data = get_df_data(filename, usecols=['user_id', 'item_id', 'is_click'])

            n_users = df_data['user_id'].nunique()
            n_items = df_data['item_id'].nunique()
            n_items = 7583 # hard coded! Becuase there are 7583 items in the test data while there are 7538 items in the training data.

            df_data_filtered = df_data[df_data["is_click"]>=0]
            
            groupby = df_data_filtered.loc[:, ["user_id", "item_id"]].groupby(by="item_id")
            df_pop = groupby.user_id.apply(list).reset_index()
            df_pop["popularity"] = df_pop['user_id'].apply(lambda x: len(x) / n_users)

            item_pop_df = pd.DataFrame(np.arange(n_items), columns=["item_id"])
            item_pop_df = item_pop_df.merge(df_pop, how="left", on="item_id")
            item_pop_df['popularity'].fillna(0, inplace=True)
            item_popularity = item_pop_df['popularity']
            pickle.dump(item_popularity, open(item_popularity_path, 'wb'))
        
        return item_popularity

    def load_user_feat(self):
        print("load user features")
        filepath = os.path.join(DATAPATH, 'user_features_pure.csv')
        df_user = pd.read_csv(filepath, usecols=['user_id', 'user_active_degree',
                                                 'is_live_streamer', 'is_video_author', 'follow_user_num_range',
                                                 'fans_user_num_range', 'friend_user_num_range',
                                                 'register_days_range'] + [f'onehot_feat{x}' for x in range(18)]
                              )
        for col in ['user_active_degree',
                    'is_live_streamer', 'is_video_author', 'follow_user_num_range',
                    'fans_user_num_range', 'friend_user_num_range', 'register_days_range']:

            df_user[col] = df_user[col].map(lambda x: chr(0) if x == 'UNKNOWN' else x)
            lbe = LabelEncoder()
            df_user[col] = lbe.fit_transform(df_user[col])
            # print(lbe.classes_)
            if chr(0) in lbe.classes_.tolist() or -124 in lbe.classes_.tolist():
                assert lbe.classes_[0] in {-124, chr(0)}
                # do not add one
            else:
                df_user[col] += 1
        for col in [f'onehot_feat{x}' for x in range(18)]:
            df_user.loc[df_user[col].isna(), col] = -124
            lbe = LabelEncoder()
            df_user[col] = lbe.fit_transform(df_user[col])
            # print(lbe.classes_)
            if chr(0) in lbe.classes_.tolist() or -124 in lbe.classes_.tolist():
                assert lbe.classes_[0] in {-124, chr(0)}
                # do not add one
            else:
                df_user[col] += 1

        df_user = df_user.set_index("user_id")
        return df_user

    def load_item_feat(self):
        list_feat, df_feat = KuaiRandData.load_category()
        video_mean_duration = KuaiRandData.load_video_duration()
        df_item = df_feat.join(video_mean_duration, on=['item_id'], how="left")

        return df_item


    @staticmethod
    def load_mat(yname="is_click", read_user_num=None):
        filename = ""
        if yname == "is_click":
            filename = "kuairand_is_click.csv"
        elif yname == "is_like":
            filename = "kuairand_is_like.csv"
        elif yname == "long_view":
            filename = "kuairand_long_view.csv"
        elif yname == "watch_ratio_normed":
            filename = "kuairand_watchratio.csv"

        filepath_GT = os.path.join(DATAPATH, "MF_results_GT", filename)
        df_mat = pd.read_csv(filepath_GT, header=0)

        if not read_user_num is None:
            df_mat_part = df_mat.loc[df_mat["user_id"] < read_user_num]
        else:
            df_mat_part = df_mat

        num_user = df_mat_part['user_id'].nunique()
        num_item = df_mat_part['item_id'].nunique()
        assert num_user == 27285
        assert num_item == 7583

        mat = csr_matrix((df_mat_part["value"], (df_mat_part['user_id'], df_mat_part['item_id'])),
                         shape=(num_user, num_item)).toarray()
        
        return mat

    @staticmethod
    def load_category(tag_label="tags"):
        print("load item feature")
        filepath = os.path.join(DATAPATH, 'video_features_basic_pure.csv')
        df_item = pd.read_csv(filepath, usecols=["tag"], dtype=str)
        df_item['tag'] = df_item['tag'].fillna('0').map(lambda x: int(x.split(',')[0]))
        df_item['tag'], tag_mapping = pd.factorize(df_item['tag'])


        list_feat = df_item['tag'].to_list()
        df_feat = pd.DataFrame(list_feat, columns=['feat0'])
        df_feat.index.name = "item_id"
        df_feat[df_feat.isna()] = -1
        df_feat = df_feat + 1
        df_feat = df_feat.astype(int)
        df_feat[tag_label] = list_feat

        return list_feat, df_feat

    @staticmethod
    def load_video_duration():
        duration_path = os.path.join(PRODATAPATH, "video_duration_normed.csv")
        if os.path.isfile(duration_path):
            video_mean_duration = pd.read_csv(duration_path, header=0)["duration_normed"]
        else:
            small_path = os.path.join(DATAPATH, "test_processed.csv")
            small_duration = get_df_data(small_path, usecols=["item_id", 'duration_normed'])
            big_path = os.path.join(DATAPATH, "train_processed.csv")
            big_duration = get_df_data(big_path, usecols=["item_id", 'duration_normed'])
            duration_all = small_duration.append(big_duration)
            video_mean_duration = duration_all.groupby("item_id").agg(lambda x: sum(list(x)) / len(x))[
                "duration_normed"]
            video_mean_duration.to_csv(duration_path, index=False)

        video_mean_duration.index.name = "item_id"
        return video_mean_duration

    @staticmethod
    def get_saved_distance_mat(yname, mat):
        distance_mat_path = os.path.join(PRODATAPATH, f"distance_mat_{yname}.csv")
        if os.path.isfile(distance_mat_path):
            print("loading small distance matrix...")
            mat_distance = pickle.load(open(distance_mat_path, "rb"))
            print("loading completed.")
        else:
            print("computing distance matrix for the first time...")
            num_item = mat.shape[1]
            distance = np.zeros([num_item, num_item])
            mat_distance = get_distance_mat(mat, distance)
            print(f"saving the distance matrix to {distance_mat_path}...")
            pickle.dump(mat_distance, open(distance_mat_path, 'wb'))
        return mat_distance


if __name__ == "__main__":
    dataset = KuaiRandData()
    df_train, df_user_train, df_item_train, _ = dataset.get_train_data()
    df_val, df_user_val, df_item_val, _ = dataset.get_val_data()
    print(df_train.columns)
    print("KuaiRand: Train #user={}  #item={}  #inter={}".format(df_train['user_id'].nunique(), df_train['item_id'].nunique(), len(df_train)))
    print("KuaiRand: Test  #user={}  #item={}  #inter={}".format(df_val['user_id'].nunique(), df_val['item_id'].nunique(), len(df_val)))
