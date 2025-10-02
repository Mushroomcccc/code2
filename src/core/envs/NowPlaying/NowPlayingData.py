import os
import sys
import pickle
from numba import njit
import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn.preprocessing import LabelEncoder
from scipy.sparse import csr_matrix
from src.core.envs.BaseData import BaseData


sys.path.extend([".", "./src", "./src/DeepCTR-Torch", "./src/tianshou"])
from src.core.envs.BaseData import BaseData
from scipy.spatial.distance import pdist, squareform

# ROOTPATH = os.path.dirname(__file__)
ROOTPATH = "data/NowPlaying"
DATAPATH = os.path.join(ROOTPATH, "data_raw")
PRODATAPATH = os.path.join(ROOTPATH, "data_processed")

for path in [PRODATAPATH]:
    if not os.path.exists(path):
        os.mkdir(path)


class NowPlayingData(BaseData):
    def __init__(self):
        super(NowPlayingData, self).__init__()
        self.train_data_path = "nowplaying_train.csv"
        self.val_data_path = "nowplaying_test.csv"

    def get_features(self, is_userinfo=False):
        user_features = ["user_id"]
        if not is_userinfo:
            user_features = ["user_id"]
        item_features = ["item_id"] + ["feat" + str(i) for i in range(1)] + ["score"]
        reward_features = ["rating"]
        return user_features, item_features, reward_features

    def get_df(self, name="nowplaying_train.csv"):
        filename = os.path.join(DATAPATH, name)
        df_data = pd.read_csv(filename,
                              usecols=['user_id', 'item_id', 'score', 'mode', 'rating', 'time'])
        #df_data['rating'] = df_data['rating'].astype(float)
        df_user = self.load_user_feat()

        # load feature info
        list_feat, df_feat = NowPlayingData.load_category()
        df_item = self.load_item_feat()
        df_data = df_data.join(df_feat, on=['item_id'], how="left")

        return df_data, df_user, df_item, list_feat

    def get_domination(self):
        df_data, _, df_item, _ = self.get_df(self.train_data_path)
        feature_domination_path = os.path.join(PRODATAPATH, "feature_domination.pickle")

        if os.path.isfile(feature_domination_path):
            item_feat_domination = pickle.load(open(feature_domination_path, 'rb'))
        else:
            item_feat_domination = self.get_sorted_domination_features(
                df_data, df_item, is_multi_hot=True, yname="rating", threshold=0.6)
            pickle.dump(item_feat_domination, open(feature_domination_path, 'wb'))
        return item_feat_domination
    
    def get_item_similarity(self):
        item_similarity_path = os.path.join(PRODATAPATH, "item_similarity.pickle")

        if os.path.isfile(item_similarity_path):
            item_similarity = pickle.load(open(item_similarity_path, 'rb'))
        else:
            mat_distance = NowPlayingData.get_saved_distance_mat()
            item_similarity = 1 / (mat_distance + 1)
            pickle.dump(item_similarity, open(item_similarity_path, 'wb'))
        return item_similarity
        
    def get_item_popularity(self):
        item_popularity_path = os.path.join(PRODATAPATH, "item_popularity.pickle")

        if os.path.isfile(item_popularity_path):
            item_popularity = pickle.load(open(item_popularity_path, 'rb'))
        else:
            train_path = os.path.join(DATAPATH, "nowplaying_train.csv")
            train_df = pd.read_csv(train_path, usecols=["user_id", "item_id", 'rating'])
            test_path = os.path.join(DATAPATH, "nowplaying_test.csv")
            test_df = pd.read_csv(test_path, usecols=['user_id', "item_id", 'rating'])
            df_data = pd.concat([train_df, test_df])
            n_users = df_data['user_id'].nunique()
            n_items = df_data['item_id'].nunique()
            
            df_data_filtered = df_data[df_data['rating']>=0.]
            
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
        filename = os.path.join(DATAPATH, self.train_data_path)
        df_data = pd.read_csv(filename, usecols=['user_id'])
        user_num = df_data['user_id'].nunique()
        df_user = pd.DataFrame(np.arange(user_num), columns=["user_id"])
        df_user.set_index("user_id", inplace=True)
        return df_user
    
    def load_item_feat(self):
        list_feat, df_feat = NowPlayingData.load_category()
        video_mean_duration = NowPlayingData.load_item_score()
        df_item = df_feat.join(video_mean_duration, on=['item_id'], how="left")

        return df_item

    @staticmethod
    def load_mat():
        filename_GT = os.path.join(DATAPATH, "rating_matrix.csv")
        mat = pd.read_csv(filename_GT, header=None).to_numpy()
        mat[mat < 0] = 0
        mat[mat > 5] = 5
        return mat

    @staticmethod
    def load_category(tag_label="key"):
        # load categories:
        print("load item feature")
        filepath = os.path.join(DATAPATH, 'item_categories.csv')
        df_feat0 = pd.read_csv(filepath, header=0)
        df_feat0['feat'] = df_feat0['key'].fillna('[-1]').map(lambda x: int(eval(x)[0]))
        df_feat0['feat'], tag_mapping = pd.factorize(df_feat0['feat'])

        list_feat = df_feat0.feat.to_list()
        # df_feat = pd.DataFrame(list_feat, columns=['feat0', 'feat1', 'feat2', 'feat3'], dtype=int)
        df_feat = pd.DataFrame(list_feat, columns=['feat0'])
        df_feat.index.name = "item_id"
        df_feat[df_feat.isna()] = -1
        df_feat = df_feat + 1
        df_feat = df_feat.astype(int)
        df_feat[tag_label] = list_feat

        return list_feat, df_feat

    @staticmethod
    def load_item_score():
        score_path = os.path.join(DATAPATH, "track_score_normed.csv")
        if os.path.isfile(score_path):
            item_mean_score = pd.read_csv(score_path, header=0)["score"]
        else:
            train_path = os.path.join(DATAPATH, "nowplaying_train.csv")
            train_score = pd.read_csv(train_path, usecols=["item_id", 'score'])
            test_path = os.path.join(DATAPATH, "nowplaying_test.csv")
            test_score = pd.read_csv(test_path, usecols=["item_id", 'score'])
            score_all = test_score.append(train_score)
            item_mean_score = score_all.groupby("item_id").agg(lambda x: sum(list(x)) / len(x))[
                "score"]
            item_mean_score.to_csv(score_path, index=False)

        item_mean_score.index.name = "item_id"
        return item_mean_score


    @staticmethod
    def get_saved_distance_mat():
        distance_mat_path = os.path.join(PRODATAPATH, f"distance_mat.csv")
        if os.path.isfile(distance_mat_path):
            print("loading small distance matrix...")
            distance_matrix_square = pickle.load(open(distance_mat_path, "rb"))
            print("loading completed.")
        else:
            print("computing distance matrix for the first time...")
            df_item = pd.read_csv(os.path.join(DATAPATH, "df_item.csv"))
            cols_dis = ['score', 'instrumentalness', 'liveness', 'speechiness', 
                     'danceability', 'valence', 'loudness', 'tempo', 'acousticness']
            # 提取相关的特征列
            df_features = df_item[cols_dis]
            # 计算 L2 距离矩阵（欧氏距离）
            distance_matrix = pdist(df_features, metric='euclidean')
            # 将距离矩阵转换为对称矩阵
            distance_matrix_square = squareform(distance_matrix)
            pickle.dump(distance_matrix_square, open(distance_mat_path, 'wb'))
        return distance_matrix_square


if __name__ == "__main__":
    dataset = NowPlayingData()
    df_train, df_user_train, df_item_train, _ = dataset.get_train_data()
    df_val, df_user_val, df_item_val, _ = dataset.get_val_data()
    print("KuaiRec: Train #user={}  #item={}  #inter={}".format(df_train['user_id'].nunique(), df_train['item_id'].nunique(), len(df_train)))
    print("KuaiRec: Test  #user={}  #item={}  #inter={}".format(df_val['user_id'].nunique(), df_val['item_id'].nunique(), len(df_val)))
