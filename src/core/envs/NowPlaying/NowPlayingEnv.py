import os
import sys
import itertools
from collections import Counter
import numpy as np

from src.core.envs.BaseEnv import BaseEnv
sys.path.extend(["./src", "./src/DeepCTR-Torch", "./src/tianshou"])

from src.core.envs.NowPlaying.NowPlayingData import NowPlayingData

# ROOTPATH = os.path.dirname(__file__)
ROOTPATH = "data/nowplaying"
DATAPATH = os.path.join(ROOTPATH, "data_raw")


class NowPlayingEnv(BaseEnv):
    metadata = {'render.modes': ['human']}

    def __init__(self, yname, mat=None, mat_distance=None, list_feat=None, 
                 num_leave_compute=5, leave_threshold=1, max_turn=100, random_init=False):
        self.yname = yname
        if mat is not None:
            self.mat = mat
            self.list_feat = list_feat
            self.mat_distance = mat_distance
        else:
            self.mat, self.list_feat, self.mat_distance = self.load_env_data(yname)

        super(NowPlayingEnv, self).__init__(num_leave_compute, leave_threshold, max_turn, random_init)
        self.category_distribution = np.array([0.0] * np.unique(self.list_feat))

    @staticmethod
    def load_env_data():
        mat = NowPlayingData.load_mat()
        list_feat, df_feat = NowPlayingData.load_category()
        mat_distance = NowPlayingData.get_saved_distance_mat()
        return mat, list_feat, mat_distance

    def get_dis(self):
        return self.category_distribution   
    
    def _determine_whether_to_leave(self, t, action):
        self.category_distribution[:] = 0.
        self.category_distribution[self.list_feat[action]] += 1
        if t == 0:
            return False, self.category_distribution

        window_actions = self.sequence_action[t - self.num_leave_compute:t]
        hist_categories_each = list(map(lambda x: self.list_feat[x], window_actions))
        hist_dict = Counter(hist_categories_each)
        category_a = self.list_feat[action]

        # 计算用户历史交互物品类别的概率分布
        self.category_distribution[:] = 0.
        his_actions = self.sequence_action[:t]
        for x in his_actions:
            c = self.list_feat[x]
            self.category_distribution[c] += 1
        self.category_distribution[self.list_feat[action]] += 1
        self.category_distribution /= np.sum(self.category_distribution)
        
        # 新退出机制一：当num_leave_compute窗口内有重复的类别，最大交互次数减1
        if hist_dict[category_a] > self.leave_threshold:
            self.bored_punishment += 2

        # 待实现
        # 新退出机制二：当用户的交互次数达到n时，如果其交互物品类别的分布的熵小于alpha*最大熵，用户退出
        # 新的退出机制三：当前的推荐类别概率分布的熵小于alpha*最大熵，用户退出
        # if entropy < self.entropy_threshold:
        #     return True, self.category_distribution
        # 计算物品类别概率分布熵
        # p = self.category_distribution
        # p = p / np.sum(p)  # 若不是概率分布，先归一化
        # eps = 1e-10  # 防止 log(0)
        # entropy = -np.sum(p * np.log(p + eps))
        return False, self.category_distribution
        # if hist_dict[category_a] > self.leave_threshold:
        #     return True, self.category_distribution
        # else:
        #     return False, self.category_distribution

