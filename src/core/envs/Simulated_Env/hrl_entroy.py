import numpy as np
import torch
from torch import FloatTensor
from src.core.envs.Simulated_Env.base import BaseSimulatedEnv
import math

# from virtualTB.model.UserModel import UserModel
# from src.core.envs.VirtualTaobao.virtualTB.utils import *


class HRLEntroySimulatedEnv(BaseSimulatedEnv):
    def __init__(self, ensemble_models,
                 env_task_class, task_env_param: dict, task_name: str,
                 predicted_mat=None,
                 item_similarity=None,
                 item_popularity=None,
                 lambda_entropy=0.001,
                 ):
        super().__init__(ensemble_models, env_task_class, task_env_param, task_name, predicted_mat)
        self.item_similarity = item_similarity
        self.item_popularity = item_popularity
        self.lambda_entropy = lambda_entropy

    def reset(self):
        return super().reset()

    def _compute_pred_reward(self, action):
        if self.env_name == "VirtualTB-v0":
            feature = np.concatenate((self.cur_user, np.array([self.reward, 0, self.total_turn]), action), axis=-1)
            feature_tensor = torch.unsqueeze(torch.tensor(feature, device=self.user_model.device, dtype=torch.float), 0)
            # pred_reward = self.user_model(feature_tensor).detach().cpu().numpy().squeeze().round()
            pred_reward = self.user_model.forward(feature_tensor).detach().cpu().numpy().squeeze()
            if pred_reward < 0:
                pred_reward = 0
            if pred_reward > 10:
                pred_reward = 10
        else:  # elif self.env_name == "KuaiEnv-v0":
            # get prediction
            pred_reward = self.predicted_mat[self.cur_user, action]  # todo

        pred_reward = pred_reward - self.MIN_R
        return pred_reward

    def step(self, action: FloatTensor):
        # 1. Collect ground-truth transition info
        self.action = action
        # real_state, real_reward, real_done, real_info = self.env_task.step(action)
        real_state, real_reward, _, c_dis, real_terminated, real_truncated, real_info = self.env_task.step(action)

        t = int(self.total_turn)
        terminated = real_terminated

        if t < self.env_task.max_turn:
            self._add_action_to_history(t, action)
        
        # ---------Calculate L-reward---------
        l_reward = self._compute_pred_reward(action)

        self.cum_reward += l_reward
        self.total_turn = self.env_task.total_turn

        # Rethink commented, do not use new user as new state
        # if terminated:
        #     self.state, self.info = self.env_task.reset()
        self.state = self._construct_state(l_reward)

        # info =  {'CTR': self.cum_reward / self.total_turn / 10}
        # ---------Calculate H-reward----------
        # Calculate the entropy of the probability distribution of items
        p = c_dis
        eps = 1e-10  
        entropy = -np.sum(p * np.log(p + eps))
        h_reward = l_reward + self.lambda_entropy * entropy
        l_reward = h_reward

        truncated = False
        info = {'cum_reward': self.cum_reward}
        return self.state, l_reward, h_reward, c_dis, terminated, truncated, info
