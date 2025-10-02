import numpy as np
import torch

from src.core.envs.Simulated_Env.base import BaseSimulatedEnv
import math


# from virtualTB.model.UserModel import UserModel
# from src.core.envs.VirtualTaobao.virtualTB.utils import *


class DNaIRSimulatedEnv(BaseSimulatedEnv):
    def __init__(self, ensemble_models,
                 env_task_class, task_env_param: dict, task_name: str,
                 predicted_mat=None,
                 item_quality=None,
                 item_popularity=None,
                 lambda_novelty=0.01,
                 ):
        super().__init__(ensemble_models, env_task_class, task_env_param, task_name, predicted_mat)
        self.item_quality = item_quality
        self.item_novelty = 1 / np.log(item_popularity + 2.)
        self.lambda_novelty = lambda_novelty

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

        # get diversity
        # torch.mean(0.4 * quality * 1 / torch.log10(popularity + 1.1), dim=1)
        quality = self._cal_quality(action)
        novelty = self._cal_novelty(action)

        intrinsic_reward = self.lambda_novelty * quality * novelty  # #math.log10(popularity + 1.1)
        final_reward = pred_reward + intrinsic_reward - self.MIN_R
        return final_reward

    def _cal_quality(self, action):
        if hasattr(self.env_task, "lbe_item"):
            p_id = self.env_task.lbe_item.inverse_transform([action])[0]
        else:
            p_id = action
        item_quality = self.item_quality[p_id]

        return item_quality

    def _cal_novelty(self, action):
        if hasattr(self.env_task, "lbe_item"):
            p_id = self.env_task.lbe_item.inverse_transform([action])[0]
        else:
            p_id = action
        item_pop = self.item_novelty[p_id]
        return item_pop
