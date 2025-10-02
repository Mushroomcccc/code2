import argparse
import sys
import traceback
import pickle
import random
import numpy as np
from copy import deepcopy
from gymnasium.spaces import Box
from torch.distributions import Independent, Normal
from gymnasium.spaces import Discrete
import math
from torch import nn

import torch

sys.path.extend([".", "./run", "./src", "./src/DeepCTR-Torch", "./src/tianshou"])
from policy.policy_utils import get_args_all, learn_policy, prepare_dir_log, prepare_user_model, setup_state_tracker, \
    get_true_env, prepare_test_envs

from src.core.collector.collector_set import CollectorSet
from src.core.util.data import get_env_args
from src.core.collector.collector import Collector
from src.core.policy.RecPolicy import RecPolicy

from src.core.envs.Simulated_Env.hrl_entroy import HRLEntroySimulatedEnv 

from src.tianshou.tianshou.data import HVectorReplayBuffer
from src.tianshou.tianshou.env import DummyVectorEnv
from src.tianshou.tianshou.utils.net.common import ActorCritic, Net
from src.tianshou.tianshou.utils.net.continuous import HActorProb
from src.tianshou.tianshou.utils.net.continuous import Critic as CCritic
from src.tianshou.tianshou.utils.net.discrete import Actor, Critic
from src.tianshou.tianshou.policy import HA2CCPolicy

# from util.upload import my_upload
import logzero

try:
    import envpool
except ImportError:
    envpool = None


def get_args_HDCRec():
    parser = argparse.ArgumentParser()
    # Env
    parser.add_argument('--lambda_entropy', default=0.001, type=float)

    # Agent
    parser.add_argument("--model_name", type=str, default="NovHA2C")
    parser.add_argument('--vf-coef', type=float, default=0.5)
    parser.add_argument('--ent-coef', type=float, default=0.0)
    parser.add_argument('--gae-lambda', type=float, default=1.)
    parser.set_defaults(exploration_noise=False)
    parser.add_argument('--remap_eps', default=0.01, type=float)
    parser.add_argument('--rew-norm', action="store_true", default=False)
    parser.add_argument('--action_scaling', default=False, type=bool)
    parser.add_argument('--action_bound_method', type=str, default="")
    parser.add_argument('--copy_state', default=1, type=int)
    parser.add_argument('--div', default=2., type=float)
    parser.add_argument('--max_grad_norm', default=1., type=float)
    parser.add_argument('--sigma_max', default=0.5, type=float)

    parser.add_argument("--message", type=str, default="FairHA2C")

    args = parser.parse_known_args()[0]
    return args


def prepare_train_envs_local(args, ensemble_models, env, dataset, kwargs_um):
    with open(ensemble_models.PREDICTION_MAT_PATH, "rb") as file:
        predicted_mat = pickle.load(file)

    item_similarity, item_popularity = dataset.get_item_similarity(), dataset.get_item_popularity()

    kwargs = {
        "ensemble_models": ensemble_models,
        "env_task_class": type(env),
        "task_env_param": kwargs_um,
        "task_name": args.env,
        "predicted_mat": predicted_mat,
        
        "item_similarity": item_similarity,
        "item_popularity": item_popularity,
        "lambda_entropy": args.lambda_entropy,
    }

    train_envs = DummyVectorEnv(
        [lambda: HRLEntroySimulatedEnv(**kwargs) for _ in range(args.training_num)], is_hrl=True)

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    train_envs.seed(args.seed)

    return train_envs


def setup_policy_model(args, state_tracker, train_envs, test_envs_dict, kwargs_um):
    if args.cpu:
        args.device = "cpu"
    else:
        args.device = torch.device("cuda:{}".format(args.cuda) if torch.cuda.is_available() else "cpu")
    
    # 上层智能体
    category = kwargs_um['list_feat']
    category_num = np.unique(category).shape[0]
    if 'lbe_item' in kwargs_um:
        category = list(map(lambda x: category[x], kwargs_um['lbe_item'].classes_))
    h_state_tracker = deepcopy(state_tracker) if args.copy_state==1 else state_tracker
    activation = nn.Tanh # nn.ReLU if args.env=='MovieLensEnv-v0' else nn.Tanh,GELU
    net = Net(args.state_dim + category_num * 1, hidden_sizes=args.hidden_sizes, device=args.device, activation=activation)

    h_actor = HActorProb(net, category_num, device=args.device, unbounded=True, softmax=True, sigma_max=args.sigma_max).to(args.device)
    h_critic = CCritic(
        net,
        device=args.device
    ).to(args.device)

    h_div = args.div
    h_optim_RL = torch.optim.Adam(ActorCritic(h_actor, h_critic).parameters(), lr=args.lr /h_div)
    h_optim_state = torch.optim.Adam(h_state_tracker.parameters(), lr=args.lr / h_div)
    h_optim = [h_optim_RL, h_optim_state]

    # 下层智能体
    l_state_tracker = state_tracker
    activation = nn.GELU # nn.GELU if args.env=='NowPlaying-v0' else nn.Tanh
    net = Net(args.state_dim + category_num * 2, hidden_sizes=args.hidden_sizes, device=args.device, activation=activation)

    l_actor = Actor(net, args.action_shape, device=args.device).to(args.device)
    l_critic = Critic(
        net,
        device=args.device
    ).to(args.device)

    l_optim_RL = torch.optim.Adam(ActorCritic(l_actor, l_critic).parameters(), lr=args.lr)
    l_optim_state = torch.optim.Adam(l_state_tracker.parameters(), lr=args.lr)
    l_optim = [l_optim_RL, l_optim_state]

    def h_dist(*logits):
        return Independent(Normal(*logits), 1)
    l_dist = torch.distributions.Categorical

    policy = HA2CCPolicy(
        h_actor,
        l_actor,
        h_critic,
        l_critic,
        h_optim,
        l_optim,
        h_dist_fn=h_dist,
        l_dist_fn=l_dist,
        h_state_tracker=h_state_tracker,
        l_state_tracker=l_state_tracker,
        discount_factor=args.gamma,
        gae_lambda=args.gae_lambda,
        vf_coef=args.vf_coef,
        ent_coef=args.ent_coef,
        max_grad_norm=args.max_grad_norm,
        reward_normalization=args.rew_norm,
        action_space=Discrete(args.action_shape),
        action_bound_method=args.action_bound_method,  # not clip
        action_scaling=args.action_scaling,
        device=args.device,
        item_types=category,
    )

    policy.set_eps(args.explore_eps)
    
    rec_policy = RecPolicy(args, policy, state_tracker)

    # Prepare the collectors and logs
    train_collector = Collector(
        rec_policy, train_envs,
        HVectorReplayBuffer(args.buffer_size, len(train_envs)),
        # preprocess_fn=state_tracker.build_state,
        exploration_noise=args.exploration_noise,
        remove_recommended_ids=args.remove_recommended_ids,
        is_hrl=True
    )

    test_collector_set = CollectorSet(rec_policy, test_envs_dict, args.buffer_size, args.test_num,
                                      #   preprocess_fn=state_tracker.build_state,
                                      exploration_noise=args.exploration_noise,
                                      force_length=args.force_length,
                                      is_hrl=True, 
                                      info=args.env + '_' + args.message)

    return rec_policy, train_collector, test_collector_set, [h_optim_RL, h_optim_state, l_optim_RL, h_optim_state]


def main(args):
    # %% 1. Prepare the saved path.
    MODEL_SAVE_PATH, logger_path = prepare_dir_log(args)

    # %% 2. Prepare user model and environment
    ensemble_models = prepare_user_model(args)
    env, dataset, kwargs_um = get_true_env(args)
    train_envs = prepare_train_envs_local(args, ensemble_models, env, dataset, kwargs_um)
    test_envs_dict = prepare_test_envs(args, env, kwargs_um, is_hrl=True)

    # %% 3. Setup policy
    state_tracker = setup_state_tracker(args, ensemble_models, env, train_envs, test_envs_dict)
    policy, train_collector, test_collector_set, optim = setup_policy_model(args,state_tracker, train_envs, test_envs_dict, kwargs_um)

    # %% 4. Learn policy
    learn_policy(args, env, dataset, policy, train_collector, test_collector_set, state_tracker, optim, MODEL_SAVE_PATH,
                 logger_path, trainer="onpolicy")


if __name__ == '__main__':
    trainer = "onpolicy"
    args_all = get_args_all(trainer)
    #args_all.remove_recommended_ids = True
    args = get_env_args(args_all)
    args_A2C = get_args_HDCRec()
    args_all.__dict__.update(args.__dict__)
    args_all.__dict__.update(args_A2C.__dict__)
    try:
        main(args_all)
    except Exception as e:
        var = traceback.format_exc()
        print(var)
        logzero.logger.error(var)
