import argparse
from collections import Counter, defaultdict
import os
import sys
import traceback
import pickle
import random

import numpy as np
import torch
from gymnasium.spaces import Discrete
from tqdm import tqdm

sys.path.extend([".", "./run/policy", "./src", "./src/DeepCTR-Torch", "./src/tianshou"])

from policy_utils import get_args_all, learn_policy, prepare_dir_log, prepare_user_model, prepare_test_envs, setup_state_tracker

# os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

from src.core.collector.collector_set import CollectorSet
from src.core.util.data import get_env_args, get_true_env
from src.core.collector.collector import Collector
from src.core.envs.Simulated_Env.penalty_ent_exp import PenaltyEntExpSimulatedEnv, get_features_of_last_n_items_features
from src.core.policy.RecPolicy import RecPolicy

from src.tianshou.tianshou.data import VectorReplayBuffer
from src.tianshou.tianshou.env import DummyVectorEnv
from src.tianshou.tianshou.utils.net.common import ActorCritic, Net
from src.tianshou.tianshou.utils.net.discrete import Actor, Critic
from src.tianshou.tianshou.policy import PPOPolicy
from src.core.util.layers import Linear, create_embedding_matrix

# from util.upload import my_upload
import logzero

try:
    import envpool
except ImportError:
    envpool = None


def get_args_CIRS():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, default="CIRS")
    parser.add_argument('--vf-coef', type=float, default=0.5)
    parser.add_argument('--ent-coef', type=float, default=0.0)
    parser.add_argument('--max-grad-norm', type=float, default=None)
    parser.add_argument('--gae-lambda', type=float, default=1.)
    parser.add_argument('--rew-norm', action="store_true", default=False)

    # Env
    parser.add_argument('--is_exposure_intervention', dest='use_exposure_intervention', action='store_true')
    parser.add_argument('--no_exposure_intervention', dest='use_exposure_intervention', action='store_false')
    parser.set_defaults(use_exposure_intervention=True)
    parser.add_argument('--tau', default=100, type=float)
    parser.add_argument('--gamma_exposure', default=1, type=float)
    parser.add_argument("--entropy_window", type=int, nargs="*", default=[])

    parser.add_argument("--version", type=str, default="v1")

    parser.add_argument("--read_message", type=str, default="CIRS_UM")
    parser.add_argument("--message", type=str, default="CIRS")

    args = parser.parse_known_args()[0]
    return args


# def prepare_dataset(args, dataset, MODEL_SAVE_PATH, DATAPATH):
#     dataset_train, df_user, df_item, x_columns, y_columns, ab_columns = \
#         load_dataset_train(args, dataset, args.tau, args.entity_dim, args.feature_dim, MODEL_SAVE_PATH, DATAPATH)
#     if not args.is_ab:
#         ab_columns = None

#     dataset_val, df_user_val, df_item_val = load_dataset_val(args, dataset, args.entity_dim, args.feature_dim)
    
#     assert dataset_train.x_columns[1].vocabulary_size >= dataset_val.x_columns[1].vocabulary_size  # item_ids of training set should cover the test set!

#     ab_embedding_dict = create_embedding_matrix(ab_columns, init_std, sparse=False, device=device)
#             for tensor in ab_embedding_dict.values():
#                 nn.init.normal_(tensor.weight, mean=1, std=init_std)
#     return dataset_train, dataset_val, df_user, df_item, df_user_val, df_item_val, x_columns, y_columns, ab_columns


def prepare_train_envs(args, ensemble_models, env, dataset, kwargs_um):
    args.entropy_window = [] # CIRS does not need to compute entropy!
    args.lambda_entropy = 0
    entropy_dict = dict()

    with open(ensemble_models.PREDICTION_MAT_PATH, "rb") as file:
        predicted_mat = pickle.load(file)

   # alpha_u, beta_i = None, None  ## TODO
    ab_path = ensemble_models.AB_PATH
    with open(ab_path, "rb") as f:
        ab_columns = pickle.load(f)
    ab_embedding_dict = create_embedding_matrix(ab_columns, sparse=False)

    alpha_u = ab_embedding_dict['alpha_u'].weight.detach().cpu().numpy()
    beta_i = ab_embedding_dict['beta_i'].weight.detach().cpu().numpy()


    kwargs = {
        "ensemble_models": ensemble_models,
        "env_task_class": type(env),
        "task_env_param": kwargs_um,
        "task_name": args.env,
        "predicted_mat": predicted_mat,
        "version": args.version,
        "tau": args.tau,
        "use_exposure_intervention": args.use_exposure_intervention,
        "gamma_exposure": args.gamma_exposure,
        "alpha_u": alpha_u,
        "beta_i": beta_i,
        "entropy_dict": entropy_dict,
        "entropy_window": args.entropy_window,
        "lambda_entropy": args.lambda_entropy,
        "step_n_actions": max(args.entropy_window) if len(args.entropy_window) else 0,
        "min_r": 0,
    }

    train_envs = DummyVectorEnv(
        [lambda: PenaltyEntExpSimulatedEnv(**kwargs) for _ in range(args.training_num)])

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    train_envs.seed(args.seed)

    return train_envs


def setup_policy_model(args, state_tracker, train_envs, test_envs_dict):
    if args.cpu:
        args.device = "cpu"
    else:
        args.device = torch.device("cuda:{}".format(args.cuda) if torch.cuda.is_available() else "cpu")

    # model
    net = Net(args.state_dim, hidden_sizes=args.hidden_sizes, device=args.device)
    actor = Actor(net, args.action_shape, device=args.device).to(args.device)
    critic = Critic(net, device=args.device).to(args.device)
    optim_RL = torch.optim.Adam(ActorCritic(actor, critic).parameters(), lr=args.lr)
    optim_state = torch.optim.Adam(state_tracker.parameters(), lr=args.lr)
    optim = [optim_RL, optim_state]

    dist = torch.distributions.Categorical
    policy = PPOPolicy(
        actor,
        critic,
        optim,
        dist,
        state_tracker=state_tracker,
        discount_factor=args.gamma,
        gae_lambda=args.gae_lambda,
        vf_coef=args.vf_coef,
        ent_coef=args.ent_coef,
        max_grad_norm=args.max_grad_norm,
        reward_normalization=args.rew_norm,
        action_space=Discrete(args.action_shape),
       # value_clip=args.value_clip,
        action_bound_method="",  # not clip
        action_scaling=False
    )

    rec_policy = RecPolicy(args, policy, state_tracker)

    # Prepare the collectors and logs
    train_collector = Collector(
        rec_policy, train_envs,
        VectorReplayBuffer(args.buffer_size, len(train_envs)),
        # preprocess_fn=state_tracker.build_state,
        exploration_noise=args.exploration_noise,
        remove_recommended_ids=args.remove_recommended_ids
    )

    test_collector_set = CollectorSet(rec_policy, test_envs_dict, args.buffer_size, args.test_num,
                                      #   preprocess_fn=state_tracker.build_state,
                                      exploration_noise=args.exploration_noise,
                                      force_length=args.force_length,
                                      info=args.env + '_' + args.message)

    return rec_policy, train_collector, test_collector_set, optim


def main(args):
    # %% 1. Prepare the saved path.
    MODEL_SAVE_PATH, logger_path = prepare_dir_log(args)

    # %% 2. Prepare user model and environment
    ensemble_models = prepare_user_model(args)
    env, dataset, kwargs_um = get_true_env(args)
    train_envs = prepare_train_envs(args, ensemble_models, env, dataset, kwargs_um)
    test_envs_dict = prepare_test_envs(args, env, kwargs_um)

    # %% 3. Setup policy
    state_tracker = setup_state_tracker(args, ensemble_models, env, train_envs, test_envs_dict)
    policy, train_collector, test_collector_set, optim = setup_policy_model(args, state_tracker, train_envs,
                                                                            test_envs_dict)

    # %% 4. Learn policy
    learn_policy(args, env, dataset, policy, train_collector, test_collector_set, state_tracker, optim, MODEL_SAVE_PATH,
                 logger_path, trainer="onpolicy")


if __name__ == '__main__':
    trainer = "onpolicy"
    args_all = get_args_all(trainer)
    #args_all.remove_recommended_ids =  True
    args = get_env_args(args_all)
    args_DORL = get_args_CIRS()
    args_all.__dict__.update(args.__dict__)
    args_all.__dict__.update(args_DORL.__dict__)
    try:
        main(args_all)
    except Exception as e:
        var = traceback.format_exc()
        print(var)
        logzero.logger.error(var)
