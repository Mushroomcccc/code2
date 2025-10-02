import argparse
import sys
import traceback
import pickle
import random

import numpy as np
import pandas as pd
import torch
import math
from gymnasium.spaces import Box


sys.path.extend([".", "./run", "./src", "./src/DeepCTR-Torch", "./src/tianshou"])
from policy.policy_utils import get_args_all, learn_policy, prepare_dir_log, prepare_user_model, prepare_test_envs, \
    setup_state_tracker
from src.core.util.data import get_env_args, get_true_env
# os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

from src.core.collector.collector_set import CollectorSet
from src.core.util.data import get_env_args
from src.core.collector.collector import Collector
from src.core.policy.RecPolicy import RecPolicy

from src.tianshou.tianshou.data import VectorReplayBuffer

from src.tianshou.tianshou.utils.net.common import Net
from src.tianshou.tianshou.utils.net.continuous import ActorProb, Critic
from src.tianshou.tianshou.policy import SACPolicy
from src.core.envs.Simulated_Env.sac4ir_env import SAC4IRSimulatedEnv
from src.tianshou.tianshou.env import DummyVectorEnv


from src.tianshou.tianshou.exploration import GaussianNoise

# from util.upload import my_upload
import logzero
import math

try:
    import envpool
except ImportError:
    envpool = None


def get_args_SAC4IR():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, default="SAC4IR")
    #Env
    parser.add_argument('--lambda_temper', default=0.01, type=float)
    # sac special
    parser.add_argument('--actor-lr', type=float, default=1e-4)
    parser.add_argument('--critic-lr', type=float, default=1e-4)
    parser.add_argument('--tau', type=float, default=0.01)
    parser.add_argument('--gama', type=float, default=0.9)

    parser.add_argument('--remap_eps', default=0.01, type=float)
    parser.add_argument('--target_entropy_ratio', default=0.9, type=float)
    parser.add_argument('--rew-norm', action="store_true", default=False)


    parser.add_argument("--message", type=str, default="SAC4IR")

    args = parser.parse_known_args()[0]
    return args


def setup_policy_model(args, state_tracker, train_envs, test_envs_dict):
    if args.cpu:
        args.device = "cpu"
    else:
        args.device = torch.device("cuda:{}".format(args.cuda) if torch.cuda.is_available() else "cpu")

    # Continuous Action，action_shape = state_tracker.emb_dim
    # model
    net = Net(args.state_dim, hidden_sizes=args.hidden_sizes, device=args.device)
    actor = ActorProb(net, state_tracker.emb_dim, device=args.device).to(args.device)
    net = Net(args.state_dim + state_tracker.emb_dim, hidden_sizes=args.hidden_sizes, device=args.device)
    critic_1 = Critic(net, device=args.device).to(args.device)
    net = Net(args.state_dim + state_tracker.emb_dim, hidden_sizes=args.hidden_sizes, device=args.device)
    critic_2 = Critic(net, device=args.device).to(args.device)

    optim_actor = torch.optim.Adam(actor.parameters(), lr=args.lr)
    optim_critic_1 = torch.optim.Adam(critic_1.parameters(), lr=args.lr)
    optim_critic_2 = torch.optim.Adam(critic_2.parameters(), lr=args.lr)
    optim_state = torch.optim.Adam(state_tracker.parameters(), lr=args.lr)

    policy = SACPolicy(
        actor,
        optim_actor,
        critic_1,
        optim_critic_1,
        critic_2,
        optim_critic_2,
        state_tracker,
        optim_state,
        n_actions=args.action_shape,
        tau=args.tau,
        gamma=args.gamma,
        exploration_noise=GaussianNoise(sigma=args.explore_eps),
        reward_normalization=args.rew_norm,
        estimation_step=args.n_step,
        action_space=Box(shape=(state_tracker.emb_dim,), low=0, high=1),
        device=args.device,
        target_entropy_ratio=args.target_entropy_ratio,
    )

    rec_policy = RecPolicy(args, policy, state_tracker)

    # Prepare the collectors and logs
    assert args.exploration_noise == True
    train_collector = Collector(
        rec_policy, train_envs,
        VectorReplayBuffer(args.buffer_size, len(train_envs)),
        exploration_noise=args.exploration_noise,
        remove_recommended_ids=args.remove_recommended_ids
    )

    test_collector_set = CollectorSet(rec_policy, test_envs_dict, args.buffer_size, args.test_num,
                                      force_length=args.force_length,
                                      info=args.env + '_' + args.message)

    return rec_policy, train_collector, test_collector_set, [optim_actor, optim_critic_1, optim_critic_2,
                                                             optim_state]  # TODO


def prepare_train_envs_local(args, ensemble_models, env, dataset, kwargs_um):
    with open(ensemble_models.PREDICTION_MAT_PATH, "rb") as file:
        predicted_mat = pickle.load(file)
    env_data = "KuaiRand_Pure" if args.env == "KuaiRand-v0" else "NowPlaying"
    path = f"./data/{env_data}/data_raw/"
    # item_popularity = dataset.get_item_popularity()
    item_popularity = pd.read_csv(f"{path}item_popularity.csv")['popularity'].to_numpy()

    kwargs = {
        "ensemble_models": ensemble_models,
        "env_task_class": type(env),
        "task_env_param": kwargs_um,
        "task_name": args.env,
        "predicted_mat": predicted_mat,

        "item_popularity": item_popularity,
        "lambda_temper": args.lambda_temper,
    }

    train_envs = DummyVectorEnv(
        [lambda: SAC4IRSimulatedEnv(**kwargs) for _ in range(args.training_num)])

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    train_envs.seed(args.seed)

    return train_envs

def main(args):
    # %% 1. Prepare the saved path.
    MODEL_SAVE_PATH, logger_path = prepare_dir_log(args)

    # %% 2. Prepare user model and environment
    ensemble_models = prepare_user_model(args)
    env, dataset, kwargs_um = get_true_env(args)
    train_envs = prepare_train_envs_local(args, ensemble_models, env, dataset, kwargs_um)
    test_envs_dict = prepare_test_envs(args, env, kwargs_um)

    # %% 3. Setup policy
    state_tracker = setup_state_tracker(args, ensemble_models, env, train_envs, test_envs_dict)
    policy, train_collector, test_collector_set, optim = setup_policy_model(args, state_tracker, train_envs,
                                                                            test_envs_dict)

    # %% 4. Learn policy
    learn_policy(args, env, dataset, policy, train_collector, test_collector_set, state_tracker, optim, MODEL_SAVE_PATH,
                 logger_path, trainer="offpolicy")


if __name__ == '__main__':
    trainer = "offpolicy"
    args_all = get_args_all(trainer)
    args = get_env_args(args_all)
    args_DDPG = get_args_SAC4IR()
    args_all.__dict__.update(args.__dict__)
    args_all.__dict__.update(args_DDPG.__dict__)
    try:
        main(args_all)
    except Exception as e:
        var = traceback.format_exc()
        print(var)
        logzero.logger.error(var)
