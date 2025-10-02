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
from src.tianshou.tianshou.policy import A2CPolicy

# from util.upload import my_upload
import logzero

try:
    import envpool
except ImportError:
    envpool = None


def get_args_DORL():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, default="DORL")
    parser.add_argument('--vf-coef', type=float, default=0.5)
    parser.add_argument('--ent-coef', type=float, default=0.0)
    parser.add_argument('--max-grad-norm', type=float, default=None)
    parser.add_argument('--gae-lambda', type=float, default=1.)
    parser.add_argument('--rew-norm', action="store_true", default=False)

    # Env
    parser.add_argument('--is_exposure_intervention', dest='use_exposure_intervention', action='store_true')
    parser.add_argument('--no_exposure_intervention', dest='use_exposure_intervention', action='store_false')
    parser.set_defaults(use_exposure_intervention=False)

    parser.add_argument('--is_feature_level', dest='feature_level', action='store_true')
    parser.add_argument('--no_feature_level', dest='feature_level', action='store_false')
    parser.set_defaults(feature_level=True)

    parser.add_argument('--is_sorted', dest='is_sorted', action='store_true')
    parser.add_argument('--no_sorted', dest='is_sorted', action='store_false')
    parser.set_defaults(is_sorted=True)

    parser.add_argument("--entropy_window", type=int, nargs="*", default=[1, 2])
    parser.add_argument("--version", type=str, default="v1")
    parser.add_argument('--tau', default=0, type=float)
    parser.add_argument('--gamma_exposure', default=10, type=float)
    parser.add_argument("--which_tracker", type=str, default="avg") 
    parser.add_argument('--lambda_entropy', default=5, type=float)
    parser.add_argument('--lambda_variance', default=0., type=float)

    parser.add_argument("--read_message", type=str, default="UM")
    parser.add_argument("--message", type=str, default="DORL")

    args = parser.parse_known_args()[0]
    return args


def get_entropy(mylist, need_count=True):
    if len(mylist) <= 1:
        return 1
    if need_count:
        cnt_dict = Counter(mylist)
    else:
        cnt_dict = mylist
    prob = np.array(list(cnt_dict.values())) / sum(cnt_dict.values())
    log_prob = np.log2(prob)
    entropy = - np.sum(log_prob * prob) / np.log2(len(cnt_dict))
    # entropy = - np.sum(log_prob * prob) / np.log2(len(cnt_dict) + 1)
    return entropy

def get_save_entropy_mat(dataset, entropy_window, ent_savepath, feature_level=True, is_sorted=True):
    df_train, df_user, df_item, list_feat = dataset.get_train_data()
    # assert list_feat is not None
    if args.env !='NowPlaying-v0':
        v = df_item["tags"]
    else:
        v = df_item["key"]
    map_item_feat = dict(zip(df_item.index, v)) if feature_level else None

    # if os.path.exists(ent_savepath):
    #     map_entropy = pickle.load(open(ent_savepath, 'rb'))
    #     return map_entropy, map_item_feat

    # num_item = df_train["item_id"].nunique()
    if not "timestamp" in df_train.columns:
        df_train.rename(columns={"time_ms": "timestamp"}, inplace=True)

    map_entropy = None

    # entropy_user = None
    # if 0 in entropy_window:
    #     df_train = df_train.sort_values("user_id")
    #     interaction_list = df_train[["user_id", "item_id"]].groupby("user_id").agg(list)
    #     entropy_user = interaction_list["item_id"].map(partial(get_entropy))
    #
    #     savepath = os.path.join(self.Entropy_PATH, "user_entropy.csv")
    #     entropy_user.to_csv(savepath, index=True)

    if len(set(entropy_window) - set([0])):

        if "timestamp" in df_train.columns:
            df_uit = df_train[["user_id", "item_id", "timestamp"]].sort_values(["user_id", "timestamp"])
        else:
            df_uit = df_train[["user_id", "item_id"]].sort_values(["user_id"])

        map_hist_count = defaultdict(lambda: defaultdict(int))
        lastuser = int(-1)

        def update_map(map_hist_count, hist_tra, item, require_len, is_sort=True):
            if len(hist_tra) < require_len:
                return

            history = tuple(sorted(hist_tra[-require_len:]) if is_sort else hist_tra[-require_len:])
            map_hist_count[history][item] += 1

        hist_tra = []

        # for k, (user, item, time) in tqdm(df_uit.iterrows(), total=len(df_uit), desc="count frequency..."):
        for row in tqdm(df_uit.to_numpy(), total=len(df_uit), desc="count frequency..."):
            if "timestamp" in df_train.columns:
                (user, item, time) = row
            else:
                (user, item) = row
            user = int(user)
            item = int(item)
            if feature_level:
                features = map_item_feat[item]

            if user != lastuser:
                lastuser = user
                hist_tra = []

            if feature_level:
                for feat in [features]:
                    for require_len in set(entropy_window) - set([0]):
                        hist_feats = get_features_of_last_n_items_features(require_len, hist_tra, map_item_feat, is_sort=is_sorted)
                        for hist_feat_list in hist_feats:
                            update_map(map_hist_count, hist_feat_list, feat, require_len)
            else: # item level
                for require_len in set(entropy_window) - set([0]):
                    update_map(map_hist_count, hist_tra, item, require_len)
            hist_tra.append(item)

        map_entropy = {}
        for k, v in tqdm(map_hist_count.items(), total=len(map_hist_count), desc="compute entropy..."):
            map_entropy[k] = get_entropy(v, need_count=False)

        # savepath = os.path.join(self.Entropy_PATH, "map_entropy.pickle")
        # pickle.dump(map_entropy, open(ent_savepath, 'wb'))

        # print(map_hist_count)
    return map_entropy, map_item_feat


def prepare_train_envs(args, ensemble_models, env, dataset, kwargs_um):
    entropy_dict = dict()
    # if 0 in args.entropy_window:
    #     entropy_path = os.path.join(ensemble_models.Entropy_PATH, "user_entropy.csv")
    #     entropy = pd.read_csv(entropy_path)
    #     entropy.set_index("user_id", inplace=True)
    #     entropy_mat_0 = entropy.to_numpy().reshape([-1])
    #     entropy_dict.update({"on_user": entropy_mat_0})
    if len(set(args.entropy_window) - set([0])):
        savepath = os.path.join(ensemble_models.Entropy_PATH, "map_entropy.pickle")
        map_entropy, map_item_feat = get_save_entropy_mat(dataset, args.entropy_window, savepath, args.feature_level, args.is_sorted)

        entropy_dict.update({"map": map_entropy})

    entropy_set = set(args.entropy_window)
    entropy_min = 0
    entropy_max = 0
    if len(entropy_set):
        for entropy_term in entropy_set:
            entropy_min += min([v for k, v in entropy_dict["map"].items() if len(k) == entropy_term] + [1])
            entropy_max += max([v for k, v in entropy_dict["map"].items() if len(k) == entropy_term] + [1])

    with open(ensemble_models.PREDICTION_MAT_PATH, "rb") as file:
        predicted_mat = pickle.load(file)

    alpha_u, beta_i = None, None  ## TODO

    with open(ensemble_models.VAR_MAT_PATH, "rb") as file:
        maxvar_mat = pickle.load(file)

    kwargs = {
        "ensemble_models": ensemble_models,
        "env_task_class": type(env),
        "task_env_param": kwargs_um,
        "task_name": args.env,
        "predicted_mat": predicted_mat,
        "maxvar_mat": maxvar_mat,
        "lambda_variance": args.lambda_variance,
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
        "entropy_min": entropy_min,
        "entropy_max": entropy_max,
        "feature_level": args.feature_level,
        "map_item_feat": map_item_feat,
        "is_sorted": args.is_sorted
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
    policy = A2CPolicy(
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
        remove_recommended_ids = args.remove_recommended_ids
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
    policy, train_collector, test_collector_set, optim = setup_policy_model(args, state_tracker, train_envs, test_envs_dict)

    # %% 4. Learn policy
    learn_policy(args, env, dataset, policy, train_collector, test_collector_set, state_tracker, optim, MODEL_SAVE_PATH,
                 logger_path, trainer="onpolicy")


if __name__ == '__main__':
    trainer = "onpolicy"
    args_all = get_args_all(trainer)
    args = get_env_args(args_all)
    args_DORL = get_args_DORL()
    args_all.__dict__.update(args.__dict__)
    args_all.__dict__.update(args_DORL.__dict__)
    try:
        main(args_all)
    except Exception as e:
        var = traceback.format_exc()
        print(var)
        logzero.logger.error(var)