from copy import deepcopy
from typing import Any, Dict, Optional, Tuple, Union

import numpy as np
import torch
from torch.distributions import Independent, Normal

from tianshou.data import Batch, ReplayBuffer
from tianshou.exploration import BaseNoise
from tianshou.policy import DDPGPolicy


class SACPolicy(DDPGPolicy):
    """Implementation of Soft Actor-Critic. arXiv:1812.05905.

    :param torch.nn.Module actor: the actor network following the rules in
        :class:`~tianshou.policy.BasePolicy`. (s -> logits)
    :param torch.optim.Optimizer actor_optim: the optimizer for actor network.
    :param torch.nn.Module critic1: the first critic network. (s, a -> Q(s, a))
    :param torch.optim.Optimizer critic1_optim: the optimizer for the first
        critic network.
    :param torch.nn.Module critic2: the second critic network. (s, a -> Q(s, a))
    :param torch.optim.Optimizer critic2_optim: the optimizer for the second
        critic network.
    :param float tau: param for soft update of the target network. Default to 0.005.
    :param float gamma: discount factor, in [0, 1]. Default to 0.99.
    :param (float, torch.Tensor, torch.optim.Optimizer) or float alpha: entropy
        regularization coefficient. Default to 0.2.
        If a tuple (target_entropy, log_alpha, alpha_optim) is provided, then
        alpha is automatically tuned.
    :param bool reward_normalization: normalize the reward to Normal(0, 1).
        Default to False.
    :param BaseNoise exploration_noise: add a noise to action for exploration.
        Default to None. This is useful when solving hard-exploration problem.
    :param bool deterministic_eval: whether to use deterministic action (mean
        of Gaussian policy) instead of stochastic action sampled by the policy.
        Default to True.
    :param bool action_scaling: whether to map actions from range [-1, 1] to range
        [action_spaces.low, action_spaces.high]. Default to True.
    :param str action_bound_method: method to bound action to range [-1, 1], can be
        either "clip" (for simply clipping the action) or empty string for no bounding.
        Default to "clip".
    :param Optional[gym.Space] action_space: env's action space, mandatory if you want
        to use option "action_scaling" or "action_bound_method". Default to None.
    :param lr_scheduler: a learning rate scheduler that adjusts the learning rate in
        optimizer in each policy.update(). Default to None (no lr_scheduler).

    .. seealso::

        Please refer to :class:`~tianshou.policy.BasePolicy` for more detailed
        explanation.
    """

    def __init__(
            self,
            actor: torch.nn.Module,
            actor_optim: torch.optim.Optimizer,
            critic1: torch.nn.Module,
            critic1_optim: torch.optim.Optimizer,
            critic2: torch.nn.Module,
            critic2_optim: torch.optim.Optimizer,
            state_tracker: torch.nn.Module,
            optim_state: Optional[torch.optim.Optimizer],
            tau: float = 0.005,
            gamma: float = 0.99,
            alpha: Union[float, Tuple[float, torch.Tensor, torch.optim.Optimizer]] = 0.2,
            reward_normalization: bool = False,
            estimation_step: int = 1,
            exploration_noise: Optional[BaseNoise] = None,
            deterministic_eval: bool = True,
            # 计算最大熵
            target_entropy_ratio: float = 0.8,
            n_actions: int = None,
            device = None,
            **kwargs: Any,
    ) -> None:
        print(gamma)
        super().__init__(
            None, None, None, None, None, tau, gamma, exploration_noise,
            reward_normalization, estimation_step, **kwargs,
        )
        self.actor, self.actor_optim = actor, actor_optim
        self.critic1, self.critic1_old = critic1, deepcopy(critic1)
        self.critic1_old.eval()
        self.critic1_optim = critic1_optim
        self.critic2, self.critic2_old = critic2, deepcopy(critic2)
        self.critic2_old.eval()
        self.critic2_optim = critic2_optim
        self.state_tracker = state_tracker
        self.optim_state = optim_state
        self.avg_prob = 1.0 / n_actions  # 计算平均概率
        self._target_entropy = -np.log(self.avg_prob) * target_entropy_ratio
        self._log_alpha = torch.zeros(1, requires_grad=True, device=device)
        self._alpha = self._log_alpha.exp()
        self._alpha_optim = torch.optim.Adam([self._log_alpha], lr=1e-4)

        self._is_auto_alpha = True
        #self._alpha: Union[float, torch.Tensor]
        # if isinstance(alpha, tuple):
        #     self._is_auto_alpha = True
        #     self._target_entropy, self._log_alpha, self._alpha_optim = alpha
        #     assert alpha[1].shape == torch.Size([1]) and alpha[1].requires_grad
        #     self._alpha = self._log_alpha.detach().exp()
        # else:
        #     self._alpha = alpha

        self._deterministic_eval = deterministic_eval
        self.__eps = np.finfo(np.float32).eps.item()

    def train(self, mode: bool = True) -> "SACPolicy":
        self.training = mode
        self.actor.train(mode)
        self.critic1.train(mode)
        self.critic2.train(mode)
        return self

    def sync_weight(self) -> None:
        self.soft_update(self.critic1_old, self.critic1, self.tau)
        self.soft_update(self.critic2_old, self.critic2, self.tau)

    def forward(  # type: ignore
            self,
            batch: Batch,
            buffer: Optional[ReplayBuffer],
            indices: np.ndarray = None,
            is_train = True,
            state: Optional[Union[dict, Batch, np.ndarray]] = None,
            input: str = "obs",
            use_batch_in_statetracker = False,
            **kwargs: Any,
    ) -> Batch:
        is_obs=True if input=="obs" else False
        obs_emb = self.state_tracker(buffer=buffer, indices=indices, is_obs=is_obs, batch=batch,
                                     is_train=is_train, use_batch_in_statetracker=use_batch_in_statetracker)
        logits, hidden = self.actor(obs_emb, state=state, info=batch.info)
        assert isinstance(logits, tuple)
        dist = Independent(Normal(*logits), 1)
        if self._deterministic_eval and not self.training:
            act = logits[0]
        else:
            act = dist.rsample()
        log_prob = dist.log_prob(act).unsqueeze(-1)
        entroy = dist.entropy()
        # apply correction for Tanh squashing when computing logprob from Gaussian
        # You can check out the original SAC paper (arXiv 1801.01290): Eq 21.
        # in appendix C to get some understanding of this equation.
        squashed_action = torch.tanh(act)
        log_prob = log_prob - torch.log((1 - squashed_action.pow(2)) +
                                        self.__eps).sum(-1, keepdim=True)
        return Batch(
            logits=logits,
            act=squashed_action,
            state=hidden,
            dist=dist,
            log_prob=log_prob,
            obs_emb=obs_emb,
            entroy=entroy
        )

    def _target_q(self, batch, buffer: ReplayBuffer, indices: np.ndarray) -> torch.Tensor:
        # batch = buffer[indices]  # batch.obs: s_{t+n}
        # obs_next_emb = self.state_tracker(buffer=buffer, indices=indices, is_obs=False)
        obs_next_result = self(batch, buffer, indices=indices, input="obs_next")
        act_ = obs_next_result.act
        obs_next_emb = obs_next_result.obs_emb
        target_q = torch.min(
            self.critic1_old(obs_next_emb, act_),
            self.critic2_old(obs_next_emb, act_),
        ) - self._alpha * obs_next_result.log_prob
        return target_q

    def learn(self, batch: Batch, **kwargs: Any) -> Dict[str, float]:
        # critic 1&2
        td1, critic1_loss = self._mse_optimizer(
            batch, self.critic1, self.critic1_optim
        )
        td2, critic2_loss = self._mse_optimizer(
            batch, self.critic2, self.critic2_optim
        )
        batch.weight = (td1 + td2) / 2.0  # prio-buffer

        # actor
        obs_result = self(batch, self._buffer, indices=batch.indices)
        act = obs_result.act
        obs_emb = obs_result.obs_emb
        current_q1a = self.critic1(obs_emb, act).flatten()
        current_q2a = self.critic2(obs_emb, act).flatten()
        actor_loss = (
                self._alpha * obs_result.log_prob.flatten() -
                torch.min(current_q1a, current_q2a)
        ).mean()
        self.actor_optim.zero_grad()
        self.optim_state.zero_grad()
        actor_loss.backward()
        self.actor_optim.step()
        self.optim_state.step()

        if self._is_auto_alpha:
            # please take a look at issue #258 if you'd like to change this line
            alpha_loss = self._log_alpha * (self._target_entropy - obs_result.entroy.detach()).mean()
            self._alpha_optim.zero_grad()
            alpha_loss.backward()
            self._alpha_optim.step()
            self._alpha = self._log_alpha.detach().exp()

        self.sync_weight()

        result = {
            "loss/actor": actor_loss.item(),
            "loss/critic1": critic1_loss.item(),
            "loss/critic2": critic2_loss.item(),
        }
        if self._is_auto_alpha:
            result["loss/alpha"] = alpha_loss.item()
            result["alpha"] = self._alpha.item()  # type: ignore

        return result
