from typing import Any, Dict, List, Optional, Type, Union

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from tianshou.data import Batch, ReplayBuffer, to_torch_as
from tianshou.policy import BasePolicy
from tianshou.utils.net.common import ActorCritic
from tianshou.utils import RunningMeanStd


class HA2CCPolicy(BasePolicy):
    def __init__(
            self,
            h_actor: torch.nn.Module,
            l_actor: torch.nn.Module,
            h_critic: torch.nn.Module,
            l_critic: torch.nn.Module,
            h_optim: torch.optim.Optimizer,
            l_optim: torch.optim.Optimizer,
            h_dist_fn: Type[torch.distributions.Distribution],
            l_dist_fn: Type[torch.distributions.Distribution],
            h_state_tracker=None,
            l_state_tracker=None,
            vf_coef: float = 0.5,
            ent_coef: float = 0.01,
            max_grad_norm: Optional[float] = None,
            discount_factor: float = 0.99,
            gae_lambda: float = 0.95,
            max_batchsize: int = 256,
            action_scaling: bool = True,
            action_bound_method: str = "clip",
            reward_normalization: bool = False,
            deterministic_eval: bool = False,
            device=None,
            item_types=None,
            **kwargs: Any
    ) -> None:
        super().__init__(
            action_scaling=action_scaling,
            action_bound_method=action_bound_method,
            **kwargs
        )
        self.h_actor = h_actor
        self.l_actor = l_actor
        self.h_critic = h_critic
        self.l_critic = l_critic
        self.h_optim = h_optim
        self.l_optim = l_optim
        self.h_dist_fn = h_dist_fn
        self.l_dist_fn = l_dist_fn
        self.h_state_tracker = h_state_tracker
        self.l_state_tracker = l_state_tracker
        self._h_actor_critic = ActorCritic(self.h_actor, self.h_critic)
        self._l_actor_critic = ActorCritic(self.l_actor, self.l_critic)

        assert 0.0 <= gae_lambda <= 1.0, "GAE lambda should be in [0, 1]."
        self._gamma = discount_factor
        self._lambda = gae_lambda
        self._weight_vf = vf_coef
        self._weight_ent = ent_coef
        self._grad_norm = max_grad_norm
        self._rew_norm = reward_normalization
        self.ret_rms = RunningMeanStd()
        self._eps = 1e-8
        self._batch_size = max_batchsize
        self._deterministic_eval = deterministic_eval
        self.device = device

        self.n_items = h_state_tracker.num_item
        self.emb_dim = h_state_tracker.emb_dim
        item_index = np.expand_dims(np.arange(self.n_items), -1)  # [n_items, 1]
        self.item_embs = self.h_state_tracker.get_embedding(item_index, "action")
        self.item_types = torch.tensor(item_types).to(self.device).unsqueeze(0).expand(2048, -1)

    def set_eps(self, eps: float) -> None:
        """Set the eps for epsilon-greedy exploration."""
        self.eps = eps

    # Hierarchical exploration
    def forward(
            self,
            batch: Batch,
            buffer: Optional[ReplayBuffer],
            indices: np.ndarray = None,
            is_obs=None,
            is_train=True,
            state: Optional[Union[dict, Batch, np.ndarray]] = None,
            use_batch_in_statetracker=False,
            c_distribution=None,
            **kwargs: Any,
    ) -> Batch:
        # The high-level agent takes actions
        h_obs_emb = self.h_state_tracker(buffer=buffer, indices=indices, 
                                        is_obs=is_obs, batch=batch,  is_train=is_train,
                                        use_batch_in_statetracker=use_batch_in_statetracker)
        c_dis = torch.tensor(c_distribution).to(self.device)
        h_logits, _ = self.h_actor(torch.cat([h_obs_emb, c_dis], dim=-1))

        if isinstance(h_logits, tuple):
            h_dist = self.h_dist_fn(*h_logits)
        else:
            h_dist = self.h_dist_fn(h_logits)
        h_act = h_dist.sample() # [B, emb_dim] 
        h_act = torch.clamp(h_act, min=1e-4) 
  
    
        # The low-level agents take actions
        l_obs_emb = self.l_state_tracker(buffer=buffer, indices=indices, is_obs=is_obs,
                                         batch=batch, is_train=is_train,
                                         use_batch_in_statetracker=use_batch_in_statetracker)
        
        l_logits, _ = self.l_actor(torch.cat([l_obs_emb, c_dis, h_act], dim=-1))
        # Apply the action weights of the upper-level agent to the lower-level actions
        B = l_logits.size(0)
        type_indices = self.item_types[:B].to(self.device)  # shape: [B, item_num]
        type_weights = h_act.gather(dim=1, index=type_indices).to(self.device)  # shape: [B, item_num]
        l_logits = l_logits * type_weights  # shape: [B, item_num]
        l_logits = nn.functional.relu(l_logits)
        if self.action_type == "discrete":
            if is_obs:
                l_logits = l_logits * batch.mask
            else:
                l_logits = l_logits * batch.next_mask

        if isinstance(l_logits, tuple):
            l_dist = self.l_dist_fn(*l_logits)
        else:
            l_dist = self.l_dist_fn(l_logits)
        l_act = None
        if self._deterministic_eval and not self.training:
            if self.action_type == "discrete":
                l_act = l_logits.argmax(-1)
            elif self.action_type == "continuous":
                l_act = l_logits[0]
        else:
            l_act = l_dist.sample()

        return Batch(h_act=h_act, l_act=l_act, h_dist=h_dist, l_dist=l_dist)

    def process_fn(
            self, batch: Batch, buffer: ReplayBuffer, indices: np.ndarray
    ) -> Batch:
        batch = self._compute_returns(batch, buffer, indices)
        batch.h_act = to_torch_as(batch.h_act, batch.h_v_s)
        batch.l_act = to_torch_as(batch.act, batch.l_v_s)
        return batch

    def _compute_returns(
            self, batch: Batch, buffer: ReplayBuffer, indices: np.ndarray
    ) -> Batch:
        # Calculate the information of the high-level intelligent agent
        v_s, v_s_ = [], []
        with torch.no_grad():
            for minibatch in batch.split(self._batch_size, shuffle=False, merge_last=True):
                obs_emb = self.h_state_tracker(buffer=buffer, indices=minibatch.indices, is_obs=True)
                c_dis = torch.tensor(minibatch.c_distribution).to(self.device).squeeze()
                v_s.append(self.h_critic(torch.cat([obs_emb, c_dis], dim=-1)))
                next_indices = self._buffer.next(minibatch.indices)
                done = torch.tensor(buffer.terminated[next_indices]).to(self.device).squeeze()
                h_obs_next_emb = self.h_state_tracker(buffer=buffer, indices=next_indices, is_obs=True)
                next_c_dis = torch.tensor(buffer.c_distribution[next_indices]).to(self.device)
                target_v = self.h_critic(torch.cat([h_obs_next_emb, next_c_dis], dim=-1)).squeeze() * (~done) * self._gamma + torch.tensor(
                    buffer[minibatch.indices].h_rew).to(self.device).squeeze()
                v_s_.append(target_v)

        batch.h_v_s = torch.cat(v_s, dim=0).flatten()  # old value
        batch.h_v_s_ = torch.cat(v_s_, dim=0).flatten()  # target value
        batch.h_adv = batch.h_v_s_ - batch.h_v_s.detach()

        # Calculate the information of the low-level agents
        v_s, v_s_ = [], []
        with torch.no_grad():
            for minibatch in batch.split(self._batch_size, shuffle=False, merge_last=True):
                obs_emb = self.l_state_tracker(buffer=buffer, indices=minibatch.indices, is_obs=True)
                h_act = torch.tensor(minibatch.h_act).to(self.device)
                c_dis = torch.tensor(minibatch.c_distribution).to(self.device).squeeze()
                v_s.append(self.l_critic(torch.cat([obs_emb, c_dis, h_act], dim=-1)))
                next_indices = self._buffer.next(minibatch.indices)
                next_c_dis = torch.tensor(buffer[next_indices].c_distribution).to(self.device)
                next_h_act = torch.tensor(buffer[next_indices].h_act).to(self.device)
                obs_next_emb = self.l_state_tracker(buffer=buffer, indices=minibatch.indices, is_obs=False)
                v_s_.append(self.l_critic(torch.cat([obs_next_emb, next_c_dis, next_h_act], dim=-1)))
        batch.l_v_s = torch.cat(v_s, dim=0).flatten()  # old value
        v_s = batch.l_v_s.cpu().numpy()
        v_s_ = torch.cat(v_s_, dim=0).flatten().cpu().numpy()
        if self._rew_norm:  # unnormalize v_s & v_s_
            v_s = v_s * np.sqrt(self.ret_rms.var + self._eps)
            v_s_ = v_s_ * np.sqrt(self.ret_rms.var + self._eps)
        unnormalized_returns, advantages = self.compute_episodic_return(
            batch,
            buffer,
            indices,
            v_s_,
            v_s,
            gamma=self._gamma,
            gae_lambda=self._lambda
        )
        if self._rew_norm:
            batch.returns = unnormalized_returns / \
                            np.sqrt(self.ret_rms.var + self._eps)
            self.ret_rms.update(unnormalized_returns)
        else:
            batch.returns = unnormalized_returns
        batch.returns = to_torch_as(batch.returns, batch.l_v_s)
        batch.l_adv = to_torch_as(advantages, batch.l_v_s)
        return batch

    # Hierarchical learning
    def learn(  # type: ignore
            self, batch: Batch, batch_size: int, repeat: int, **kwargs: Any
    ) -> Dict[str, List[float]]:
        h_losses, losses, actor_losses, vf_losses, ent_losses = [], [], [], [], []
        h_optim_RL, h_optim_state = self.h_optim
        l_optim_RL, l_optim_state = self.l_optim
        for _ in range(repeat):
            for minibatch in batch.split(batch_size, merge_last=True):
                # Update the low-level agent
                c_dis = torch.tensor(minibatch.c_distribution).to(self.device)
                l_dist = self(minibatch, self._buffer, indices=minibatch.indices, is_obs=True, c_distribution=c_dis).l_dist
                l_log_prob = l_dist.log_prob(minibatch.l_act)
                l_log_prob = l_log_prob.reshape(len(minibatch.l_adv), -1).transpose(0, 1)
                idx = ~torch.isinf(l_log_prob)

                l_actor_loss = -(l_log_prob * minibatch.l_adv)[idx].mean()
                # calculate loss for critic
                h_act = torch.tensor(minibatch.h_act.clone().detach()).to(self.device)
                c_dis = torch.tensor(minibatch.c_distribution).to(self.device)
                l_obs_emb = self.l_state_tracker(self._buffer, minibatch.indices, is_obs=True)
                l_value = self.l_critic(torch.cat([l_obs_emb, c_dis ,h_act], dim=-1)).flatten()
                l_vf_loss = F.mse_loss(minibatch.returns.flatten(), l_value)
                # calculate regularization and overall loss
                l_ent_loss = l_dist.entropy().mean()
                l_loss = l_actor_loss + self._weight_vf * l_vf_loss - self._weight_ent * l_ent_loss
                
                l_optim_RL.zero_grad()
                l_optim_state.zero_grad()
                l_loss.backward()
                if self._grad_norm:  # clip large gradient
                    nn.utils.clip_grad_norm_(
                        self._l_actor_critic.parameters(), max_norm=self._grad_norm
                    )
                    nn.utils.clip_grad_norm_(
                        self.l_state_tracker.parameters(), max_norm=self._grad_norm
                    )
                l_optim_RL.step()
                l_optim_state.step()

                actor_losses.append(l_actor_loss.item())
                vf_losses.append(l_vf_loss.item())
                ent_losses.append(l_ent_loss.item())
                losses.append(l_loss.item())
        
        for _ in range(repeat):
            for minibatch in batch.split(batch_size, merge_last=True):
                # Update the high-level agent
                # calculate loss for actor
                c_dis = torch.tensor(minibatch.c_distribution).to(self.device)
                h_dist = self(minibatch, self._buffer, indices=minibatch.indices, is_obs=True, c_distribution=c_dis).h_dist
                h_log_prob = h_dist.log_prob(minibatch.h_act)
                h_log_prob = h_log_prob.reshape(len(minibatch.h_adv), -1).transpose(0, 1)
                idx = ~torch.isinf(h_log_prob)
                h_actor_loss = -((h_log_prob * minibatch.h_adv))[idx].mean()
                # calculate loss for critic
                h_obs_emb = self.h_state_tracker(self._buffer, minibatch.indices, is_obs=True)
                h_value = self.h_critic(torch.cat([h_obs_emb, c_dis], dim=-1)).flatten()
                h_vf_loss = ((minibatch.h_v_s_.flatten() - h_value).pow(2)).mean()

                # calculate regularization and overall loss
                h_ent_loss = (h_dist.entropy()).mean()
                h_loss = h_actor_loss + self._weight_vf * h_vf_loss - self._weight_ent * h_ent_loss
                h_optim_RL.zero_grad()
                h_optim_state.zero_grad()
                h_loss.backward()
                if self._grad_norm:  # clip large gradient
                    nn.utils.clip_grad_norm_(
                        self._h_actor_critic.parameters(), max_norm=self._grad_norm
                    )
                    nn.utils.clip_grad_norm_(
                        self.h_state_tracker.parameters(), max_norm=self._grad_norm
                    )
                h_optim_RL.step()
                h_optim_state.step()
                h_losses.append(h_loss.item())


        return {
            "h_loss": h_losses,
            "loss": losses,
            "loss/actor": actor_losses,
            "loss/vf": vf_losses,
            "loss/ent": ent_losses,
        }
    
    def exploration_noise(self, h_act, l_act):
        return l_act
