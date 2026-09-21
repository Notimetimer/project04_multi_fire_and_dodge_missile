import numpy as np
import torch
from torch import nn
from torch.distributions import Bernoulli, Categorical
import torch.nn.functional as F


BRANCH_A = 0
BRANCH_B = 1
BRANCH_C = 2
BRANCH_D = 3
HORIZONTAL_ACTIONS = (0, 1, 3, 5)


class ActorMLPHeads(nn.Module):
    def __init__(self, hidden_dims=(256, 256), fixed_dim=28, enm_shape=(2, 10), ally_shape=(1, 9)):
        super().__init__()
        self.fixed_dim = fixed_dim
        self.enm_shape = enm_shape
        self.ally_shape = ally_shape
        self.enm_num = enm_shape[0]
        self.input_dim = fixed_dim + int(np.prod(enm_shape)) + int(np.prod(ally_shape))

        layers = []
        previous_dim = self.input_dim
        for hidden_dim in hidden_dims:
            layers.extend((nn.Linear(previous_dim, hidden_dim), nn.ReLU()))
            previous_dim = hidden_dim
        self.backbone = nn.Sequential(*layers)
        self.branch_head = nn.Linear(previous_dim, 4)
        branch_condition_dim = previous_dim + 4
        self.a_head = nn.Linear(branch_condition_dim, 2)
        self.target_head = nn.Linear(branch_condition_dim, self.enm_num)
        c_condition_dim = branch_condition_dim + self.enm_num
        self.c1_head = nn.Linear(c_condition_dim, 3)
        self.c2_head = nn.Linear(c_condition_dim, len(HORIZONTAL_ACTIONS))
        self.c3_head = nn.Linear(c_condition_dim, 1)

    def forward(self, x, branch=None, target=None):
        features = self.backbone(x)
        outputs = {
            'features': features,
            'branch_logits': self.branch_head(features),
        }
        if branch is None:
            return outputs

        branch_one_hot = F.one_hot(branch, num_classes=4).to(features.dtype)
        branch_features = torch.cat((features, branch_one_hot), dim=-1)
        outputs['a_logits'] = self.a_head(branch_features)
        outputs['target_logits'] = self.target_head(branch_features)

        if target is None:
            return outputs
        target_one_hot = F.one_hot(target, num_classes=self.enm_num).to(features.dtype)
        c_features = torch.cat((branch_features, target_one_hot), dim=-1)
        outputs['c1_logits'] = self.c1_head(c_features)
        outputs['c2_logits'] = self.c2_head(c_features)
        outputs['c3_logits'] = self.c3_head(c_features).squeeze(-1)
        return outputs


class ActorMLPWrapper(nn.Module):
    def __init__(self, network=None, hidden_dims=(256, 256), fixed_dim=28,
                 enm_shape=(2, 10), ally_shape=(1, 9), device='cpu'):
        super().__init__()
        self.net = network or ActorMLPHeads(
            hidden_dims=hidden_dims,
            fixed_dim=fixed_dim,
            enm_shape=enm_shape,
            ally_shape=ally_shape,
        ).to(device)
        self.fixed_dim = fixed_dim
        self.enm_shape = enm_shape
        self.ally_shape = ally_shape
        self.enm_num = self.net.enm_num
        self.input_dim = self.net.input_dim

    @property
    def device(self):
        return next(self.parameters()).device

    def flatten_agent_inputs(self, agent_inputs):
        if isinstance(agent_inputs, torch.Tensor):
            tensor = agent_inputs.to(device=self.device, dtype=torch.float32)
            if tensor.ndim == 1:
                tensor = tensor.unsqueeze(0)
            if tensor.shape[-1] != self.input_dim:
                raise ValueError(f'Expected flattened input dimension {self.input_dim}, got {tensor.shape[-1]}')
            return tensor
        if not isinstance(agent_inputs, dict):
            raise TypeError('agent_inputs must be a dict or a flattened torch.Tensor')

        parts = []
        batch_size = None
        specs = (('fixed', (self.fixed_dim,)), ('enm', self.enm_shape), ('ally', self.ally_shape))
        for key, expected_shape in specs:
            value = torch.as_tensor(agent_inputs[key], dtype=torch.float32, device=self.device)
            if tuple(value.shape) == tuple(expected_shape):
                value = value.unsqueeze(0)
            elif value.ndim != len(expected_shape) + 1 or tuple(value.shape[1:]) != tuple(expected_shape):
                raise ValueError(f'agent_inputs[{key!r}] must have shape {expected_shape} or (B, {expected_shape})')
            if batch_size is None:
                batch_size = value.shape[0]
            elif value.shape[0] != batch_size:
                raise ValueError('All agent input components must have the same batch size')
            parts.append(value.flatten(start_dim=1))
        return torch.cat(parts, dim=-1)

    @staticmethod
    def _as_samples(value, batch_size, name):
        if isinstance(value, (list, tuple)):
            if len(value) != batch_size:
                raise ValueError(f'{name} batch size does not match agent_inputs')
            return list(value)
        if batch_size != 1 or not isinstance(value, dict):
            raise TypeError(f'{name} must be a dict for one sample or a list of dicts for a batch')
        return [value]

    @staticmethod
    def _warning(rule_input):
        warning = rule_input.get('RWR_infos', {}).get('warning', 0)
        return bool(np.asarray(warning).astype(bool).any())

    @staticmethod
    def _threat_delta_psi(rule_input):
        rwr = rule_input.get('RWR_infos', {})
        return float(np.arctan2(rwr.get('sin_threat_delta_psi', 0.0),
                                rwr.get('cos_threat_delta_psi', 1.0)))

    def build_rule_masks(self, rule_inputs, mask_inputs, batch_size):
        rules = self._as_samples(rule_inputs, batch_size, 'rule_inputs')
        masks = self._as_samples(mask_inputs, batch_size, 'mask_inputs')
        branch_masks = torch.zeros(batch_size, 4, dtype=torch.bool, device=self.device)
        target_masks = torch.zeros(batch_size, self.enm_num, dtype=torch.bool, device=self.device)
        threat_delta_psi = torch.zeros(batch_size, dtype=torch.float32, device=self.device)

        for index, (rule_input, mask_input) in enumerate(zip(rules, masks)):
            enemy_mask = np.asarray(mask_input.get('enm', []), dtype=bool).reshape(-1)
            if enemy_mask.size != self.enm_num:
                raise ValueError(f"mask_inputs['enm'] must contain {self.enm_num} entries")
            target_masks[index] = torch.as_tensor(enemy_mask, dtype=torch.bool, device=self.device)
            warning = self._warning(rule_input)
            ego_alive = bool(mask_input.get('ego', 1))
            if not ego_alive:
                branch_masks[index, BRANCH_D] = True
            elif warning:
                branch_masks[index, BRANCH_A] = True
            elif enemy_mask.any():
                branch_masks[index, BRANCH_C] = True
            else:
                branch_masks[index, BRANCH_D] = True
            threat_delta_psi[index] = self._threat_delta_psi(rule_input)
        return branch_masks, target_masks, threat_delta_psi

    @staticmethod
    def _masked_logits(logits, mask):
        if torch.any(~mask.any(dim=-1)):
            raise ValueError('Every categorical head must have at least one valid action')
        return logits.masked_fill(~mask, torch.finfo(logits.dtype).min)

    @staticmethod
    def _choose(distribution, explore):
        return distribution.sample() if explore else distribution.probs.argmax(dim=-1)

    def forward(self, agent_inputs, rule_inputs, mask_inputs):
        flat_inputs = self.flatten_agent_inputs(agent_inputs)
        branch_mask, target_mask, threat_delta_psi = self.build_rule_masks(
            rule_inputs, mask_inputs, flat_inputs.shape[0])
        outputs = self.net(flat_inputs)
        outputs.update({
            'flat_inputs': flat_inputs,
            'branch_logits': self._masked_logits(outputs['branch_logits'], branch_mask),
            'branch_mask': branch_mask,
            'target_mask': target_mask,
            'threat_delta_psi': threat_delta_psi,
        })
        return outputs

    def get_action(self, agent_inputs, rule_inputs, mask_inputs, explore=True):
        outputs = self.forward(agent_inputs, rule_inputs, mask_inputs)
        branch_dist = Categorical(logits=outputs['branch_logits'])
        branch = self._choose(branch_dist, explore)
        fallback_target = outputs['target_mask'].to(torch.int64).argmax(dim=-1)
        conditioned = self.net(outputs['flat_inputs'], branch=branch)
        target_logits = self._masked_logits(
            conditioned['target_logits'],
            outputs['target_mask'] | (branch != BRANCH_C).unsqueeze(-1))
        target_dist = Categorical(logits=target_logits)
        target = self._choose(target_dist, explore)
        conditioned = self.net(outputs['flat_inputs'], branch=branch, target=target)

        a_dist = Categorical(logits=conditioned['a_logits'])
        c1_dist = Categorical(logits=conditioned['c1_logits'])
        c2_dist = Categorical(logits=conditioned['c2_logits'])
        c3_dist = Bernoulli(logits=conditioned['c3_logits'])
        a_action = self._choose(a_dist, explore)
        c1_action = self._choose(c1_dist, explore)
        c2_action = self._choose(c2_dist, explore)
        c3_action = c3_dist.sample().to(torch.long) if explore else (c3_dist.probs >= 0.5).to(torch.long)
        raw_actions = {
            'branch': branch,
            'a': a_action,
            'target': target,
            'c1': c1_action,
            'c2': c2_action,
            'c3': c3_action,
        }
        env_actions = self.to_env_actions(raw_actions, outputs['threat_delta_psi'])
        diagnostics = {
            'branch_probs': branch_dist.probs,
            'target_probs': target_dist.probs,
            'a_probs': a_dist.probs,
            'c1_probs': c1_dist.probs,
            'c2_probs': c2_dist.probs,
            'c3_probs': c3_dist.probs,
        }
        return env_actions, raw_actions, diagnostics

    def to_env_actions(self, actions, threat_delta_psi):
        results = []
        batch_size = actions['branch'].shape[0]
        for index in range(batch_size):
            branch = int(actions['branch'][index].item())
            target = int(actions['target'][index].item())
            if branch == BRANCH_A:
                if int(actions['a'][index].item()) == 0:
                    maneuver = [2, 3]
                else:
                    maneuver = [2, 2 if threat_delta_psi[index].item() > 0 else 4]
                fire = False
            elif branch == BRANCH_C:
                maneuver = [int(actions['c1'][index].item()), HORIZONTAL_ACTIONS[int(actions['c2'][index].item())]]
                fire = bool(actions['c3'][index].item())
            else:
                maneuver = [1, 0]
                fire = False
            results.append({
                'branch': ('A', 'B', 'C', 'D')[branch],
                'flight_action': {'do': maneuver, 'at': target},
                'fire_action': {'do': fire, 'at': target},
                'radar_action': 1 if branch == BRANCH_C else 0,
            })
        return results[0] if batch_size == 1 else results

    def evaluate_actions(self, agent_inputs, rule_inputs, mask_inputs, actions):
        outputs = self.forward(agent_inputs, rule_inputs, mask_inputs)
        branch = actions['branch'].to(device=self.device, dtype=torch.long).reshape(-1)
        target = actions['target'].to(device=self.device, dtype=torch.long).reshape(-1)
        a_action = actions['a'].to(device=self.device, dtype=torch.long).reshape(-1)
        c1_action = actions['c1'].to(device=self.device, dtype=torch.long).reshape(-1)
        c2_action = actions['c2'].to(device=self.device, dtype=torch.long).reshape(-1)
        c3_action = actions['c3'].to(device=self.device, dtype=torch.float32).reshape(-1)

        conditioned = self.net(outputs['flat_inputs'], branch=branch, target=target)
        target_logits = self._masked_logits(
            conditioned['target_logits'],
            outputs['target_mask'] | (branch != BRANCH_C).unsqueeze(-1))
        branch_dist = Categorical(logits=outputs['branch_logits'])
        target_dist = Categorical(logits=target_logits)
        a_dist = Categorical(logits=conditioned['a_logits'])
        c1_dist = Categorical(logits=conditioned['c1_logits'])
        c2_dist = Categorical(logits=conditioned['c2_logits'])
        c3_dist = Bernoulli(logits=conditioned['c3_logits'])

        is_a = (branch == BRANCH_A).to(torch.float32)
        is_c = (branch == BRANCH_C).to(torch.float32)
        log_prob = branch_dist.log_prob(branch)
        log_prob = log_prob + is_a * a_dist.log_prob(a_action)
        log_prob = log_prob + is_c * (
            target_dist.log_prob(target) + c1_dist.log_prob(c1_action)
            + c2_dist.log_prob(c2_action) + c3_dist.log_prob(c3_action))
        entropy = branch_dist.entropy() + is_a * a_dist.entropy()
        entropy = entropy + is_c * (
            target_dist.entropy() + c1_dist.entropy() + c2_dist.entropy() + c3_dist.entropy())
        return log_prob.unsqueeze(-1), entropy.unsqueeze(-1), outputs
