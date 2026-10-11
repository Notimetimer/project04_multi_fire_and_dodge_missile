import numpy as np
import torch
from torch import nn
from torch.distributions import Bernoulli, Categorical
import torch.nn.functional as F


BRANCH_evade = 0
BRANCH_B = 1
BRANCH_attack = 2
BRANCH_idle = 3
HORIZONTAL_ACTIONS = (0, 1, 3, 5)
# 分支语义名：evade=规避(有告警), B=未定义, attack=攻击(有敌机), idle=待命/无动作
BRANCH_NAMES = ('evade', 'B', 'attack', 'idle')


# FIXME 多次重复调用forward方法，没有必要这样，应该改回一次调用的方式

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
        # 分支选择头
        self.branch_head = nn.Linear(previous_dim, 4)

        # 分支选择后的维度为什么要+4？
        branch_condition_dim = previous_dim + 4
        # 规避头
        self.evad_head = nn.Linear(branch_condition_dim, 2)
        # 目标选取头
        self.target_head = nn.Linear(branch_condition_dim, self.enm_num)
        # 
        c_condition_dim = branch_condition_dim + self.enm_num
        # 水平头
        self.hor_head = nn.Linear(c_condition_dim, len(HORIZONTAL_ACTIONS))
        # 火箭头
        self.fire_head = nn.Linear(c_condition_dim, 1)
        # 垂直头
        self.vert_head = nn.Linear(c_condition_dim, 3)

    def forward(self, x, branch=None, target=None):
        features = self.backbone(x)
        outputs = {
            'features': features,
            'branch_logits': self.branch_head(features),
        }
        # branch=None时，只返回分支选择头
        if branch is None:
            return outputs
        # 分支选择头
        branch_one_hot = F.one_hot(branch, num_classes=4).to(features.dtype)
        branch_features = torch.cat((features, branch_one_hot), dim=-1)
        outputs['evad_logits'] = self.evad_head(branch_features)
        # 目标选择头
        outputs['target_logits'] = self.target_head(branch_features)

        if target is None:
            return outputs
        # 目标选择头
        target_one_hot = F.one_hot(target, num_classes=self.enm_num).to(features.dtype)
        c_features = torch.cat((branch_features, target_one_hot), dim=-1)
        outputs['vert_logits'] = self.vert_head(c_features)
        outputs['hor_logits'] = self.hor_head(c_features)
        outputs['fire_logits'] = self.fire_head(c_features).squeeze(-1)
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

    def _unpack_inputs(self, agent_inputs, rule_inputs=None, mask_inputs=None):
        if isinstance(agent_inputs, (list, tuple)) and len(agent_inputs) > 0 and isinstance(agent_inputs[0], (list, tuple)) and len(agent_inputs[0]) == 3:
            rule_inputs = [x[1] for x in agent_inputs]
            mask_inputs = [x[2] for x in agent_inputs]
            agent_inputs = [x[0] for x in agent_inputs]
        elif isinstance(agent_inputs, (list, tuple)) and len(agent_inputs) == 3 and not isinstance(agent_inputs[0], (dict, torch.Tensor, np.ndarray)) and isinstance(agent_inputs[1], dict) and isinstance(agent_inputs[2], dict):
            agent_inputs, rule_inputs, mask_inputs = agent_inputs
        return agent_inputs, rule_inputs, mask_inputs

    def flatten_agent_inputs(self, agent_inputs):
        if isinstance(agent_inputs, (list, tuple)):
            if len(agent_inputs) == 0:
                raise ValueError('agent_inputs cannot be empty')
            if isinstance(agent_inputs[0], (list, tuple)) and len(agent_inputs[0]) == 3:
                agent_inputs = [x[0] for x in agent_inputs]
            if isinstance(agent_inputs[0], dict):
                agent_inputs = {
                    'fixed': np.array([inp['fixed'] for inp in agent_inputs]),
                    'enm': np.array([inp['enm'] for inp in agent_inputs]),
                    'ally': np.array([inp['ally'] for inp in agent_inputs])
                }
            elif isinstance(agent_inputs[0], torch.Tensor):
                agent_inputs = torch.stack(agent_inputs)

        if isinstance(agent_inputs, torch.Tensor):
            tensor = agent_inputs.to(device=self.device, dtype=torch.float32)
            if tensor.ndim == 1:
                tensor = tensor.unsqueeze(0)
            if tensor.shape[-1] != self.input_dim:
                raise ValueError(f'Expected flattened input dimension {self.input_dim}, got {tensor.shape[-1]}')
            return tensor

        if not isinstance(agent_inputs, dict):
            raise TypeError('agent_inputs must be a dict, a list of dicts, or a flattened torch.Tensor')

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
                branch_masks[index, BRANCH_idle] = True
            elif warning:
                branch_masks[index, BRANCH_evade] = True
            elif enemy_mask.any():
                branch_masks[index, BRANCH_attack] = True
            else:
                branch_masks[index, BRANCH_idle] = True
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

    def forward(self, agent_inputs, rule_inputs=None, mask_inputs=None):
        agent_inputs, rule_inputs, mask_inputs = self._unpack_inputs(agent_inputs, rule_inputs, mask_inputs)
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

    def get_action(self, agent_inputs, rule_inputs=None, mask_inputs=None, explore=True):
        agent_inputs, rule_inputs, mask_inputs = self._unpack_inputs(agent_inputs, rule_inputs, mask_inputs)
        outputs = self.forward(agent_inputs, rule_inputs, mask_inputs)
        branch_dist = Categorical(logits=outputs['branch_logits'])
        branch = self._choose(branch_dist, explore)
        conditioned = self.net(outputs['flat_inputs'], branch=branch)
        target_logits = self._masked_logits(
            conditioned['target_logits'],
            outputs['target_mask'] | (branch != BRANCH_attack).unsqueeze(-1))
        target_dist = Categorical(logits=target_logits)
        target = self._choose(target_dist, explore)
        conditioned = self.net(outputs['flat_inputs'], branch=branch, target=target)

        evad_dist = Categorical(logits=conditioned['evad_logits'])
        vert_dist = Categorical(logits=conditioned['vert_logits'])
        hor_dist = Categorical(logits=conditioned['hor_logits'])
        fire_dist = Bernoulli(logits=conditioned['fire_logits'])
        evad_action = self._choose(evad_dist, explore)
        vert_action = self._choose(vert_dist, explore)
        hor_action = self._choose(hor_dist, explore)
        fire_action = fire_dist.sample().to(torch.long) if explore else (fire_dist.probs >= 0.5).to(torch.long)
        raw_actions = {
            'branch': branch,  # 调试信息
            'evade': evad_action,
            'target': target,
            'vert': vert_action,
            'hor': hor_action,
            'fire': fire_action,
        }
        env_actions = self.to_env_actions(raw_actions, outputs['threat_delta_psi'])
        diagnostics = {
            'branch_probs': branch_dist.probs,  # 调试信息
            'target_probs': target_dist.probs,
            'evad_probs': evad_dist.probs,
            'vert_probs': vert_dist.probs,
            'hor_probs': hor_dist.probs,
            'fire_probs': fire_dist.probs,
        }
        return env_actions, raw_actions, diagnostics

    def to_env_actions(self, actions, threat_delta_psi):
        results = []
        batch_size = actions['branch'].shape[0]
        for index in range(batch_size):
            branch = int(actions['branch'][index].item())
            target = int(actions['target'][index].item())
            if branch == BRANCH_evade:
                if int(actions['evade'][index].item()) == 0:
                    maneuver = [2, 3]
                else:
                    maneuver = [2, 2 if threat_delta_psi[index].item() > 0 else 4]
                fire = False
            elif branch == BRANCH_attack:
                maneuver = [int(actions['vert'][index].item()), HORIZONTAL_ACTIONS[int(actions['hor'][index].item())]]
                fire = bool(actions['fire'][index].item())
            else:
                maneuver = [1, 0]
                fire = False
            results.append({
                'branch': BRANCH_NAMES[branch],  # 调试信息
                'flight_action': {'do': maneuver, 'at': target},
                'fire_action': {'do': fire, 'at': target},
                'radar_action': 1 if branch == BRANCH_attack else 0,
            })
        return results[0] if batch_size == 1 else results

    def evaluate_actions(self, agent_inputs, rule_inputs=None, mask_inputs=None, actions=None):
        if actions is None and mask_inputs is None and rule_inputs is not None:
            actions = rule_inputs
            rule_inputs = None

        agent_inputs, rule_inputs, mask_inputs = self._unpack_inputs(agent_inputs, rule_inputs, mask_inputs)
        outputs = self.forward(agent_inputs, rule_inputs, mask_inputs)

        if isinstance(actions, (list, tuple)):
            actions_dict = {}
            for k in ['branch', 'target', 'evade', 'vert', 'hor', 'fire']:
                if k in actions[0]:
                    actions_dict[k] = [d[k] for d in actions if k in d]
            actions = actions_dict

        def _to_tensor(val, dtype):
            if isinstance(val, torch.Tensor):
                return val.to(device=self.device, dtype=dtype).reshape(-1)
            return torch.as_tensor(np.array(val), dtype=dtype, device=self.device).reshape(-1)

        branch = _to_tensor(actions['branch'], torch.long)
        target = _to_tensor(actions['target'], torch.long)
        evad_action = _to_tensor(actions['evade'], torch.long)
        vert_action = _to_tensor(actions['vert'], torch.long)
        hor_action = _to_tensor(actions['hor'], torch.long)
        fire_action = _to_tensor(actions['fire'], torch.float32)

        conditioned = self.net(outputs['flat_inputs'], branch=branch, target=target)
        target_logits = self._masked_logits(
            conditioned['target_logits'],
            outputs['target_mask'] | (branch != BRANCH_attack).unsqueeze(-1))
        branch_dist = Categorical(logits=outputs['branch_logits'])
        target_dist = Categorical(logits=target_logits)
        evad_dist = Categorical(logits=conditioned['evad_logits'])
        vert_dist = Categorical(logits=conditioned['vert_logits'])
        hor_dist = Categorical(logits=conditioned['hor_logits'])
        fire_dist = Bernoulli(logits=conditioned['fire_logits'])

        is_a = (branch == BRANCH_evade).to(torch.float32)
        is_c = (branch == BRANCH_attack).to(torch.float32)
        log_prob = branch_dist.log_prob(branch)
        log_prob = log_prob + is_a * evad_dist.log_prob(evad_action)
        log_prob = log_prob + is_c * (
            target_dist.log_prob(target) + vert_dist.log_prob(vert_action)
            + hor_dist.log_prob(hor_action) + fire_dist.log_prob(fire_action))
        entropy = branch_dist.entropy() + is_a * evad_dist.entropy()
        entropy = entropy + is_c * (
            target_dist.entropy() + vert_dist.entropy() + hor_dist.entropy() + fire_dist.entropy())
        
        entropy_details = {'cat': None, 'bern': None, 'cont': None}
        outputs['cat'] = None
        outputs['bern'] = None
        outputs['cont'] = None
        return log_prob.unsqueeze(-1), entropy.unsqueeze(-1), entropy_details, outputs, None

