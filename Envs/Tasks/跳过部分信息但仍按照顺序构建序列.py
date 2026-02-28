import numpy as np

key_order = [
            "target_alive", "target_observable", "target_locked",
            "missile_in_mid_term", "locked_by_target", "warning",
            "target_information",  # 8
            "ego_main",  # 7
            "ego_control",  # 7
            "weapon",  # 1
            "threat",  # 3
            "border",  # 2
        ]

state_init = {}
state_init["target_alive"] = 1  # 默认目标存活
state_init["target_observable"] = 2  # 默认完全可见
state_init["target_locked"] = 0
state_init["missile_in_mid_term"] = 0
state_init["locked_by_target"] = 0
state_init["warning"] = 0
state_init["target_information"] = np.array([1, 0, 0, 100e3, 0, 0, 0, 0])
state_init["ego_main"] = np.array([300, 5000, 0, 1, 0, 1, 0])
state_init["ego_control"] = np.array(
    [0, 0, 0, 0, 0, 0, 0])  # pqr[0, 0, 0, 0, 0, 0, 0] 历史动作[0, 0, 340, 0, 0, 0, 0]
state_init["weapon"] = 120
state_init["threat"] = np.array([-30e3, 0, 0])  # [pi,0,30e3]  [0,0,30e3]
state_init["border"] = np.array([50e3, 0])

print(state_init)

# 部分任务
print("for 抽取")
part1_required = [
            "target_alive", "target_observable", "target_locked",
            "target_information",  # 8
            "ego_main",  # 7
            "border",  # 2
        ]
part1 = {k: (state_init[k].copy() if hasattr(state_init[k], "copy") else state_init[k])
         for k in part1_required}
print(part1)


print("itemgetter 抽取")
from operator import itemgetter
vals = itemgetter(*part1_required)(state_init)   # 返回 tuple
part1 = {k: (v.copy() if hasattr(v, "copy") else v) for k, v in zip(part1_required, vals)}
print(part1)