# # 反向转回字典方便排查
# b_check_obs = copy.deepcopy(env.state_init)
# key_order = env.key_order
# # 将扁平向量 b_obs 按 key_order 的顺序还原到字典 b_check_obs
# arr = np.atleast_1d(np.asarray(b_obs)).reshape(-1)
# idx = 0
# for k in key_order:
#     if k not in b_check_obs:
#         raise KeyError(f"key '{k}' not in state_init")
#     v0 = b_check_obs[k]
#     # 可迭代的按长度切片，还原为 list 或 ndarray（保留原类型）
#     if isinstance(v0, (list, tuple, np.ndarray)):
#         length = len(v0)
#         slice_v = arr[idx: idx + length]
#         if isinstance(v0, np.ndarray):
#             b_check_obs[k] = slice_v.copy()
#         else:
#             b_check_obs[k] = slice_v.tolist()
#         idx += length
#     else:
#         # 标量
#         b_check_obs[k] = float(arr[idx])
#         idx += 1
# if idx != arr.size:
#     # 长度不匹配时给出提示（便于调试）
#     print(f"Warning: flattened obs length mismatch: used {idx} of {arr.size}")
