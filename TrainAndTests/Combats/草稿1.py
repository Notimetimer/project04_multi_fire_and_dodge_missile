    def add(self, data):
        """
        将新数据加入 self.addon_dict，并对整个 buffer 按照 Return 的降序进行剪裁。
        保留 Return 最高的 max_size 条样本，不再单纯按时间新旧删除。
        data: 包含 'obs', 'states', 'actions', 'returns' 的字典
        """
        # 1. 验证输入数据长度一致性（仅检查 IL 必需的键）
        data_lengths = {}
        for key in ['obs', 'states', 'returns']:
            if key in data:
                arr = np.array(data[key], dtype=np.float32)
                data_lengths[key] = len(arr)
        
        if 'actions' in data:
            if isinstance(data['actions'], dict):
                for sub_k in data['actions']:
                    arr = np.array(data['actions'][sub_k])
                    data_lengths[f'actions.{sub_k}'] = len(arr)
        
        # 检查所有字段长度是否统一
        unique_lengths = set(data_lengths.values())
        if len(unique_lengths) > 1:
            raise ValueError(f"Inconsistent data lengths in IL_transition_buffer.add(): {data_lengths}")
        
        min_len = list(unique_lengths)[0] if unique_lengths else 0
        if min_len == 0:
            return
        
        # 2. 预处理：将传入数据转换为 numpy arrays
        processed = {}
        for key in ['obs', 'states', 'returns']:
            if key in data:
                processed[key] = np.array(data[key], dtype=np.float32)

        if 'actions' in data:
            if isinstance(data['actions'], list):
                # 调用全局定义的动作重组函数
                processed['actions'] = restructure_actions(data['actions'])
            elif isinstance(data['actions'], dict):
                processed['actions'] = {}
                for sub_k in data['actions']:
                    arr = np.array(data['actions'][sub_k])
                    if arr.ndim == 1:
                        arr = arr.reshape(-1, 1)
                    processed['actions'][sub_k] = arr
            else:
                processed['actions'] = data['actions']

        # 3. 拼接数据到 self.addon_dict
        for k in self.addon_dict:
            if k not in processed:
                continue

            if k == 'actions':
                for sub_k in self.addon_dict[k]:
                    if sub_k in processed[k]:
                        old_arr = self.addon_dict[k][sub_k]
                        new_arr = np.array(processed[k][sub_k])
                        if new_arr.ndim == 1:
                            new_arr = new_arr.reshape(-1, 1)
                        self.addon_dict[k][sub_k] = np.concatenate([old_arr, new_arr], axis=0)
            else:
                old_arr = self.addon_dict[k]
                new_arr = np.array(processed[k])
                # 确保维度匹配以便拼接
                if old_arr.ndim > 1 and new_arr.ndim == 1:
                    new_arr = new_arr.reshape(-1, *old_arr.shape[1:])
                self.addon_dict[k] = np.concatenate([old_arr, new_arr], axis=0)

        # 4. 核心改进：按 Return 排序并剪裁
        # 获取合并后所有样本的 Return 值并扁平化
        all_returns = self.addon_dict['returns'].flatten()
        total_len = len(all_returns)
        
        if total_len > self.max_size:
            # 获取 Return 降序排列的索引（从大到小）
            keep_indices = np.argsort(-all_returns)[:self.max_size]
            
            # 按照高质量索引重新切片所有数据键，确保样本在各个字段间同步
            for k in self.addon_dict:
                if k == 'actions':
                    for sub_k in self.addon_dict[k]:
                        self.addon_dict[k][sub_k] = self.addon_dict[k][sub_k][keep_indices]
                else:
                    self.addon_dict[k] = self.addon_dict[k][keep_indices]