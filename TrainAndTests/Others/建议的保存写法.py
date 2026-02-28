# 保存
checkpoint = {
    "actor": agent.actor.state_dict(),
    "critic": agent.critic.state_dict(),
    "actor_optimizer": agent.actor_optimizer.state_dict(),
    "critic_optimizer": agent.critic_optimizer.state_dict(),
    "meta": {"gru_hidden_size": agent.gru_hidden_size, "gru_num_layers": agent.gru_num_layers}
}
th.save(checkpoint, checkpoint_path)

# 加载（建议先构造 actor，再把 actor.backbone 传给 critic）
ckpt = th.load(checkpoint_path, map_location=device)
agent.actor.load_state_dict(ckpt["actor"])
# 如果你在构造 critic 时用了 actor.backbone，则下面这一行会同时更新共享 backbone 与 critic 头
agent.critic.load_state_dict(ckpt["critic"], strict=False)
agent.actor_optimizer.load_state_dict(ckpt["actor_optimizer"])
agent.critic_optimizer.load_state_dict(ckpt["critic_optimizer"])
