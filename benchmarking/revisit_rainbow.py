"""Reimplementation of revisiting rainbow.

Paper: https://github.com/JohanSamir/rainbow_extend/blob/main/lifting_veil/Configs/rainbow_invaders.gin
"""

import cardio_rl as crl
import flax.linen as nn
import gymnasium as gym
import jax
import jax.numpy as jnp

import sprinter
from sprinter.agents.rainbow import NetworkOutputs, Rainbow  # type: ignore


class Q_critic(nn.Module):
    act_dim: int
    support: jnp.ndarray

    @nn.compact
    def __call__(self, state):
        n_atoms = len(self.support)

        z = sprinter.nn.MinAtarEncoder()(state)
        z = jnp.reshape(z, -1)
        z = nn.relu(sprinter.nn.NoisyDense(128)(z))

        v = sprinter.nn.NoisyDense(n_atoms)(z)
        a = sprinter.nn.NoisyDense(self.act_dim * n_atoms)(z)

        v = jnp.expand_dims(v, -2)
        a = jnp.reshape(a, (self.act_dim, n_atoms))
        q_logits = v + a - a.mean(-2, keepdims=True)

        q_dist = jax.nn.softmax(q_logits)
        q_values = jnp.sum(q_dist * self.support, axis=-1)
        q_values = jax.lax.stop_gradient(q_values)
        return NetworkOutputs(q_values=q_values, q_logits=q_logits)


def main():
    # env = gym.make("MinAtar/Freeway-v1")
    env = gym.make("MinAtar/SpaceInvaders-v1")

    agent = Rainbow(
        env=env,
        critic=Q_critic(env.action_space.n, jnp.linspace(0.0, 100.0, 51)),
        optim_kwargs={
            "learning_rate": 0.0001,
            "eps": 0.0003125,
        },
        v_max=100.0,
        v_min=0.0,
        schedule_len=100_000,
    )

    runner = crl.OffPolicyRunner(
        env=env,
        agent=agent,
        buffer=crl.buffers.PrioritisedBuffer(env=env, capacity=100_000, n_steps=3),
        rollout_len=4,
        batch_size=32,
        warmup_len=1_000,
        n_step=3,
    )

    def how_many_rollouts(env_steps: int, warmup_len: int, rollout_len: int) -> int:
        return int((env_steps - warmup_len) / rollout_len)

    steps = how_many_rollouts(1_000_000, runner.warmup_len, runner.rollout_len)

    runner.run(rollouts=steps, eval_freq=6_250, eval_episodes=20)


if __name__ == "__main__":
    main()
