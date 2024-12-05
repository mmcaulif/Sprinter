import logging
from typing import Optional

import cardio_rl as crl
import distrax  # type: ignore
import flax.linen as nn
import gymnasium as gym
import jax
import jax.numpy as jnp
import numpy as np
import optax  # type: ignore
import rlax  # type: ignore
from flax.training.train_state import TrainState


class Network(nn.Module):
    act_dim: int

    @nn.compact
    def __call__(self, state):
        z_pi = nn.relu(nn.Dense(64)(state))
        z_pi = nn.relu(nn.Dense(64)(z_pi))
        logits = nn.Dense(self.act_dim)(z_pi)

        z_v = nn.relu(nn.Dense(64)(state))
        z_v = nn.relu(nn.Dense(64)(z_v))
        value = nn.Dense(1)(z_v)
        return logits, value.squeeze(-1)


class A2C(crl.Agent):
    def __init__(
        self,
        env: gym.Env,
        gamma: float = 0.99,
        optim_kwargs: dict = {"learning_rate": 3e-4},
        seed: Optional[int] = None,
    ):
        seed = seed or np.random.randint(0, int(2e16))
        logging.info(f"Seed: {seed}")
        self.key = jax.random.PRNGKey(seed)
        self.key, init_key = jax.random.split(self.key)

        self.env = env

        net = Network(2)
        dummy = jnp.zeros(env.observation_space.shape)
        params = net.init(init_key, dummy)
        optimizer = optax.adam(**optim_kwargs)
        self.ts = TrainState.create(apply_fn=net.apply, params=params, tx=optimizer)

        def _step(train_state: TrainState, state: np.ndarray, key):
            logits, _ = jax.vmap(train_state.apply_fn, in_axes=(None, 0))(
                train_state.params, state
            )
            action = distrax.Categorical(logits).sample(seed=key)
            return action

        self._step = jax.jit(_step)

        def _eval_step(train_state: TrainState, state: np.ndarray):
            logits, _ = train_state.apply_fn(train_state.params, state)
            return logits.argmax(-1)

        self._eval_step = jax.jit(_eval_step)

        def _update(train_state: TrainState, s, a, r, s_p, d):
            def loss_fn(params, apply_fn, s, a, r, s_p, d):
                logits, v = jax.vmap(apply_fn, in_axes=(None, 0))(params, s)
                _, v_p = jax.vmap(apply_fn, in_axes=(None, 0))(params, s_p)

                discount = gamma * (1 - d)
                returns = jax.lax.stop_gradient(r + discount * v_p)
                error = returns - v
                v_loss = rlax.l2_loss(error).mean()

                policy_loss = jax.vmap(rlax.policy_gradient_loss)(
                    logits, a, error, jnp.ones_like(discount)
                ).mean()

                entropy_loss = jax.vmap(rlax.entropy_loss)(
                    logits, jnp.ones_like(discount)
                ).mean()

                loss = policy_loss + 0.5 * v_loss + 0.001 * entropy_loss

                return loss

            grads = jax.grad(loss_fn)(
                train_state.params, train_state.apply_fn, s, a, r, s_p, d
            )
            train_state = train_state.apply_gradients(grads=grads)
            return train_state

        self._update = jax.jit(_update)

    def update(self, batches):
        def _permute(x):
            return jnp.swapaxes(x, 0, 1)

        batches = jax.tree.map(_permute, batches)

        self.ts = self._update(
            self.ts,
            batches["s"],
            batches["a"],
            batches["r"],
            batches["s_p"],
            batches["d"],
        )
        return {}

    def step(self, state):
        self.key, act_key = jax.random.split(self.key)
        action = self._step(self.ts, state, act_key)
        action = np.asarray(action)
        return action, {}

    def eval_step(self, state):
        action = self._eval_step(self.ts, state)
        return np.asarray(action)


def main():
    envs = gym.make_vec("CartPole-v1", num_envs=16)
    eval_env = gym.make("CartPole-v1")

    runner = crl.OnPolicyRunner(
        env=envs,
        agent=A2C(env=envs),
        rollout_len=32,
        eval_env=eval_env,
    )

    runner.run(2_500, eval_freq=128, eval_episodes=20)


if __name__ == "__main__":
    main()
