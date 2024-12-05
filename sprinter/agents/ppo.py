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
from rlax._src import distributions  # type: ignore

"""
Currently is quite messy as dimensions are very awkward, try and fix, possibly by changing vmap in_axes
"""


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


class PPO(crl.Agent):
    def __init__(
        self,
        env: gym.Env,
        gamma: float = 0.99,
        epochs: int = 2,
        minibatches: int = 4,
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

        self.epochs = epochs
        self.minibatches = minibatches

        def _step(train_state: TrainState, state: np.ndarray, key):
            logits, _ = jax.vmap(train_state.apply_fn, in_axes=(None, 0))(
                train_state.params, state
            )
            dist = distrax.Categorical(logits)
            action = dist.sample(seed=key)
            log_prob = dist.log_prob(action)
            return action, log_prob

        self._step = jax.jit(_step)

        def _eval_step(train_state: TrainState, state: np.ndarray):
            logits, _ = train_state.apply_fn(train_state.params, state)
            return logits.argmax(-1)

        self._eval_step = jax.jit(_eval_step)

        def _calc_gae(
            r,
            discount,
            lambda_,
            v,
            v_p,
        ):
            td_error = r + discount * v_p - v

            def _body(acc, xs):
                deltas, discounts, lambda_ = xs
                acc = deltas + discounts * lambda_ * acc
                return acc, acc

            _, gae = jax.lax.scan(
                _body, 0.0, (td_error, discount, lambda_), reverse=True
            )

            return gae

        _batch_calc_gae = jax.vmap(_calc_gae)

        def _pre_update(train_state, s, r, s_p, d):
            _, v = jax.vmap(train_state.apply_fn, in_axes=(None, 0))(params, s)
            _, v_p = jax.vmap(train_state.apply_fn, in_axes=(None, 0))(params, s_p)

            discount = gamma * (1 - d)
            lambda_ = jnp.full_like(discount, 0.95)
            gae = _batch_calc_gae(r, discount, lambda_, v, v_p)
            returns = gae + v
            return returns, gae

        self._pre_update = jax.jit(_pre_update)

        def _update(train_state: TrainState, s, a, returns, adv, old_log_probs):
            def loss_fn(params, apply_fn, s, a, returns, adv, old_log_probs):
                logits, v = jax.vmap(apply_fn, in_axes=(None, 0))(params, s)

                error = jax.lax.stop_gradient(returns) - v
                v_loss = rlax.l2_loss(error).mean()

                log_probs = distributions.softmax().logprob(a, logits)

                ratio = log_probs - jax.lax.stop_gradient(old_log_probs)

                policy_loss = jax.vmap(
                    rlax.clipped_surrogate_pg_loss, in_axes=(0, 0, None)
                )(ratio, adv, 0.2).mean()

                entropy_loss = jax.vmap(rlax.entropy_loss)(
                    logits, jnp.ones_like(a, dtype=float)
                ).mean()
                loss = policy_loss + 0.5 * v_loss + 0.001 * entropy_loss
                return loss

            grads = jax.grad(loss_fn)(
                train_state.params,
                train_state.apply_fn,
                s,
                a,
                returns,
                adv,
                old_log_probs,
            )
            train_state = train_state.apply_gradients(grads=grads)
            return train_state

        self._update = jax.jit(_update)

    def update(self, batches):
        def _permute(x):
            return jnp.swapaxes(x, 0, 1)

        batches = jax.tree.map(_permute, batches)

        returns, gae = self._pre_update(
            self.ts,
            batches["s"],
            batches["r"],
            batches["s_p"],
            batches["d"],
        )

        # batchsize = len(batches["s"])   # TODO: see below
        batchsize = batches["s"].shape[1]
        idxs = np.arange(batchsize)
        mb_size = batchsize // self.minibatches

        """
        TODO: Fix minibatching, currently its minibatching based off the agent axis, not the timestep axis
        """

        for _ in range(self.epochs):
            np.random.shuffle(idxs)
            for n in range(self.minibatches):
                mb_idxs = idxs[n * mb_size : (n + 1) * mb_size]
                # TODO: see above
                self.ts = self._update(
                    self.ts,
                    batches["s"][:, mb_idxs],
                    batches["a"][:, mb_idxs],
                    returns[:, mb_idxs],
                    gae[:, mb_idxs],
                    batches["log_prob"][:, mb_idxs],
                )

        return {}

    def step(self, state):
        self.key, act_key = jax.random.split(self.key)
        action, log_prob = self._step(self.ts, state, act_key)
        action = np.asarray(action)
        return action, {"log_prob": log_prob}

    def eval_step(self, state):
        action = self._eval_step(self.ts, state)
        return np.asarray(action)


def main():
    envs = gym.make_vec("CartPole-v1", num_envs=16)
    eval_env = gym.make("CartPole-v1")

    runner = crl.OnPolicyRunner(
        env=envs,
        agent=PPO(env=envs),
        rollout_len=256,
        eval_env=eval_env,
    )

    runner.run(2_500, eval_freq=32, eval_episodes=50)


if __name__ == "__main__":
    main()
