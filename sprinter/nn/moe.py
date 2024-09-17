import flax.linen as nn
import jax
import jax.numpy as jnp


class Expert(nn.Module):
    dims: int

    @nn.compact
    def __call__(self, x):
        return nn.relu(nn.Dense(self.dims)(x))


class SoftMOE(nn.Module):
    experts: int = 4
    expert_dim: int = 32
    slots: int = 2

    @nn.compact
    def __call__(self, tokens):
        embedding_d = tokens.shape[-1]

        phi = self.param(
            "phi",
            nn.initializers.lecun_normal(),
            (embedding_d, self.experts, self.slots),
        )

        logits = jnp.einsum("td, dnp -> tnp", tokens, phi)

        dispatch_weights = jax.nn.softmax(logits, axis=1)
        combine_weights = jax.nn.softmax(logits, axis=(-2, -1))

        mixture_inputs = jnp.einsum("td, tnp -> npd", tokens, dispatch_weights)

        Ys = nn.vmap(
            Expert,
            in_axes=0,
            out_axes=0,
            variable_axes={"params": 0},
            split_rngs={"params": True},
            axis_size=self.experts,
        )(self.expert_dim)(mixture_inputs)

        Y = jnp.einsum("npd, tnp -> td", Ys, combine_weights)
        return Y
