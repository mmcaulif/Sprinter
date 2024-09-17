import functools
import jax
import jax.numpy as jnp
import gymnasium as gym

from cardio_rl.wrappers import AtariWrapper

@functools.partial(jax.vmap, in_axes=(0, 0, 0, None))
def _crop_with_indices(img, x, y, cropped_shape):
  cropped_image = jax.lax.dynamic_slice(img, [x, y, 0], cropped_shape[1:])
  return cropped_image


def _per_image_random_crop(key, img, cropped_shape):
  """Random crop an image."""
  batch_size, width, height = cropped_shape[:-1]
  key_x, key_y = jax.random.split(key, 2)
  x = jax.random.randint(
      key_x, shape=(batch_size,), minval=0, maxval=img.shape[1] - width
  )
  y = jax.random.randint(
      key_y, shape=(batch_size,), minval=0, maxval=img.shape[2] - height
  )
  return _crop_with_indices(img, x, y, cropped_shape)


def _intensity_aug(key, x, scale=0.05):
  """Follows the code in Schwarzer et al. (2020) for intensity augmentation."""
  r = jax.random.normal(key, shape=(x.shape[0], 1, 1, 1))
  noise = 1.0 + (scale * jnp.clip(r, -2.0, 2.0))
  return x * noise


def drq_image_augmentation(key, obs, img_pad=4):
  """Padding and cropping for DrQ."""
  flat_obs = obs.reshape(-1, *obs.shape[-3:])
  paddings = [(0, 0), (img_pad, img_pad), (img_pad, img_pad), (0, 0)]
  cropped_shape = flat_obs.shape
  # The reference uses ReplicationPad2d in pytorch, but it is not available
  # in Jax. Use 'edge' instead.
  flat_obs = jnp.pad(flat_obs, paddings, 'edge')
  key1, key2 = jax.random.split(key, num=2)
  cropped_obs = _per_image_random_crop(key2, flat_obs, cropped_shape)
  # cropped_obs = _random_crop(key2, flat_obs, cropped_shape)
  aug_obs = _intensity_aug(key1, cropped_obs)
  return aug_obs.reshape(*obs.shape)


env = gym.make("PongNoFrameskip-v4")
env = AtariWrapper(env)

s, _ = env.reset()

s = jnp.expand_dims(s, 0)

print(s.shape)
exit()

aug_s = jax.jit(drq_image_augmentation)(jax.random.PRNGKey(0), s)

print(s)
print()
print(aug_s)
