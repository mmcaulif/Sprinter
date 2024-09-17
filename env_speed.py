import gymnasium as gym
import time

from cardio_rl.wrappers import AtariWrapper

COUNT = 100_000

def speed_test(env: gym.Env):
    start = time.time()
    s, _ = env.reset()
    # exit(s.shape)
    count = 0
    while True:
        a = env.action_space.sample()
        s, r, d, t, _ = env.step(a)
        count += 1
        if count > COUNT:
            break
        if d or t:
            s, _ = env.reset()

    return time.time() - start


print("FPS:")

env = gym.make("AmidarNoFrameskip-v4")
env = AtariWrapper(env)

print(f"\tAtari: {COUNT/speed_test(env)}")

# env = gym.make("MinAtar/Freeway-v1")
# print(f"\tMinAtar: {COUNT/speed_test(env)}")

# env = gym.make("CartPole-v1")
# print(f"\tCartPole: {COUNT/speed_test(env)}")
