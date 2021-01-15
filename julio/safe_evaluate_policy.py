from stable_baselines3.common.vec_env import VecEnv, DummyVecEnv
import numpy as np

# O método `evaluate_policy` original da biblioteca chama `reset` duas vezes por
# episódio se o ambiente passado usar DummyVecEnv (através da VecNormalize, por exemplo).
# Isso me causou problemas ao fazer comparações com baselines.
#
# O método abaixo altera o `evaluate_policy` original garantindo apenas uma chamada
# ao método `reset` por episódio e garantindo a mesma sequência de chamadas independentemente
# o ambiente ser o original ou de usar o DummyVecEnv.
#
# MAS *** ainda tem uma chamada adicional do reset ao final.
# - Assim chamadas sucessivas do método não garantem funcionamento correto.
# - Comparações só são válidas se forem para uma única chamada ao método.
#
# Obs: eu abri uma Issue sobre isso na biblioteca:
# https://github.com/hill-a/stable-baselines/issues/906

def safe_evaluate_policy(model, env, n_eval_episodes=10, deterministic=True,
                    render=False, callback=None, reward_threshold=None,
                    return_episode_rewards=False):
    """
    Runs policy for ``n_eval_episodes`` episodes and returns average reward.
    This is made to work only with one env.

    :param model: (BaseAlgorithm) The RL agent you want to evaluate.
    :param env: (gym.Env or VecEnv) The gym environment. In the case of a ``VecEnv``
        this must contain only one environment.
    :param n_eval_episodes: (int) Number of episode to evaluate the agent
    :param deterministic: (bool) Whether to use deterministic or stochastic actions
    :param render: (bool) Whether to render the environment or not
    :param callback: (callable) callback function to do additional checks,
        called after each step.
    :param reward_threshold: (float) Minimum expected reward per episode,
        this will raise an error if the performance is not met
    :param return_episode_rewards: (bool) If True, a list of reward per episode
        will be returned instead of the mean.
    :return: (float, float) Mean reward per episode, std of reward per episode
        returns ([float], [int]) when ``return_episode_rewards`` is True
    """
    if isinstance(env, VecEnv):
        assert env.num_envs == 1, "You must pass only one environment when using this function"        
        wrap_action = False
    else:
        env = DummyVecEnv([lambda:env])
        wrap_action = True


    episode_rewards, episode_lengths = [], []
    demands_by_epis = []
    obs = env.reset()
    demands_by_epis.append(env.envs[0].customer_demands.copy())
    num_episodes = 0
    while num_episodes < n_eval_episodes:
        done, state = False, None
        episode_reward = 0.0
        episode_length = 0
        while not done:
            action, state = model.predict(obs, state=state, deterministic=deterministic)
            if wrap_action: action = [action]
            obs, reward, done, _info = env.step(action)
            episode_reward += reward
            if callback is not None:
                callback(locals(), globals())
            episode_length += 1
            if render:
                env.render()
        num_episodes += 1
        episode_rewards.append(episode_reward)
        episode_lengths.append(episode_length)
        if num_episodes < n_eval_episodes:
            demands_by_epis.append(env.envs[0].customer_demands.copy())
    mean_reward = np.mean(episode_rewards)
    std_reward = np.std(episode_rewards)
    if reward_threshold is not None:
        assert mean_reward > reward_threshold, ('Mean reward below threshold: '
                                                f'{mean_reward:.2f} < {reward_threshold:.2f}')
    if return_episode_rewards:
        return episode_rewards, episode_lengths, demands_by_epis
    return mean_reward, std_reward
