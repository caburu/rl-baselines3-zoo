from typing import Optional, Union

import numpy as np
import gym

from stable_baselines3.common.callbacks import BaseCallback, EvalCallback
from stable_baselines3.common.vec_env import VecEnv

class ExtendedEvalCallback(EvalCallback):
    """
    Extends Eval Callback by adding a new child callback called after each evaluation.
    
    :param eval_env: The environment used for initialization
    :param callback_on_new_best: Callback to trigger
        when there is a new best model according to the ``mean_reward``
    :param callback_after_eval: Callback to trigger after each evaluation.
    :param n_eval_episodes: The number of episodes to test the agent
    :param eval_freq: Evaluate the agent every eval_freq call of the callback.
    :param log_path: Path to a folder where the evaluations (``evaluations.npz``)
        will be saved. It will be updated at each evaluation.
    :param best_model_save_path: Path to a folder where the best model
        according to performance on the eval env will be saved.
    :param deterministic: Whether the evaluation should
        use a stochastic or deterministic actions.
    :param render: Whether to render or not the environment during evaluation
    :param verbose:
    :param warn: Passed to ``evaluate_policy`` (warns if ``eval_env`` has not been
        wrapped with a Monitor wrapper)
    """
    def __init__(
        self,
        eval_env: Union[gym.Env, VecEnv],
        callback_on_new_best: Optional[BaseCallback] = None,
        callback_after_eval: Optional[BaseCallback] = None,
        n_eval_episodes: int = 5,
        eval_freq: int = 10000,
        log_path: str = None,
        best_model_save_path: str = None,
        deterministic: bool = True,
        render: bool = False,
        verbose: int = 1,
        warn: bool = True,
    ):
        super(ExtendedEvalCallback, self).__init__(
            eval_env,
            callback_on_new_best=callback_on_new_best,
            n_eval_episodes=n_eval_episodes,
            eval_freq=eval_freq,
            log_path=log_path,
            best_model_save_path=best_model_save_path,
            deterministic=deterministic,
            render=render,
            verbose=verbose,
            warn=warn)

        self.callback_after_eval = callback_after_eval
        # Give access to the parent
        if self.callback_after_eval is not None:
            self.callback_after_eval.parent = self
    
    def _init_callback(self) -> None:
        super(ExtendedEvalCallback, self)._init_callback()
        if self.callback_after_eval is not None:
            self.callback_after_eval.init_callback(self.model)
    
    def _on_step(self) -> bool:
        continue_training = super(ExtendedEvalCallback, self)._on_step()
        
        if continue_training:
            if self.eval_freq > 0 and self.n_calls % self.eval_freq == 0:
                # Trigger callback if needed
                if self.callback_after_eval is not None:
                    return self.callback_after_eval.on_step()
        return continue_training
        
class StopTrainingOnNoBestAtLastNEvals(BaseCallback):
    """
    Stop the training early if there is no new best mean reward in the last N evaluations.
    
    It is possible to define a minimum number of evaluations before start to verify evaluations without improvement.    
    
    It must be used with the ``ExtendedEvalCallback``.
    
    :param max_no_improvement_evals:  Maximum number of evaluations without new best mean reward.
    :param min_evals: Number of evaluations before start to count possible no improvements.
    :param verbose:
    """

    def __init__(self, max_no_improvement_evals: int, min_evals: int = 0, verbose: int = 0):
        super(StopTrainingOnNoBestAtLastNEvals, self).__init__(verbose=verbose)
        self.max_no_improvement_evals = max_no_improvement_evals
        self.min_evals = min_evals
        self.last_best_mean_reward = -np.inf
        self.no_improvement_evals = 0

    def _on_step(self) -> bool:
        assert self.parent is not None, "``StopTrainingOnNoBestAtLastNEvals`` callback must be used " "with an ``ExtendedEvalCallback``"
        
        continue_training = True
        
        if self.n_calls > self.min_evals:
            if self.parent.best_mean_reward > self.last_best_mean_reward:
                self.no_improvement_evals = 0                
            else:
                self.no_improvement_evals += 1
                if self.no_improvement_evals > self.max_no_improvement_evals:
                    continue_training = False        
        
        self.last_best_mean_reward = self.parent.best_mean_reward
                
        if self.verbose > 0 and not continue_training:
            print(
                f"Stopping training because there was no best mean reward in the last {self.no_improvement_evals:d} evaluations"
            )
        
        return continue_training


# Código pra testar

# from stable_baselines3 import PPO

# # Separate evaluation env
# eval_env = gym.make('Pendulum-v0')
# # Use deterministic actions for evaluation
# callback_after_eval = StopTrainingOnNoBestAtLastNEvals(
#                             max_no_improvement_evals=2,
#                             min_evals=2,
#                             verbose=1)
# eval_callback = ExtendedEvalCallback(
#                              eval_env, best_model_save_path='./logs/',
#                              log_path='./logs/', eval_freq=500,
#                              deterministic=True, render=False,
#                              callback_after_eval=callback_after_eval)

# model = PPO('MlpPolicy', 'Pendulum-v0')
# model.learn(5000, callback=eval_callback)

