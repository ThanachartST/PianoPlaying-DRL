import dm_env
import numpy as np
from dm_env import specs
from typing import Sequence
from dataclasses import dataclass

# LOCAL LIB
from common.EnvironmentWrapper import TimeseriesObservationWrapper

@dataclass(frozen=True)
class EnvironmentSpec:
    observation: specs.Array
    action: specs.Array

    @staticmethod
    def make(env: dm_env.Environment) -> "EnvironmentSpec":
        return EnvironmentSpec(
            observation=env.observation_spec(),
            action=env.action_spec(),
        )

    def sample_action(self, random_state: np.random.RandomState) -> np.ndarray:
        if not isinstance(self.action, specs.BoundedArray):
            raise ValueError("Only BoundedArray action specs are supported.")

        action = random_state.uniform(
            low=self.action.minimum, high=self.action.maximum, size=self.action.shape
        )
        return action.astype(self.action.dtype)

    @property
    def observation_dim(self) -> int:
        return self.observation.shape[-1]

    @property
    def action_dim(self) -> int:
        return self.action.shape[-1]

# FIXME: For RNN Task we need to fixed this Spec Class
@dataclass(frozen=True)
class TimeseriesEnvironmentSpec: # FIXME: Change the name of this class
    static_obs: specs.Array
    seq_obs: specs.Array
    action: specs.Array

    @staticmethod
    def make(env: dm_env.Environment) -> "TimeseriesEnvironmentSpec":
        '''
        NOTE: We should change the action spec into BoundedArray.
        The previous version using specs.Array, which the minimum and maximum 
        of all action were in range [-1,1], but I realize that some joint 
        was not act like that.
        NOTE: This issue effect only warmup steps.
        '''

        # Check environment observation spec, it must come with dict instance
        obs_spec: dict = env.observation_spec()
        if isinstance(obs_spec, dict):
            
            # Get static observation
            if obs_spec.get(TimeseriesObservationWrapper.STATIC_OBS_STR):
                static_obs_spec = obs_spec[TimeseriesObservationWrapper.STATIC_OBS_STR]
            else:
                raise KeyError(f'{TimeseriesObservationWrapper.STATIC_OBS_STR} Not in environment observation spec ')
            
            # Get sequential observation
            if obs_spec.get(TimeseriesObservationWrapper.SEQ_OBS_STR):
                seq_obs_spec = obs_spec[TimeseriesObservationWrapper.SEQ_OBS_STR]
            else:
                raise KeyError(f'{TimeseriesObservationWrapper.SEQ_OBS_STR} Not in environment observation spec ')


        else:
            raise TypeError('The observation spec is incorrectly')

        return TimeseriesEnvironmentSpec(static_obs = static_obs_spec,
                                         seq_obs = seq_obs_spec,
                                         action = env.action_spec())

    def sample_action(self, random_state: np.random.RandomState) -> np.ndarray:
        if not isinstance(self.action, specs.BoundedArray):
            raise ValueError("Only BoundedArray action specs are supported.")

        action = random_state.uniform(
            low=self.action.minimum, high=self.action.maximum, size=self.action.shape
        )
        return action.astype(self.action.dtype)

    @property
    def observation_dim(self) -> Sequence[tuple]:
        return (self.seq_obs.shape, self.static_obs.shape)

    @property
    def action_dim(self) -> int:
        return self.action.shape[-1]


# def zeros_like(spec: specs.Array) -> jnp.ndarray:
#     return jax.tree_util.tree_map(lambda x: jnp.zeros(x.shape, x.dtype), spec)
