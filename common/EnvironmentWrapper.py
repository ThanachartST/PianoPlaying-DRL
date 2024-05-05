
from typing import Optional, Sequence

import tree
import numpy as np

import dm_env
from dm_env import specs
from dm_env_wrappers._src.base import EnvironmentWrapper
from dm_env_wrappers._src.concatenate_observations import _zeros_like, _concat

class TimeseriesObservationWrapper(EnvironmentWrapper):
    '''
    NOTE: The wrapper using on Robopianist tasks only!!!
    '''
    SEQ_OBS_STR = 'sequential_obs'
    STATIC_OBS_STR = 'static_obs'

    def __init__(self, environment: dm_env.Environment):
        super().__init__(environment)
        observation_spec = environment.observation_spec()
        self._obs_names = list(observation_spec.keys())

        dummy_obs = _zeros_like(observation_spec)
        dummy_obs = self._convert_observation(dummy_obs)
        self._observation_spec = {obs_key: specs.Array(shape=obs_arr.shape, dtype=obs_arr.dtype, name=obs_key) 
                                  for obs_key, obs_arr in dummy_obs.items() }

    def step(self, action) -> dm_env.TimeStep:
        timestep = self._environment.step(action)
        return timestep._replace(observation=self._convert_observation(timestep.observation))

    def reset(self) -> dm_env.TimeStep:
        timestep = self._environment.reset()
        return timestep._replace(observation=self._convert_observation(timestep.observation))

    # FIXME: OPTIMIZE IT
    def _convert_observation(self, observation):
        '''
        Separate the observation into static and sequence observation
        '''
        _temp_static_obs_dict = {}
        # New observation dict {obs_name: obs_val}
        for obs_name, obs_arr in observation.items():
            
            # NOTE: I try to find the variable that defined the 'goal' key, but it have not.
            if obs_name != 'goal':
                _temp_static_obs_dict[obs_name] = obs_arr

            else:
                # NOTE: n_keys plus 1 for current state
                # NOTE: _n_steps_lookahead plus 1 for sustain pedal state
                # Reshape observation into (num_piano_keys, num_steps_lookahead)
                seq_obs = obs_arr.reshape(self._environment.task.piano.n_keys + 1, 
                                          self._environment.task._n_steps_lookahead + 1)

        static_obs = _concat(_temp_static_obs_dict)

        # return dict of array
        return {self.SEQ_OBS_STR: seq_obs, 
                self.STATIC_OBS_STR: static_obs}
    
    def observation_spec(self):
        return self._observation_spec