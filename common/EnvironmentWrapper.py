import dm_env
from dm_env import specs
from dm_env_wrappers._src import base
from dm_env_wrappers._src.concatenate_observations import _zeros_like, _concat

class RecurrentObservationWrapper(base.EnvironmentWrapper):
    '''
    NOTE: The wrapper using on Robopianist tasks only!!!
    '''
    SEQ_OBS_NAME = 'sequential_obs'
    STATIC_OBS_NAME = 'static_obs'

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

    def _convert_observation(self, observation: dict):
        '''
        Separate the observation into static and sequence observation
        '''
        if not isinstance(observation, dict):
            raise TypeError('The observation must be dictionary')

        # NOTE: n_keys plus 1 for sustain pedal state
        # NOTE: _n_steps_lookahead plus 1 for current state
        # Reshape observation into (num_steps_lookahead, num_piano_keys)
        seq_obs = observation['goal'].reshape(self._environment.task._n_steps_lookahead + 1,
                                              self._environment.task.piano.n_keys + 1)
        
        # Remove goals in temporary observation
        _temp_obs = observation.copy()
        _temp_obs.pop('goal')
        # Concat "_temp_obs" to single element
        static_obs = _concat(_temp_obs)

        # return dict of array
        return {self.SEQ_OBS_NAME: seq_obs, 
                self.STATIC_OBS_NAME: static_obs}
    
    def observation_spec(self):
        return self._observation_spec