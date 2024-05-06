# OPEN-SOURCE LIBRARY
import dm_env
import numpy as np
from torch import Tensor
from typing import NamedTuple, Optional
from common.EnvironmentSpec import RecurrentEnvironmentSpec

class RecurrentTransitionTensor(NamedTuple):
    seq_state: Tensor
    static_state: Tensor
    action: Tensor
    reward: Tensor
    discount: Tensor
    next_seq_state: Tensor
    next_static_state: Tensor

class RecurrentReplayBuffer:
    def __init__(self,
                 spec: RecurrentEnvironmentSpec, 
                 max_size: int,
                 batch_size: int) -> None:
        '''
        Initialize ReplayBuffer object by declaring the necessary variable

        Args:
            spec: EnvironmentSpec object
            max_size: the maximum number of declare buffer
            batch_size: the number determine the size of data on samples function 
        '''

        # Determine the maximum size of ReplayBuffer
        self._max_size = max_size

        # The size for samples data  
        self._batch_size = batch_size


        # Get the static and sequential observations spec
        self.seq_obs_spec = spec.seq_obs
        self.static_obs_spec = spec.static_obs

        # Decalre the array that store, state, action, next_state, reward, discount_factors
        self._seq_states = np.zeros(shape = (max_size, *self.seq_obs_spec.shape), 
                                    dtype = np.float32)
        self._next_seq_states = np.zeros(shape = (max_size, *self.seq_obs_spec.shape), 
                                         dtype = np.float32)

        self._static_states = np.zeros(shape = (max_size, *self.static_obs_spec.shape), 
                                       dtype = np.float32)
        self._next_static_states = np.zeros(shape = (max_size, *self.static_obs_spec.shape), 
                                            dtype = np.float32)

        self._actions = np.zeros(shape = (max_size, spec.action_dim), 
                                 dtype = np.float32)
        self._rewards = np.zeros(shape = (max_size), 
                                 dtype = np.float32)
        self._discounts = np.zeros(shape = (max_size), 
                                   dtype = np.float32)

        # The pointer pointing to the index of the current index
        self._ptr: int = 0
        # The current size of ReplayBuffer
        self._size: int = 0

        # Initialize the information variable with None
        self._prev: Optional[dm_env.TimeStep] = None
        self._action: Optional[np.ndarray] = None
        self._latest: Optional[dm_env.TimeStep] = None

    def insert(self,
               timestep: dm_env.TimeStep,
               action: Optional[np.ndarray]) -> None:
        '''
        Insert the environment information and action on current timestep.

        Args:
            timestep: TimeStep object, contain the environment information
                ['_seq_states', '_static_states', 'reward', 'discount'
                 '_next_seq_states', '_next_static_states' ]

        '''
        self._prev = self._latest
        self._action = action
        self._latest = timestep

        if action is not None:
            # Store the information state, action, next_state, reward, discount_factors
            self._seq_states[self._ptr] = self._prev.observation[self.seq_obs_spec.name]
            self._static_states[self._ptr] = self._prev.observation[self.static_obs_spec.name]

            self._next_seq_states[self._ptr] = self._latest.observation[self.seq_obs_spec.name]
            self._next_static_states[self._ptr] = self._latest.observation[self.static_obs_spec.name]

            self._actions[self._ptr] = action
            self._rewards[self._ptr] = self._latest.reward
            self._discounts[self._ptr] = self._latest.discount

            # Update pointer and current size
            self._ptr = (self._ptr + 1) % self._max_size
            self._size = min(self._size + 1, self._max_size)

    def sample(self, device) -> RecurrentTransitionTensor:
        ''' 
        Samples the state, action, next state, reward, discout factor
        from the replay buffer. 

        Args:
            device: torch device
        
        Returns:
            RecurrentTransitionTensor: NamedTuple object 
                with keys: ['seq_state', 'static_state', 
                            'action', 'reward', 'discount', 
                            'next_seq_state', 'next_static_state' ]
        '''
        
        # Random integer in range [0, self._size], return int array with shape (self._batch_size)
        self._ind = np.random.randint(0, self._size, size=self._batch_size)
    
        return RecurrentTransitionTensor(
            seq_state         = Tensor(self._seq_states[self._ind]).to(device),
            static_state      = Tensor(self._static_states[self._ind]).to(device), 
            action            = Tensor(self._actions[self._ind]).to(device),
            reward            = Tensor(self._rewards[self._ind]).unsqueeze(1).to(device),
            discount          = Tensor(self._discounts[self._ind]).unsqueeze(1).to(device),
            next_seq_state    = Tensor(self._next_seq_states[self._ind]).to(device), 
            next_static_state = Tensor(self._next_static_states[self._ind]).to(device))
    
    
    def __len__(self) -> int:
        '''
        Return the current number of buffer or size. 
        '''
        return self._size

    @property
    def is_ready(self) -> bool:
        ''' 
        Return true, if the samples in replay buffer
        more than batch size
        '''
        
        return self._batch_size <= len(self)
