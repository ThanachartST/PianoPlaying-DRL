# OPEN-SOURCE LIBRARY
import os
import tyro
import time
import wandb
import random
import numpy as np
from tqdm import tqdm
from pathlib import Path
from typing import Optional, Tuple
import dm_env_wrappers as wrappers
from dataclasses import dataclass, asdict
from robopianist import suite
import robopianist.wrappers as robopianist_wrappers # type:ignore
import torch

# LOCAL LIBRARY
from core.RecurrentReplayBuffer import RecurrentReplayBuffer
from common.EnvironmentSpec import RecurrentEnvironmentSpec
from common.EnvironmentWrapper import RecurrentObservationWrapper
from algorithm.RecurrentDroQSAC import RecurrentDroQSACAgent, RecurrentDroQSACConfig


@dataclass(frozen=True)
class Args:
    # FIXME: change root_dir into os.getcwd()
    # The current file directory
    root_dir: str = os.path.join(os.getcwd(), 'log' )                           # directory for saving training details
    midi_path: Optional[Path] = None                                                       # path to midi file (.proto)
    seed: int = 42                                                              
    total_steps: int = 1_000_000                                                # total timesteps
    warmup_steps: int = 5_000                                                   # warmup timesteps
    log_interval: int = 1_000                                                   # time interval for logging training details
    eval_interval: int = 10_000                                                 # time interval for evaluation
    eval_episodes: int = 1                                                      # number of episodes for evaluation
    batch_size: int = 256                                                       
    discount: float = 0.99                                                      # discount factor
    tqdm_bar: bool = True #False                                                 
    replay_capacity: int = 1_000_000                                            # capacity of replay buffer
    project: str = "robopianist"                                                # wandb project name
    entity: str = ""                                                            # wandb entity    
    name: str = ""                                                              # wandb name
    tags: str = ""                                                              # wandb tags    
    notes: str = ""                                                             # wandb notes
    mode: str = "disabled"                                                      # wandb mode
    environment_name: str = "RoboPianist-debug-TwinkleTwinkleRousseau-v0"       # folder name for saving training details   
    n_steps_lookahead: int = 10                                                 
    trim_silence: bool = False                                                  
    gravity_compensation: bool = False
    reduced_action_space: bool = False
    control_timestep: float = 0.05
    stretch_factor: float = 1.0
    shift_factor: int = 0
    wrong_press_termination: bool = False
    disable_fingering_reward: bool = False
    disable_forearm_reward: bool = False
    disable_colorization: bool = False
    disable_hand_collisions: bool = False
    primitive_fingertip_collisions: bool = False
    frame_stack: int = 1
    clip: bool = True
    record_dir: Optional[Path] = None
    record_every: int = 1
    record_resolution: Tuple[int, int] = (480, 640)
    camera_id: Optional[str | int] = "piano/back"
    action_reward_observation: bool = False
    agent_config: RecurrentDroQSACConfig = RecurrentDroQSACConfig()


def prefix_dict(prefix: str, 
                d: dict) -> dict:
    return {f"{prefix}/{k}": v for k, v in d.items()}


def get_env(args: Args, 
            record_dir: Optional[Path] = None):
    '''    
    Initialize playing-piano environment.

    Args:
        args: Args object containg environment, agent and wandb configuration
        record_dir: Directory for saving recorded video

    Returns:
        env: Environment object

    '''
    # set up environment
    env = suite.load(
        environment_name=args.environment_name,
        midi_file=args.midi_path,
        seed=args.seed,
        stretch=args.stretch_factor,
        shift=args.shift_factor,
        task_kwargs=dict(
            n_steps_lookahead=args.n_steps_lookahead,
            trim_silence=args.trim_silence,
            gravity_compensation=args.gravity_compensation,
            reduced_action_space=args.reduced_action_space,
            control_timestep=args.control_timestep,
            wrong_press_termination=args.wrong_press_termination,
            disable_fingering_reward=args.disable_fingering_reward,
            disable_forearm_reward=args.disable_forearm_reward,
            disable_colorization=args.disable_colorization,
            disable_hand_collisions=args.disable_hand_collisions,
            primitive_fingertip_collisions=args.primitive_fingertip_collisions,
            change_color_on_activation=True,
        ),
    )
    if record_dir is not None:
        env = robopianist_wrappers.PianoSoundVideoWrapper(
            environment=env,
            record_dir=record_dir,
            record_every=args.record_every,
            camera_id=args.camera_id,
            height=args.record_resolution[0],
            width=args.record_resolution[1],
        )
        env = wrappers.EpisodeStatisticsWrapper(
            environment=env, deque_size=args.record_every
        )
        env = robopianist_wrappers.MidiEvaluationWrapper(
            environment=env, deque_size=args.record_every
        )
    else:
        env = wrappers.EpisodeStatisticsWrapper(environment=env, deque_size=1)
    if args.action_reward_observation:
        env = wrappers.ObservationActionRewardWrapper(env)
    # FIXME: This wrappers make the observation come without keys
    # env = wrappers.ConcatObservationWrapper(env)
    if args.frame_stack > 1:
        env = wrappers.FrameStackingWrapper(
            env, num_frames=args.frame_stack, flatten=True
        )
    env = wrappers.CanonicalSpecWrapper(env, clip=args.clip)
    env = wrappers.SinglePrecisionWrapper(env)
    env = wrappers.DmControlWrapper(env)
    env = RecurrentObservationWrapper(env)
    return env


def main(args: Args) -> None:
    if args.name:
        run_name = args.name
    else:
        run_name = f"SAC-{args.environment_name}-{args.seed}-{time.time()}"

    # Create experiment directory.
    experiment_dir = Path(args.root_dir) / run_name
    experiment_dir.mkdir(parents=True)

    # Seed RNGs.
    random.seed(args.seed)
    np.random.seed(args.seed)

    # initialize wandb
    wandb.init(
        project=args.project,
        entity=args.entity or None,
        tags=(args.tags.split(",") if args.tags else []),
        notes=args.notes or None,
        config=asdict(args),
        mode=args.mode,
        name=run_name,
    )

    # initialize environment
    env = get_env(args)
    eval_env = get_env(args, record_dir=experiment_dir / "eval")

    spec = RecurrentEnvironmentSpec.make(env)
    # initialize agent
    agent = RecurrentDroQSACAgent(spec=spec,
                                  config=args.agent_config,
                                  gamma=args.discount)

    # initialize replay buffer
    replay_buffer = RecurrentReplayBuffer(spec=spec, 
                                          max_size=args.replay_capacity,
                                          batch_size=args.batch_size)
    # raise ValueError()

    # reset environment
    timestep = env.reset()
    replay_buffer.insert(timestep, None)

    start_time = time.time()
    for i in tqdm(range(1, args.total_steps + 1), disable=not args.tqdm_bar):
        # sample action
        if i < args.warmup_steps:
            action = spec.sample_action(random_state=env.random_state)
        else:
            action = agent.sample_actions(timestep.observation)

        # apply acction to environment
        timestep = env.step(action)

        # store timestep tp replay buffer
        replay_buffer.insert(timestep, action)

        # reset environment if the episode ends
        if timestep.last():
            wandb.log(prefix_dict("train", env.get_statistics()), step=i)
            timestep = env.reset()
            replay_buffer.insert(timestep, None)

        # train an agent
        if i >= args.warmup_steps:
            if replay_buffer.is_ready:
                # sample batch from replay buffer
                transitions = replay_buffer.sample(agent.device)
                # update an agent
                metrics = agent.update(transitions)
                if i % args.log_interval == 0:
                    wandb.log(prefix_dict("train", metrics), step=i)

        # evaluate an agent
        if i % args.eval_interval == 0:
            for _ in range(args.eval_episodes):
                timestep = eval_env.reset()
                while not timestep.last():
                    timestep = eval_env.step(agent.eval_actions(timestep.observation))
            log_dict = prefix_dict("eval", eval_env.get_statistics())
            music_dict = prefix_dict("eval", eval_env.get_musical_metrics())
            # Print information every evaluation steps
            print( log_dict.items() )
            print( music_dict.items() )
            
            wandb.log(log_dict | music_dict, step=i)
            video = wandb.Video(str(eval_env.latest_filename), fps=4, format="mp4")
            wandb.log({"video": video, "global_step": i})
            eval_env.latest_filename.unlink()

        if i % args.log_interval == 0:
            wandb.log({"train/fps": int(i / (time.time() - start_time))}, step=i)

    torch.save( agent, f'./{args.environment_name}.pt' )

if __name__ == "__main__":
    main(tyro.cli(Args, description=__doc__))
