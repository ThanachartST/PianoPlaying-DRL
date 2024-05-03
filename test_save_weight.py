from train import get_env, Args
from algorithm.DroQSAC import DroQSACAgent, DroQSACConfig
from common.EnvironmentSpec import EnvironmentSpec
import tyro
import torch

#   NOTE: we can save the agent class. However for the purpose 
#   which checkpoint then resume training process,
#   The problem is when we stop training and save the agent module
#       The wandb information did not follow with the saving module.
#       To overcome this problem, we need to save the "run_name" variable 
#       from the last experiment then add into CLI arguments 
#   TODO: Add arguments in Args class, "saved_weight_path" this path will have the format as follow
#       <run_name>.pt || run_name -> f"SAC-{args.environment_name}-{args.seed}-{time.time()}"
#       using the run_name for args.name instead the empty string
def main(args: Args) -> None:
    env = get_env(Args)
    spec = EnvironmentSpec.make(env)
    agent = DroQSACAgent(spec=spec,
                         config=args.agent_config,
                         gamma=args.discount)
    before_remove_statedict = agent.policy_net.state_dict()
    
    torch.save( agent, './test.pt' )
    del agent
    try:
        print( agent.policy_net.state_dict() )
    except:
        print( f'agent has been delete' )

    agent: DroQSACAgent = torch.load( './test.pt' )
    # print( agent.policy_net.state_dict() )
    after_remove_statedict = agent.policy_net.state_dict()

    for before_val, after_val in zip( before_remove_statedict.values(), after_remove_statedict.values() ):
        if not torch.equal( before_val, after_val ):
            raise ValueError

if __name__ == "__main__":
    main(tyro.cli(Args, description=__doc__))