# Setup git submodules: [ 'robopianist', 'mujoco' ]
# First please move to PianoPlaying-DRL directory.
# Note that you should use the python environment.
function get_git_submodule()
{
    git submodule init
    git submodule update
}

get_git_submodule
cd ./robopianist
get_git_submodule
bash ./scripts/install_deps.sh
pip install -e ".[dev]"
make test