# Recurrent DroQ SAC for piano playing

Combining MLP and RNN networks with the DroQSAC algorithm for controlling the robot hands to play the piano. Our network is designed for music pattern recognition, resulting in better convergence efficiency compared to MLP.
<!-- Add video here -->
<!-- [![Video](./docs/video/FurEllise_RNN_84.mp4)](./docs/video/FurEllise_RNN_84.mp4) -->

- [Installation](#installation)
- [Folder Structure](#folder-structure)
- [Dataset](#dataset)
- [Acknowledgements](#acknowledgements)

## Installation 

**1. Clone the Repository**
- Clone the repository and navigate into
```bash
git clone https://github.com/ThanachartST/PianoPlaying-DRL.git && cd PianoPlaying-DRL
```

**2. Set Up Git Submodules**
- Initialize and update submodules for this repository and  [Robopianist](https://github.com/google-research/robopianist/tree/main).

```bash
git submodule init && git submodule update
cd robopianist
git submodule init && git submodule update
```

**3.Install dependencies**
- **<u>Option 1:</u>** Using conda environment.

```bash
conda env create -f environment.yml
conda activate pianist
pip install -e ".[dev]"
```

- **<u>Option 2:</u>** Using python environment. Please ensure that the Python environment is activated.

```bash
pip install -r ./PianoPlaying-DRL/requirement.txt
pip install -e ".[dev]"
```

**4.Verify the installation (Optional)**
    
```bash
make test
```


## Folder structure



```bash
├── algorithm
│   └── RecurrentDroQSAC.py
|── common
│   ├── EnvironmentSpec.py
│   └── EnvironmentWrapper.py
|── core
│   ├── Distribution.py
│   ├── Network.py
│   └── RecurrentReplayBuffer.py
├── train.py
├── run.sh
```

- `algorithm`: This folder contains the RL algorithm class, which includes the main algorithm, DroQSAC.
- `common`: The common module for the piano playing tasks.
- `core`: The core components contributing to DroQSAC.
- `train.py`: This Python file is used to train the agent. You can specify training arguments with the CLI command, which is editable in `run.sh`.

## Datasets

<!-- - [Pig Datasets](https://arxiv.org/abs/1904.10237) -->


## Acknowledgements
<!-- 
- Mujoco
- Mujoco menagerie
- Robopianist
- MIDI licences -->


