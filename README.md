# Recurrent DroQ SAC for piano playing

Combining MLP and RNN networks with the DroQSAC algorithm for controlling the robot hands to play the piano. Our network is designed for music pattern recognition, resulting in better convergence efficiency compared to MLP.

Our document can be found [here](./PianoPlaying-DRL/docs/Piano_Playing_with_DroQSAC_and_RNN.pdf).
<!-- Add video here -->
<!-- [![Video](./docs/video/FurEllise_RNN_84.mp4)](./docs/video/FurEllise_RNN_84.mp4) -->

- [Installation](#installation)
- [Dataset](#dataset)
- [Usage](#usage)
- [Folder Structure](#folder-structure)
- [Acknowledgements](#acknowledgements)

## Installation 

**1. Clone the repository and navigate to the repository directory.**
```bash
git clone https://github.com/ThanachartST/PianoPlaying-DRL.git && cd PianoPlaying-DRL
```

**2. Set Up Git Submodules**
- Initialize and update submodules for this repository and  [Robopianist](https://github.com/google-research/robopianist).

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

## Datasets

<!-- - [Pig Datasets](https://arxiv.org/abs/1904.10237) -->

The dataset is downloaded from the [PIG website](https://beam.kisarazu.ac.jp/~saito/research/PianoFingeringDataset/), and the preprocessing script from [Robopianist dataset document](https://github.com/google-research/robopianist/blob/main/docs/dataset.md). Please complete the instructions before proceeding to the next steps.


## Usage

To train the agent, you can use the following command. Make sure you are in the repository directory and activated your environment before running.

```bash
bash run.sh
```

> [!NOTE]
> You can tune the hyperparameters in the `run.sh` script, such as changing the training song, increasing the batch size, and adjusting the network parameters.

## Folder structure

```bash
├── algorithm
│   └── RecurrentDroQSAC.py
├── common
│   ├── EnvironmentSpec.py
│   └── EnvironmentWrapper.py
├── core
│   ├── Distribution.py
│   ├── Network.py
│   └── RecurrentReplayBuffer.py
├── train.py
├── run.sh
```

- `algorithm`: This folder contains the RL algorithm class, which includes the main algorithm, DroQSAC.
- `common`: The common module for piano-playing tasks to define the tasks and the environment specifications.
- `core`: The core components contributing to DroQSAC include the networks module, replay buffer, and distributions class.
- `train.py`: This Python file is used to train the agent. You can specify training arguments with the CLI command, which is editable in `run.sh`.


## Acknowledgements

We would like to thank the following open-source resources.

- [Robopianist](https://github.com/google-research/robopianist) authors for their main reference techniques and methodologies.
- [Mujoco](https://github.com/google-deepmind/mujoco_menagerie) for developing the physics engine used in this project.
- [Pig Datasets](https://arxiv.org/abs/1904.10237) for providing the datasets supporting the piano-playing tasks.