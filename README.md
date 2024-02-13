# AR-Trip

## Brief Introduction

The implementation of AR-Trip(Anti Repetition for Trip Recommendation). We devise a cycle-aware framework to mitigate the repetition, including drifting, guiding and adapting.

## Environmental Requirements

We run the code on the device with RTX3060(12 GB), i5 12400F, and 16G memory. Please Install the dependencies via anaconda:

### Create virtual environment

```
conda create -n AR-Trip python=3.9.18
```

### Activate environment

```
conda activate AR-Trip
```

### Install pytorch and cuda toolkit

```
conda install pytorch==1.13.1 torchvision==0.14.1 torchaudio==0.13.1 pytorch-cuda=11.6 -c pytorch -c nvidia
```

### Install other requirements

```
conda install numpy pandas
pip install scikit-learn
```

## Folder Structure

| Folder Name |                    Description                    |
| :---------: | :-----------------------------------------------: |
|    asset    |        Metadata and preprocessing process         |
|   results   |       Storage related experimental results        |
|     src     |            the source code of AR-Trip             |
|  README.md  |             This instruction document             |
|   run.bat   | The necessary command-line parameters for running |

## How to run this Program

The detailed operation mode and parameter settings of each model can be found in **run.bat**. 

```
@echo off

REM Setting Python Interpreter Path
set python_path=(alter to your python path)

python .\src\run.py --dataset Osak --lr 0.001 --batch_size 4 --d_model 32 --decoding_type Adapting --training_type Penalty --Drifting --Guiding

python .\src\run.py --dataset Glas --lr 0.001 --batch_size 4 --d_model 32 --decoding_type Adapting --training_type Penalty --Drifting --Guiding

python .\src\run.py --dataset Edin --lr 0.001 --batch_size 16 --d_model 32 --decoding_type Adapting --training_type Penalty --Drifting --Guiding

python .\src\run.py --dataset Toro --lr 0.001 --batch_size 8 --d_model 32 --decoding_type Adapting --training_type Penalty --Drifting --Guiding
```

If your operating system is **Windows**, you can use the command In the working directory as

```
.\run.bat
```

to run this script file directly.  You can also directly paste commands into the terminal to run the program just like

```
python .\src\run.py --dataset Glas --lr 0.001 --batch_size 4 --d_model 32 --decoding_type Adapting --training_type Penalty --Drifting --Guiding
```

Hope such instruction could help you with our projects. Any comments and feedback are appreciated.
