@echo off

REM Setting Python Interpreter Path(your own python path)
set python_path=D:\anaconda3\envs\AR-Trip\python.exe

REM run the AR-Trip
python .\src\run.py --dataset Osak --lr 0.001 --batch_size 4 --d_model 32 --decoding_type Adapting --training_type Penalty --Drifting --Guiding

python .\src\run.py --dataset Glas --lr 0.001 --batch_size 4 --d_model 32 --decoding_type Adapting --training_type Penalty --Drifting --Guiding

python .\src\run.py --dataset Edin --lr 0.001 --batch_size 16 --d_model 32 --decoding_type Adapting --training_type Penalty --Drifting --Guiding

python .\src\run.py --dataset Toro --lr 0.001 --batch_size 8 --d_model 32 --decoding_type Adapting --training_type Penalty --Drifting --Guiding

