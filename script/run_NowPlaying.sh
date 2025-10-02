# Run User Model
python run/usermodel/run_DeepFM_ensemble.py --env NowPlaying-v0  --seed 2025 --cuda 1     --epoch 5 --n_models 5 --loss "pointneg" --message "pointneg" 
python run/usermodel/run_DeepFM_ensemble.py --env NowPlaying-v0  --seed 2025 --cuda 1     --epoch 5 --is_ab 1 --tau 10 --n_models 1  --loss "pointneg" --message "CIRS_UM" 

# Run Policy
python run/policy/run_CQL.py --env NowPlaying-v0  --cuda 1  --min-q-weight 0.3 --explore_eps 0.4 --read_message "pointneg"  --message "CQL" --num_leave_compute 5 
python run/policy/run_SQN.py        --env NowPlaying-v0  --cuda 1  --unlikely-action-threshold 0.6 --explore_eps 0.4 --read_message "pointneg"  --message "SQN"  --num_leave_compute 5 
python run/policy/run_DDPG.py --env NowPlaying-v0  --cuda 1  --remap 0.001 --explore_eps 1.2 --read_message "pointneg"  --message "DDPG"    --num_leave_compute 5 
python run/policy/run_TD3.py  --env NowPlaying-v0  --cuda 1   --remap 0.001 --explore_eps 1.5 --read_message "pointneg"  --message "TD3" --num_leave_compute 5 
python run/policy/run_PG.py  --env NowPlaying-v0  --cuda 0  --remap_eps 0.002 --read_message "pointneg"  --message "PG"  --num_leave_compute 5  
# Nov
python run/policy/run_CIRS.py --env NowPlaying-v0  --cuda 0   --tau 10 --gamma_exposure 10 --read_message "CIRS_UM"  --message "CIRS" --num_leave_compute 5 
python run/policy/run_DORL.py   --env NowPlaying-v0  --cuda 0 --which_tracker avg  --read_message "pointneg"  --lambda_variance 0.01 --lambda_entropy 0.05 --message "DORL" --num_leave_compute 5  
python run/policy/run_SAC4IR.py     --env NowPlaying-v0 --cuda 0   --target_entropy_ratio 0.9 --explore_eps 0.1 --read_message "pointneg" --lambda_temper 0.01 --num_leave_compute 5  --message "SAC4IR" 
python run/policy/run_DNaIR.py  --env NowPlaying-v0  --cuda 0   --read_message "pointneg" --lambda_novelty 0.01 --message "DNaIR" --num_leave_compute 5 
# New
python run/policy/run_HDCRec.py --env NowPlaying-v0  --cuda 0   --read_message "pointneg"  --message "HDCRec" --lambda_entropy 1.2 --num_leave_compute 5 

