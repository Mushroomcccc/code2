<div align= "center">
    <h1>Code For Our Papaer</h1>
</div>


## 📦 Installation

We recommend creating a new [Conda environment](https://docs.conda.io/projects/conda/en/latest/user-guide/concepts/environments.html) to manage dependencies:

1. **Create a new Conda environment** (Python 3.8.10):

   ```bash
   conda create --name hdcrec python=3.8.10 -y
   ```

2. **Activate the newly created environment:**

   ```bash
   conda activate hdcrec
   ```
   
3. **Install the required modules from pip:**
   ```bash
   sh install.sh
   ```

---

## 📁 Download the Data

You can manually download the compressed dataset from the following link:

🔗 [Download from Google Drive](https://drive.google.com/file/d/1UHuN-q3SVoQRo4I_4iBlybiPpibgD3v7/view?usp=sharing)

> ⚠️ **Note:** The decompressed file size is more than **10 GB** due to the large ground-truth user-item interaction matrix.

---

## 🚀 Running Commands

You can now run the following examples if everything is set up correctly!

- The `env` argument in all experiments should be set to one of the following environments:
  - `NowPlaying-v0`
  - `KuaiRand-v0`

#### Run user model
```shell
python run/usermodel/run_DeepFM_ensemble.py --env NowPlaying-v0  --seed 2025 --cuda 1     --epoch 5 --n_models 5 --loss "pointneg" --message "pointneg" 

python run/usermodel/run_DeepFM_ensemble.py --env NowPlaying-v0  --seed 2025 --cuda 1     --epoch 5 --is_ab 1 --tau 10 --n_models 1  --loss "pointneg" --message "CIRS_UM"  
```

#### Run policies
```shell
python run/policy/run_CQL.py --env NowPlaying-v0  --cuda 1  --min-q-weight 0.3 --explore_eps 0.4 --read_message "pointneg"  --message "CQL" --num_leave_compute 5 

python run/policy/run_SQN.py        --env NowPlaying-v0  --cuda 1  --unlikely-action-threshold 0.6 --explore_eps 0.4 --read_message "pointneg"  --message "SQN"  --num_leave_compute 5 

python run/policy/run_DDPG.py --env NowPlaying-v0  --cuda 1  --remap 0.001 --explore_eps 1.2 --read_message "pointneg"  --message "DDPG"    --num_leave_compute 5 

python run/policy/run_TD3.py  --env NowPlaying-v0  --cuda 1   --remap 0.001 --explore_eps 1.5 --read_message "pointneg"  --message "TD3" --num_leave_compute 5 

python run/policy/run_PG.py  --env NowPlaying-v0  --cuda 1  --remap_eps 0.002 --read_message "pointneg"  --message "PG"  --num_leave_compute 5  

python run/policy/run_CIRS.py --env NowPlaying-v0  --cuda 1   --tau 10 --gamma_exposure 10 --read_message "CIRS_UM"  --message "CIRS" --num_leave_compute 5 

python run/policy/run_DORL.py   --env NowPlaying-v0  --cuda 0 --which_tracker avg  --read_message "pointneg"  --lambda_variance 0.01 --lambda_entropy 0.05 --message "DORL" --num_leave_compute 5  

python run/policy/run_SAC4IR.py     --env NowPlaying-v0 --cuda 0   --target_entropy_ratio 0.9 --explore_eps 0.1 --read_message "pointneg" --lambda_temper 0.01 --num_leave_compute 5  --message "SAC4IR" 

python run/policy/run_DNaIR.py  --env NowPlaying-v0  --cuda 1   --read_message "pointneg" --lambda_novelty 0.01 --message "DNaIR" --num_leave_compute 5 

python run/policy/run_HDCRec.py --env NowPlaying-v0  --cuda 0   --read_message "pointneg"  --message "HDCRec" --lambda_entropy 1.2 --num_leave_compute 5 
```
