
# === Smoke-test knobs ===
# Set MAX_EPOCHS=1 for a smoke test, or a larger number for real training.
MAX_EPOCHS=1000

# Which lightning_logs versions to feed into generation.py.
# These are auto-assigned by Lightning as version_N in
# code/deep_traffic_generation/lightning_logs/{fcvae,tcvae}/.
# If you start from an empty lightning_logs, the runs below produce:
#   fcvae/version_0   (fcvae)
#   tcvae/version_0   (tcvae, standard prior)
#   tcvae/version_1   (tcvae, vampprior)  <-- generation.py wants this one
# If lightning_logs already has prior runs, bump these to the new version_N
# that Lightning prints at the start of each training run.
FCVAE_VERSION=version_0
TCVAE_VERSION=version_1

# Suppress cosmetic third-party deprecation/future/runtime warnings.
# Unset (or set to "default") to see them again.
export PYTHONWARNINGS=ignore

# Auto-pick accelerator: cuda -> mps -> cpu.
ACCEL=$(python -c "import torch; print('gpu' if torch.cuda.is_available() else ('mps' if torch.backends.mps.is_available() else 'cpu'))")
echo "Using accelerator: $ACCEL"

## Training

cd deep_traffic_generation

# FCVAE
# Version 0:
python fcvae.py --max_epochs $MAX_EPOCHS --accelerator $ACCEL --devices 1 --data_path ../../data/traffic_noga_tilFAF_train.pkl --h_dims 216 128 64 --encoding_dim 64 --lrstep 200 --lr 0.001 --lrgamma 0.5 --gradient_clip_val 0.5 --batch_size 500 --features track groundspeed altitude timedelta --info_features latitude longitude --info_index -1

# TCVAE
# Version 0 : standard
python tcvae.py --max_epochs $MAX_EPOCHS --accelerator $ACCEL --devices 1 --data_path ../../data/traffic_noga_tilFAF_train.pkl --prior standard --encoding_dim 64 --h_dims 64 64 64 --lrstep 200 --lr 0.001 --lrgamma 0.5 --gradient_clip_val 0.5 --batch_size 500 --n_components 1000 --features track groundspeed altitude timedelta --info_features latitude longitude --info_index -1

# Version 1 : vampprior
python tcvae.py --max_epochs $MAX_EPOCHS --accelerator $ACCEL --devices 1 --data_path ../../data/traffic_noga_tilFAF_train.pkl --prior vampprior --encoding_dim 64 --h_dims 64 64 64 --lrstep 200 --lr 0.001 --lrgamma 0.5 --gradient_clip_val 0.5 --batch_size 500 --n_components 1000 --features track groundspeed altitude timedelta --info_features latitude longitude --info_index -1

cd ..

## Generation

python3 generation.py traffic_noga_tilFAF_train.pkl $FCVAE_VERSION $TCVAE_VERSION
python3 plot.py
