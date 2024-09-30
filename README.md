-- Install pytorch_cuda for training in a conda env. To do so:

conda install pytorch=*=*cuda* torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia 

-- To launch gradients norm computations for a series of settings:

python gradients_norms.py --list_gammas 1 --n_simulations 5 --max_depth 10 --list_n 20 40 60 80 100 150 --intermediary_layer 2

-- To launch stable rank computations for a series of settings:

python stable_ranks.py --list_gammas 1 0.5 0.2 --n_simulations 5 --n_layers 10 --list_n 20 40 60 80 100 150

-- To launch training losses computations for a series of settings:

python losses.py --list_lr 1e-1 3e-1 5e-1 1e-2 3e-2 5e-2 1e-3 3e-3 5e-3 1e-4 3e-4 5e-4 1e-5 3e-5 5e-5 --n_simulations 5 --depth 1 --function_to_learn 'heaviside'
