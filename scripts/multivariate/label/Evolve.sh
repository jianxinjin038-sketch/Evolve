# MSL
## 32
python ./scripts/run_benchmark.py --config-path "unfixed_detect_label_multi_config.json" --model-name "Evolve.Evolve" --data-name-list "MSL.csv" --model-hyper-params '{"seq_len": 96, "pre_len": 32, "c_in": 1, "n_heads": 8, "d_model": 256, "latent_size": 512, "num_epochs": 1, "batch_size": 16, "lr": 1e-3, "patience": 3, "hidden_dim": 64, "d_ff": 512, "dilation": 3, "K": 20, "n_clusters": 3, "lamda": 1}' --gpus 0 --num-workers 1 --timeout 60000 --save-path "label/MSL/32"
## 64
python ./scripts/run_benchmark.py --config-path "unfixed_detect_label_multi_config.json" --model-name "Evolve.Evolve" --data-name-list "MSL.csv" --model-hyper-params '{"seq_len": 96, "pre_len": 64, "c_in": 1, "n_heads": 8, "d_model": 256, "latent_size": 128, "num_epochs": 1, "batch_size": 32, "lr": 1e-3, "patience": 3, "hidden_dim": 64}' --gpus 0 --num-workers 1 --timeout 60000 --save-path "label/MSL/64"
## 128
python ./scripts/run_benchmark.py --config-path "unfixed_detect_label_multi_config.json" --model-name "Evolve.Evolve" --data-name-list "MSL.csv" --model-hyper-params '{"seq_len": 96, "pre_len": 128, "c_in": 1, "n_heads": 8, "d_model": 256, "latent_size": 128, "num_epochs": 1, "batch_size": 16, "lr": 1e-3, "patience": 3, "hidden_dim": 64}' --gpus 0 --num-workers 1 --timeout 60000 --save-path "label/MSL/128"
## 192
python ./scripts/run_benchmark.py --config-path "unfixed_detect_label_multi_config.json" --model-name "Evolve.Evolve" --data-name-list "MSL.csv" --model-hyper-params '{"seq_len": 96, "pre_len": 192, "c_in": 1, "n_heads": 8, "d_model": 256, "latent_size": 128, "num_epochs": 1, "batch_size": 32, "lr": 1e-3, "patience": 3, "hidden_dim": 64}' --gpus 0 --num-workers 1 --timeout 60000 --save-path "label/MSL/192"


# GECCO
## 32
python ./scripts/run_benchmark.py --config-path "unfixed_detect_label_multi_config.json" --model-name "Evolve.Evolve" --data-name-list "GECCO.csv" --model-hyper-params '{"seq_len": 96, "pre_len": 32, "c_in": 9, "n_heads": 8, "d_model": 256, "latent_size": 8, "num_epochs": 20, "batch_size": 16, "lr": 1e-3, "patience": 3, "hidden_dim": 64, "d_ff": 256, "dilation": 6, "K": 10, "n_clusters": 3, "lamda": 1}' --gpus 0 --num-workers 1 --timeout 60000 --save-path "label/GECCO/32"
## 64
python ./scripts/run_benchmark.py --config-path "unfixed_detect_label_multi_config.json" --model-name "Evolve.Evolve" --data-name-list "GECCO.csv" --model-hyper-params '{"seq_len": 96, "pre_len": 64, "c_in": 9, "n_heads": 8, "d_model": 256, "latent_size": 16, "num_epochs": 1, "batch_size": 32, "lr": 1e-3, "patience": 3, "hidden_dim": 64, "d_ff": 256, "dilation": 2}' --gpus 0 --num-workers 1 --timeout 60000 --save-path "label/GECCO/64"
## 128
python ./scripts/run_benchmark.py --config-path "unfixed_detect_label_multi_config.json" --model-name "Evolve.Evolve" --data-name-list "GECCO.csv" --model-hyper-params '{"seq_len": 96, "pre_len": 128, "c_in": 9, "n_heads": 8, "d_model": 256, "latent_size": 128, "num_epochs": 1, "batch_size": 16, "lr": 1e-3, "patience": 3, "hidden_dim": 64}' --gpus 0 --num-workers 1 --timeout 60000 --save-path "label/GECCO/128"
## 192
python ./scripts/run_benchmark.py --config-path "unfixed_detect_label_multi_config.json" --model-name "Evolve.Evolve" --data-name-list "GECCO.csv" --model-hyper-params '{"seq_len": 96, "pre_len": 192, "c_in": 9, "n_heads": 8, "d_model": 256, "latent_size": 128, "num_epochs": 1, "batch_size": 64, "lr": 1e-3, "patience": 3, "hidden_dim": 64}' --gpus 0 --num-workers 1 --timeout 60000 --save-path "label/GECCO/192"
