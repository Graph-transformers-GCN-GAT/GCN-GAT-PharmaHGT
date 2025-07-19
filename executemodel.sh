#GCN-GAT model training script
python ./src/GCN-GAT/GNN_workflow.py --gpu 0 --train --randSplit --path './models/GAT_model/GAT_early_stop_skipCV_data2_best_opt2' --data './data/surfactant_data2/surfactant_data.csv' --gnn_model GAT   --seed 104 --test_size 0.10 --epochs 1500 --early_stop --patience 40 --skip_cv --batch_size 5 --unit_per_layer 384 --lr 0.0021
python ./src/GCN-GAT/GNN_workflow.py --gpu 0 --train --randSplit --path './models/GCN_model/GCN_early_stop_skipCV_data2_best_opt2' --data './data/surfactant_data2/surfactant_data.csv' --gnn_model GCNReg   --seed 2024 --test_size 0.10 --epochs 1000 --early_stop --patience 40 --skip_cv --batch_size 32 --unit_per_layer 128 --lr 0.00135

#PharmaHGT model training script

python train.py best_conf_data2_R2.json
