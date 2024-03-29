# GATraj: A Graph- and Attention-based Multi-Agent Trajectory Prediction Model
Code for ["GATraj: A Graph- and Attention-based Multi-Agent Trajectory Prediction Model"](https://www.sciencedirect.com/science/article/pii/S092427162300268X)

![](imgs/introduction.gif)

# Environment
```
pip install -r requirements.txt
```

## Train
The Default settings are to train on ETH-eth. Data cache and models will be in the subdirectory "./savedata/0/".

```
git clone https://github.com/mengmengliu1998/GATraj.git
cd GATraj
python train.py --test_set <dataset to train> --num_epochs 1000 --x_encoder_layers 3 --eta_min 1e-5  --batch_size 32\
  --learning_rate 5e-4  --randomRotate True --final_mode 20 --neighbor_thred 10\
  --using_cuda True --clip 1 --pass_time 2 --ifGaussian False --SR True --input_offset True 
```

Configuration files are also created after the first run, arguments could be modified through configuration files or command line. 
Priority: command line \> configuration files \> default values in script.


The datasets are selected on arguments '--test_set'. Five datasets in ETH/UCY are corresponding to the value of \[0,1,2,3,4\] ([**eth, hotel, zara1, zara2, univ**]). 

### Example

This command is to train model for ETH-eth
```
python train.py --test_set 0 --num_epochs 1000 --x_encoder_layers 3 --eta_min 1e-5  --batch_size 32\
  --learning_rate 5e-4  --randomRotate True --final_mode 20 --neighbor_thred 10\
  --using_cuda True --clip 1 --pass_time 2 --ifGaussian False --SR True --input_offset True
```

## Test
We provide the trained model weights in the subdirectory "./savedata/".
This command is to test model for ETH-eth, just add --phase test --load_model 1000 to the end of this training command.
```
python train.py --test_set 0 --num_epochs 1000 --x_encoder_layers 3 --eta_min 1e-5  --batch_size 32\
  --learning_rate 5e-4  --randomRotate True --final_mode 20 --neighbor_thred 10\
  --using_cuda True --clip 1 --pass_time 2 --ifGaussian False --SR True --input_offset True --phase test  --load_model 1000
```

### Cite GATraj

If you find this repo useful, please consider citing our paper
```bibtex
@article{cheng2023gatraj,
  title={Gatraj: A graph-and attention-based multi-agent trajectory prediction model},
  author={Cheng, Hao and Liu, Mengmeng and Chen, Lin and Broszio, Hellward and Sester, Monika and Yang, Michael Ying},
  journal={ISPRS Journal of Photogrammetry and Remote Sensing},
  volume={205},
  pages={163--175},
  year={2023},
  publisher={Elsevier}
}
```

### Reference

The code base of dataloader is heavily adopted from [SR-LSTM](https://github.com/zhangpur/SR-LSTM). The visulization code base for nuScenes is adopted from [PGP](https://github.com/nachiket92/PGP).
