# DoomSapiens

Train a new model
```bash
python train.py --scenario basic --total-timesteps 100000 --save-interval 10000
```
<br/><br/>
Train a pretrained model
```bash
python train.py --scenario defend_the_center --load-model defend_the_center --load-steps 150000 --total-timesteps 850000 --save-interval 50000
```
<br/><br/>
Test a model
```bash
python test.py --episodes 50 --scenario health_gathering --load-model ..\models\health_gathering_300000.zip
```
