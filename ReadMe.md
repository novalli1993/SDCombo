Evormentation install
```
conda create -n SDCombo python=3.10.0 # dcnv3 can not be install if python newer than 3.10
conda activate SDCombo
conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia

cd Segmentation\Models\InternImage\ops_dcnv3
sh make.sh
py test.py
```