# conda env 

conda create -n PENet python==3.8

conda activate PENet

conda install pytorch==1.8.0 torchvision==0.9.0 torchaudio==0.8.0 cudatoolkit=11.1

conda install numpy matplotlib Pillow scikit-image

pip install opencv-python

# train in KITTI

python main.py -b 1 -n e --data-folder /media/whn/新加卷/dataset/kitti_depth/depth --data-folder-rgb /media/whn/新加卷/dataset/kitti_raw
