
git clone https://github.com/dejian-lc/humansam.git 

cd humansam

conda create -n internvideo_new python=3.10 -y

conda activate internvideo_new 

pip install -r requirements.txt    

pip install checkpoints/flashattn_wheel/*.whl
