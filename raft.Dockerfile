FROM ufoym/deepo:pytorch-py38-cu113
WORKDIR /RAFT


RUN apt-get update && apt-get upgrade -y \
 && apt-get install -y \
    gcc \
    ffmpeg \
    libsm6 \
    libxext6 \
	gfortran \
	libopenblas-dev \
	liblapack-dev

RUN pip install \  
	numpy \
	simplejson \
	PyYAML \
	psutil \
	opencv-python \
	tensorboard \
	yacs>=0.1.6 \
	matplotlib \
	termcolor \
	pandas \
	sklearn \
	Pillow \
        av \
	ffmpeg \
	moviepy \
	'iopath<0.1.9,>=0.1.7' \
	'tqdm>=4.29.0'

# RUN pip install --pre 'torch==1.10.0.dev20210921+cu111' -f https://download.pytorch.org/whl/nightly/cu111/torch_nightly.html
# RUN pip install torch==1.10.0+cu111 -f https://download.pytorch.org/whl/cu111/torch_stable.html
# RUN pip install pytorchvideo
#RUN pip install torch pytorchvideo torchvision
# RUN pip install --pre 'torchvision==0.11.0.dev2021092+cu111' -f https://download.pytorch.org/whl/nightly/cu111/torch_nightly.html
# RUN pip install torchvision==0.11.0+cu111 -f https://download.pytorch.org/whl/cu111/torch_stable.html
#RUN pip install torch==1.10.0+cu111 torchvision==0.11.0+cu111 pytorchvideo -f https://download.pytorch.org/whl/cu111/torch_stable.html

RUN pip install easydict tensorboardx

WORKDIR /RAFT

# python3 ./main.py --config configs/ROAD/SLOWFAST_R50_ACAR_HR2O.yaml
