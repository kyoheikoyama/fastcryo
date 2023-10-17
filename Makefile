# Check if your "Software & Updates" is having nvidia-driver-515

env:
	conda create -n cryoem python=3.10

env.cuda:
	pip install --upgrade setuptools pip wheel
	pip install nvidia-pyindex
	pip install nvidia-cuda-runtime-cu11
	pip install nvidia-cuda-runtime-cu11
	pip install nvidia-cuda-cupti-cu11
	pip install nvidia-cuda-nvcc-cu11
	pip install nvidia-nvml-dev-cu11
	pip install nvidia-cuda-nvrtc-cu11
	pip install nvidia-nvtx-cu11
	pip install nvidia-cuda-sanitizer-api-cu11
	pip install nvidia-cublas-cu11
	pip install nvidia-cufft-cu11
	pip install nvidia-curand-cu11
	pip install nvidia-cusolver-cu11
	pip install nvidia-cusparse-cu11
	pip install nvidia-npp-cu11
	pip install nvidia-nvjpeg-cu11
	conda install pytorch torchvision torchaudio pytorch-cuda=11.7 -c pytorch -c nvidia -y
	conda install -c conda-forge pytorch-lightning -y
	conda install -c conda-forge tensorboard -y
	pip install psutil pandas lhafile pyarrow ipykernel


	#mkdir /media/kyohei/mrc_by_MotionCor/
motioncor.short:
	python script/motionCor2.py --SHORT_OR_ORIGINAL shortTIFF

motioncor.original:	
	python script/motionCor2.py --SHORT_OR_ORIGINAL cryoEM-data >> motioncor2log.log

motioncor:
	python script/motionCor2.py --SHORT_OR_ORIGINAL shortTIFF && python script/motionCor2.py --SHORT_OR_ORIGINAL EMPIAR

install:
	pip install -r requirements.txt

uninstall:
	for i in $(cat requirements.txt); do pip uninstall -y $i; done
