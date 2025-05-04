# mac-ripper
pov: you get a new mac and want to go full throttle on the cpu, gpu, and neural engine :P



Maxes out CPU, GPU, and Neural Engine while live-monitoring system usage:
- Multithreaded CPU overload using Python multiprocessing
- GPU stress with PyTorch + Metal Performance Shaders (MPS)
- Neural Engine benchmarking via Core ML conversion and inference
- Live CPU and RAM usage tracking with psutil
- Optional integration with native tools like stress

cmds:

python3 -m venv venv
source venv/bin/activate

pip install -r requirements.txt
brew install stress

python ripper.py