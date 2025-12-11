docker build -f Dockerfile -t my-dev-env:latest .

# test with GPUs
docker run -it --gpus all my-dev-env:latest /bin/bash