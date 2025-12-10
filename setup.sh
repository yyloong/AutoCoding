docker build -f Dockerfile -t my-dev-env:latest .

# test
docker run -it my-dev-env:latest /bin/bash