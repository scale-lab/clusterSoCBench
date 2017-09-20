sudo cp $CUDNN505/cuda/include/* /usr/local/cuda/include 
sudo cp $CUDNN505/cuda.lib64/* /usr/local/cuda/lib64/
sudo apt-get -y install libblas-dev liblapack-dev libatlas-base-dev gfortran libgflags-dev libgoogle-glog-dev liblmdb-dev
sudo apt-get -y install --no-install-recommends libboost-all-dev
sudo apt-get -y install libprotobuf-dev libleveldb-dev libsnappy-dev libopencv-dev libhdf5-serial-dev protobuf-compiler
sudo apt-get -y install python-dev python-pip
sudo -H pip install numpy
sudo -H pip install pyparsing scipy scikit-image protobuf matplotlib


