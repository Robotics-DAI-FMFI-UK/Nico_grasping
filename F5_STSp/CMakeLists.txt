cmake_minimum_required(VERSION 3.19)
project(untitled)

set(CMAKE_CXX_STANDARD 20)

add_executable(untitled main.cpp som/MSOM.cpp som/MSOM.h som/SOM.cpp som/SOM.h NeuralNetwork.cpp NeuralNetwork.h NeuralGroup.cpp NeuralGroup.h NetworkUtils.cpp NetworkUtils.h som/RecSOM.cpp som/RecSOM.h Define.h dataset/StringUtils.cpp dataset/StringUtils.h Connection.cpp Connection.h backend/base64.h backend/flab/Vector.cpp backend/flab/Vector.h backend/flab/Matrix.cpp backend/flab/Matrix.h backend/flab/Base.cpp backend/flab/Base.h filters/IFilter.h backend/flab/Tensor.cpp backend/flab/Tensor.h backend/flab/RandomGenerator.cpp backend/flab/RandomGenerator.h dataset/Dataset.cpp dataset/Dataset.h dataset/DatasetConfig.h mns/Dataset.cpp mns/Dataset.h mns/ModelMNS.cpp mns/ModelMNS.h mns/Sequence.cpp mns/Sequence.h)