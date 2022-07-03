CFLAGS = -O3 -Wall

EVERYTHING = main.cpp lib/fixed8.cpp lib/qLib8.cpp lib/Layers/Layer.cpp lib/Network.cpp lib/Datasets/MNIST.cpp lib/Layers/DenseLayer.cpp lib/Layers/ConvLayer.cpp lib/Layers/BatchNormLayer.cpp lib/Layers/FlatLayer.cpp lib/Layers/MaxPoolLayer.cpp 

all: main

main:
	g++ -std=c++17 $(CFLAGS) $(EVERYTHING) -o lenet8