#include <iostream>
#include <cstdint>
#include <fstream>
#include <ctime>

#include <algorithm>
#include "lib/fixed8.h"
#include "string.h"
#include "lib/qLib8.h"
#include "lib/Network.h"
#include "lib/Datasets/MNIST.h"
#include <vector>
#include <math.h> 


int debug(int, int, int, int);

int main(int argc, char *argv[])
{	

	bool debug_or_not = false;
	if (argc == 1)
	{
		printf("Not enough input argument.\n");
		exit(EXIT_FAILURE);
	}
	if (argc == 2)
	{
		printf("Not enough input argument.\n");
		exit(EXIT_FAILURE);
	}
	if (strcmp(argv[1],"true")!=0 && strcmp(argv[1],"false")!=0)
	{
		printf("Usage lenet8.exe --DEBUG(true || false) --network file (ie.example_network.json)\n");
		exit(EXIT_FAILURE);
	}
	// debug(4, 4, 2, 2);
	debug_or_not = (strcmp(argv[1],"true")==0);
	if(debug_or_not==true){
		std::cout<<"DEBUG = TRUE"<<"\n";
	}else{
		std::cout<<"DEBUG = False"<<"\n";
	}
	//Network
	Network network = Network();		
	network.debug(debug_or_not);
	network.build_from_json(argv[2]);
	std::cout<<"Loaded network"<<"\n";
	//Dataset	
	MNISTDataset dataset = load_mnist("example_dataset.json");
	std::cout<<"Loaded dataset"<<"\n";
	
	std::cout<<"Begin Inferencing...Total "<<dataset.ImageData.size()<<" Data Points...""\n";
	double data_amount = dataset.ImageData.size();
	data_amount = dataset.ImageData.size();
	double correct_amount = 0;
	for(int i=0;i<data_amount;i++){
		int out = network.inference(dataset.ImageData[i],dataset.ImageData_double[i]);
		if(debug_or_not==true){
			break;
		}
		if(out==dataset.labels[i]){
			std::cout<<"Correct | Label {"<<dataset.labels[i]<<"} | Predicted {"<<out<<"}\n";
			correct_amount +=1;
		}else{
			std::cout<<"Wrong | Label {"<<dataset.labels[i]<<"} | Predicted {"<<out<<"}\n";
		}
	}
	std::cout<<"TOTAL ACCURACY : "<<(correct_amount/data_amount)*100<<"% "<<"\n";
	exit(EXIT_SUCCESS);
	/*
	if (argc == 1)
	{
		printf("Not enough input argument.\n");
		exit(EXIT_FAILURE);
	}
	// debug(4, 4, 2, 2);
	double use_double = (strcmp(argv[2],"double")==0);
	
	if(use_double){
		  //Use Double
		printf("Running with Double.\n");
		if (load_weight_D(argv[1]) != 0){
			printf("Failed to load weight files.\n");
			exit(EXIT_FAILURE);
		}
	}else{
		printf("Running with Fixed point.\n");
		if (load_weight(argv[1]) != 0)
		{
			printf("Failed to load weight files.\n");
			exit(EXIT_FAILURE);
		}
	}
	printf("Loaded all weight files.\n");
	//output_file();
	int ni = 100;
	clock_t begin = clock();
	// evaluate(0);
	for (int i = 0; i < ni; ++i){
		if(use_double){
			evaluate_double(i); //<--Use Double
		}else{
			evaluate(i);
		}
	}
	clock_t end = clock();
	double time_spent = (double)(end - begin) / CLOCKS_PER_SEC;
	printf("acc: %.2f%%\n", (double(correct_cnt) / ni * 100));
	printf("Computational time: %lfs\n", time_spent);
	printf("Overflow count: %ld\n", of_cnt);
	printf("Underflow count: %ld\n", uf_cnt);
	printf("Total add count: %ld\n", add_cnt);
	*/
}

int debug(int mi, int mo, int ni, int no)
{
	std::ofstream logfile ("log.txt");
	if (logfile.is_open())
	{
		for (uint32_t i = 0; i < 256; ++i)
		{
			uint16_t result = qNum8::truncation(uint16_t(i), mi, mo, ni, no);
			logfile << qNum8::fast_inverse_quantizer(uint16_t(i), mi, mo) << "	" << qNum8::fast_inverse_quantizer(result, ni, no) << "\n";
		}
		logfile.close();
	}
	// fixed8 a, b, c, d;
	// fixed8 t, s;
	// for (u_int16_t i = 0; i < 16; ++i)
	// {
	// 	a.set(i, 1, 3);
	// 	for (u_int16_t j = 0; j < 16; ++j)
	// 	{
	// 		b.set(j, 2, 2);
	// 		c = a * b;
	// 		for (u_int16_t k = 0; k <256; ++k)
	// 		{
	// 			d.set(k, 11, 5);
	// 			s = d + c;
	// 			t = truncate(s, 3, 1);
	// 			logfile << c.fiq() << "	" << d.fiq() << "	" << s.fiq() << "	" << t.fiq() << "\n";
	// 		}
	// 	}
	// }
	// logfile.close();

	return 0;
}