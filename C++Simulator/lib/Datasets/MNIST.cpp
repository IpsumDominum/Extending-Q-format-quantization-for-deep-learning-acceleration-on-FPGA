#include "MNIST.h"

MNISTDataset load_mnist(std::string data_json_path){
    std::vector<std::vector<std::vector<fixed8>>> image_data {};
	std::vector<std::vector<std::vector<long double>>> image_data_double {};
    std::vector<double> labels;
    std::ifstream i(data_json_path);
    json j;
    i >> j;
    int data_m = j["data_Q_m"];
    int data_n = j["data_Q_n"];
    int data_amount = j["data_amount"];
    int data_width = j["data_width"];
    int data_height = j["data_height"];
    FILE *w;
    char c[15];
    double temp_double;
    fixed8 temp_fixed;
    /*
	====================================================================================
	= Images
	====================================================================================
	*/
	std::cout<<"Loading Dataset Images..."<<"\n";
    std::string filePathString = j["data_image_file_path"];
    char* filePath = filePathString.data();
	w = fopen(filePath, "r");//image_test.txt
	if (w != NULL)
	{
		for (int i = 0; i < data_amount; ++i){
            std::vector<std::vector<fixed8>> v {};
            image_data.push_back(v);
			std::vector<std::vector<long double>> v_d {};
            image_data_double.push_back(v_d);
			for (int j = 0; j < data_width; ++j){
                std::vector<fixed8> vv {};
                image_data[i].push_back(vv);
				std::vector<long double> vv_d {};
                image_data_double[i].push_back(vv_d);
				for (int k = 0; k < data_height; ++k)
				{
					if (fgets(c, 10, w) != NULL)
					{
						temp_double = double(atoi(c)) / 255;
						temp_fixed = fq(temp_double, data_m, data_n);//Q1.3 format
                        image_data[i][j].push_back(temp_fixed);
						image_data_double[i][j].push_back(temp_double);
					}
				}
            }
        }
		fclose(w);
		std::cout<<"Loaded Dataset Images."<<"\n";
	}
	else
	{
		perror( ("Error opening file while loading Dataset Images " + filePathString).data() );
		exit(EXIT_FAILURE);
	}
    /*
	====================================================================================
	= Labels
	====================================================================================
	*/
	std::cout<<"Loading Dataset Labels..."<<"\n";
    filePathString = j["data_label_file_path"];
    filePath = filePathString.data();
    w = fopen(filePath, "r");
	if (w != NULL)
	{
		for (int i = 0; i < data_amount; ++i)
		{
			if (fgets(c, 10, w) != NULL)
				labels.push_back(int(atoi(c)));
		}
		fclose(w);
		std::cout<<"Loaded Dataset Labels."<<"\n";
	}
	else
	{
		perror( ("Error opening file while loading Dataset Labels "+filePathString).data() );
		exit(EXIT_FAILURE);
	}
    MNISTDataset dataset = {image_data,image_data_double,labels};
    return dataset;
}

