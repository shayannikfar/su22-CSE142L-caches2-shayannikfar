#include "archlab.hpp"
#include <cstdlib>
#include <getopt.h>
#include <stdio.h>
#include <string.h>
#include <unistd.h>
#include "CNN/canela.hpp"
#include "math.h"
#include <omp.h>
using namespace std;

void stabilize(const std::string & version, const  tensor_t<double> & batch_data, int tile_size);

int main(int argc, char *argv[])
{
	std::vector<std::string> dataset_s;
	std::vector<std::string> default_set;
	default_set.push_back("mnist");
	uint32_t frames;
	archlab_add_option<uint32_t>("frames",  frames   , 64  ,  "images to process");
	std::string version;
	archlab_add_option<std::vector<std::string> >("dataset",
						      dataset_s,
						      default_set,
						      "mnist",
						      "Which dataset to use: 'mnist', 'emnist', 'cifar10', 'cifar100', or 'imagenet'. "
						      "Pass it multiple times to run multiple datasets.");
	archlab_add_option<std::string>("impl",
					version,
					"baseline",
					"baseline",
					"Which version to run");

	std::vector<int> omp_threads_values;
	std::vector<int> default_omp_threads_values;
	default_omp_threads_values.push_back(1);
	archlab_add_multi_option<std::vector<int> >("threads",
					      omp_threads_values,
					      default_omp_threads_values,
					      "1",
					      "How many threads use.  Pass multiple values to run with multiple thread counts.");

	size_t size;
	archlab_add_option<float>("size",
				  size,
				  1000,
				  "mininum random value in tensors");

	size_t frames;
	archlab_add_option<float>("frames",
				  frames,
				  4,
				  "mininum random value in tensors");


	std::vector<uint64_t> tile_sizes;
	std::vector<uint64_t> default_tile_sizes;
	default_tile_sizes.push_back(1);
	archlab_add_multi_option<std::vector<uint64_t> >("tile_size",
							 tile_sizes,
							 default_tile_sizes,
							 "0xDEADBEEF",		
							 "random tile_sizes to run");

	archlab_parse_cmd_line(&argc, argv);

	for(auto & tile_size: tile_sizes) {
		
		std::cout << "Running " << ds << "\n";
		

		for(auto & thread_count: omp_threads_values ) {
			theDataCollector->register_tag("omp_threads", thread_count);
			omp_set_num_threads(thread_count);

			std::cout << "Setting threadcount to " << thread_count <<"\n";
			tensor_t<double>(size, size, 1, frames) frames;
			stabilize(version, frames, tile_size);
			
		}
	}
	
	archlab_write_stats();
	return 0;
}

