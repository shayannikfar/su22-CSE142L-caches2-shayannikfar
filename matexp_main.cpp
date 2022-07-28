#include <cstdlib>
#include "archlab.hpp"
#include <unistd.h>
#include"pin_tags.h"
#include<algorithm>
#include<cstdint>
#include"function_map.hpp"
#include <dlfcn.h>
#include"params.hpp"
#include"tensor_t.hpp"

#define ELEMENT_TYPE double

uint array_size;

typedef void(*matexp_impl)(tensor_t<ELEMENT_TYPE> & , const tensor_t<ELEMENT_TYPE> &, uint32_t power,
			   uint64_t,
			   int64_t,
			   int64_t,
			   int64_t,
			   int64_t,
			   int64_t);

int main(int argc, char *argv[])
{

	
	std::vector<int> mhz_s;
	std::vector<int> default_mhz;
	load_frequencies();
	default_mhz.push_back(cpu_frequencies_array[1]);
	std::stringstream clocks;
	for(int i =0; cpu_frequencies_array[i] != 0; i++) {
		clocks << cpu_frequencies_array[i] << " ";
	}
	std::stringstream fastest;
	fastest << cpu_frequencies_array[0];

	archlab_add_multi_option<std::vector<int> >("MHz",
						    mhz_s,
						    default_mhz,
						    fastest.str(),
						    "Which clock rate to run.  Possibilities on this machine are: " + clocks.str());

	std::vector<std::string> functions;
	std::vector<std::string> default_functions;
	
	default_functions.push_back("ALL");
	archlab_add_multi_option<std::vector<std::string>>("function",
							   functions,
							   default_functions,
							   "ALL",
							   "Which functions to run.");

	std::vector<unsigned long int> powers;
	std::vector<unsigned long int> default_powers;
	default_powers.push_back(16);
	archlab_add_multi_option<std::vector<unsigned long int> >("power",
								  powers,
								  default_powers,
								  "16",
								  "Power.  Pass multiple values to run with multiple sizes.");
		
	std::vector<unsigned long int> sizes;
	std::vector<unsigned long int> default_sizes;
	default_sizes.push_back(16);
	archlab_add_multi_option<std::vector<unsigned long int> >("size",
								  sizes,
								  default_sizes,
								  "16",
								  "Size.  Pass multiple values to run with multiple sizes.");
	
	PARAM(1);
	PARAM(2);
	PARAM(3);
	PARAM(4);
	PARAM(5);


	float minv = -1.0;
	archlab_add_option<float>("min",
				  minv,
				  -1.0,
				  "mininum random value in tensors");

	float maxv = 1.0;
	archlab_add_option<float>("max",
				  maxv,
				  1.0,
				  "maxinum random value in tensors");
	

	bool tag_functions;
	archlab_add_option<bool >("tag-functions",
				  tag_functions,
				  true,
				  "true",
				  "Add tags for each function invoked");

	std::vector<uint64_t> seeds;
	std::vector<uint64_t> default_seeds;
	default_seeds.push_back(0xDEADBEEF);
	archlab_add_multi_option<std::vector<uint64_t> >("seed",
							 seeds,
							 default_seeds,
							 "0xDEADBEEF",		
							 "random seeds to run");
	archlab_parse_cmd_line(&argc, argv);

	theDataCollector->disable_prefetcher();

	if (std::find(functions.begin(), functions.end(), "ALL") != functions.end()) {
		functions.clear();
		for(auto & f : function_map::get()) {
			functions.push_back(f.first);
		}
	}
	
	for(auto & function : functions) {
		auto t= function_map::get().find(function);
		if (t == function_map::get().end()) {
			std::cerr << "Unknown function: " << function <<"\n";
			exit(1);
		}
		std::cerr << "Gonna run " << function << "\n";
	}
     
	for(auto &mhz: mhz_s) {
		set_cpu_clock_frequency(mhz);
		for(auto & seed: seeds ) {
			for(auto & size:sizes) {
				for(auto & power: powers ) {
					PARAM_LOOP(1) {
						PARAM_LOOP(2) {
							PARAM_LOOP(3) {
								PARAM_LOOP(4) {
									PARAM_LOOP(5) {
										for(auto & function : functions) {
											tensor_t<ELEMENT_TYPE> dst(size,size);
											tensor_t<ELEMENT_TYPE> A(size,size);
											randomize(A, seed, minv, maxv);
										
											START_TRACE();
											std::cerr << "Running " << function << "\n";
											function_spec_t f_spec = function_map::get()[function];
											auto fut = reinterpret_cast<matexp_impl>(f_spec.second);
											std::cerr << ".";
											pristine_machine();					
											PARAM_MARK(1);
											PARAM_MARK(2);
											PARAM_MARK(3);
											PARAM_MARK(4);
											PARAM_MARK(5);
											{								
												theDataCollector->disable_prefetcher();
												theDataCollector->register_tag("function",function);
												theDataCollector->register_tag("seed",seed);
												theDataCollector->register_tag("cmdlineMHz", mhz);
											
												if (tag_functions) DUMP_START_ALL(function.c_str(), false);

												fut(dst, A,
												    power,
												    seed,
												    PARAM_PASS(1),
												    PARAM_PASS(2),
												    PARAM_PASS(3),
												    PARAM_PASS(4),
												    PARAM_PASS(5)
													);
											
												if (tag_functions )DUMP_STOP(function.c_str());
											}								
										}
										std::cerr << "\n";
									}
								}
							}
						}
					}
				}
			}
		}
	}
	archlab_write_stats();
	return 0;
}
