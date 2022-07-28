#include"matexp_reference.hpp"
#include"matexp_solution.hpp"

#define ELEMENT_TYPE uint64_t

typedef std::tuple<int, int> Bench;

std::vector<Bench> benches = {
	std::make_tuple(600, 2),
	std::make_tuple(350, 25),
	std::make_tuple(120, 320),
};

#define ITERATIONS 8

extern "C"
void bench_solution(tensor_t<ELEMENT_TYPE> & d, const tensor_t<ELEMENT_TYPE> & C, uint32_t p,
		    uint64_t seed,
		    int64_t p1=1,
		    int64_t p2=1,
		    int64_t p3=1,
		    int64_t p4=1,
		    int64_t p5=1) {
	
	for(auto b : benches) {
		auto size = std::get<0>(b);
		auto power = std::get<1>(b);
		
		tensor_t<ELEMENT_TYPE> D(size,size);
		tensor_t<ELEMENT_TYPE> A(size,size);
		randomize(A, seed, 0, 1024);
		
		ArchLabTimer timer;
		timer.attr("size", size);
		timer.attr("power", power);
		timer.go();
		for(int i = 0; i < ITERATIONS; i++) {
			matexp_solution<ELEMENT_TYPE>(D, A, power, p1,p2,p3,p4,p5);
		}
	} 
}
FUNCTION(matexp_bench, bench_solution);


extern "C"
void bench_reference(tensor_t<ELEMENT_TYPE> & d, const tensor_t<ELEMENT_TYPE> & C, uint32_t p,
		     uint64_t seed,
		     int64_t p1=1,
		     int64_t p2=1,
		     int64_t p3=1,
		     int64_t p4=1,
		     int64_t p5=1) {

	for(auto b : benches) {
		auto size = std::get<0>(b);
		auto power = std::get<1>(b);
		
		tensor_t<ELEMENT_TYPE> D(size,size);
		tensor_t<ELEMENT_TYPE> A(size,size);
		randomize(A, seed, 0,1024);
		
		ArchLabTimer timer;
		timer.attr("size", size);
		timer.attr("power", power);
		timer.go();
		for(int i = 0; i < ITERATIONS; i++) {
			matexp_reference<ELEMENT_TYPE>(D, A, power);
		}
		
	} 
}
FUNCTION(matexp_bench, bench_reference);


extern "C"
void matexp_reference_c(tensor_t<ELEMENT_TYPE> & dst, const tensor_t<ELEMENT_TYPE> & A, uint32_t power,
			uint64_t seed,
			int64_t p1=1,
			int64_t p2=1,
			int64_t p3=1,
			int64_t p4=1,
			int64_t p5=1)
{
	ArchLabTimer timer;					
	timer.attr("size", dst.size.x);
	timer.attr("power", power);
	timer.go();
	for(int i = 0; i < ITERATIONS; i++) {
		matexp_reference<ELEMENT_TYPE>(dst, A, power, p1, p2, p3, p4, p5);
	}
//	std::cerr << dst << "\n";
}
FUNCTION(matexp, matexp_reference_c);


extern "C"
void matexp_solution_c(tensor_t<ELEMENT_TYPE> & dst, const tensor_t<ELEMENT_TYPE> & A, uint32_t power,
		       uint64_t seed,
		       int64_t p1=1,
		       int64_t p2=1,
		       int64_t p3=1,
		       int64_t p4=1,
		       int64_t p5=1)
{
	ArchLabTimer timer;					
	timer.attr("size", dst.size.x);
	timer.attr("power", power);
	timer.go();
	for(int i = 0; i < ITERATIONS; i++) {
		matexp_solution<ELEMENT_TYPE>(dst, A, power, p1, p2, p3, p4, p5);
	}
//	std::cerr << dst << "\n";
}
FUNCTION(matexp, matexp_solution_c);
