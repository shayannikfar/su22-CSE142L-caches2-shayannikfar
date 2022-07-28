#ifndef MATEXP_REFERENCE_INCLUDED
#define MATEXP_REFERENCE_INCLUDED
#include <cstdlib>
#include "archlab.hpp"
#include <unistd.h>
#include<cstdint>
#include"function_map.hpp"
#include"tensor_t.hpp"
#include"pin_tags.h"



template<typename T>
void  __attribute__((noinline,optimize("Og"))) mult_reference(tensor_t<T> &C, const tensor_t<T> &A, const tensor_t<T> &B)
{
	// This is just textbook matrix multiplication.
	
	for(int i = 0; i < C.size.x; i++) { 
		for(int j = 0; j < C.size.y; j++) {
			C.get(i,j) = 0;
			for(int k = 0; k < B.size.x; k++) {
				C.get(i,j) += A.get(i,k) * B.get(k,j);
			}
		}
	}
}

template<typename T>
void __attribute__((noinline,optimize("Og"))) matexp_reference(tensor_t<T> & dst, const tensor_t<T> & A, uint32_t power,
		      // parameters you can use for whatever purpose you want (e.g., tile sizes)
		      int64_t p1=1,
		      int64_t p2=1,
		      int64_t p3=1,
		      int64_t p4=1,
		      int64_t p5=1) {
	// Tags for moneta
	
	TAG_START("dst", dst.start_address(), dst.end_address(), false);
	TAG_START("A", A.start_address(), A.end_address(), false);

	// In psuedo code this just
	//
	// dst = I
	// for(i = 0..p)
	//    dst = dst * A
	
	// Start off with the identity matrix, since M^0 == I
	// The result will end up in dst when we are done.
	for(int32_t x = 0; x < dst.size.x; x++) {
		for(int32_t y = 0; y < dst.size.y; y++) {
			if (x == y) {
				dst.get(x,y) = 1;
			} else {
				dst.get(x,y) = 0;
			}	
		}
	}


	for(uint32_t p = 0; p < power; p++) {
		tensor_t<T> B(dst); // Copy dst, since we are going to modify it.
		TAG_START("B", B.start_address(), B.end_address(), false);
		mult_reference(dst,B,A); // multiply!
		TAG_STOP("B");
	
	}

	TAG_STOP("dst");
	TAG_STOP("A");

}

#endif
