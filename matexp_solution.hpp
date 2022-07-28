#ifndef MATEXP_SOLUTION_INCLUDED
#define MATEXP_SOLUTION_INCLUDED
#include <cstdlib>
#include "archlab.hpp"
#include <unistd.h>
#include<cstdint>
#include"function_map.hpp"
#include"tensor_t.hpp"
#include"pin_tags.h"
#include<cassert>

template<typename T>
void  __attribute__((optimize("unroll-loops")))  mult_0(tensor_t<T> &C, const tensor_t<T> &A, const tensor_t<T> &B,
	    int64_t p1,
	    int64_t p2,
	    int64_t p3,
	    int64_t p4,
	    int64_t p5) 
{
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
void t_mult_0(tensor_t<T> &C, const tensor_t<T> &B, const tensor_t<T> &A,
	    int64_t p1,
	    int64_t p2,
	    int64_t p3,
	    int64_t p4,
	    int64_t p5) 
{
	for(int i = 0; i < C.size.x; i++) {
		for(int j = 0; j < C.size.y; j++) {
			C.get(i,j) = 0;
			for(int k = 0; k < B.size.x; k++) {
				C.get(i,j) += B.get(k,i) * A.get(k,j);
			}
		}
	}
}



#define III(BODY) for(int i = 0; i < C.size.x; i += 1) {	\
		BODY;						\
	}
#define JJJ(BODY) for(int j = 0; j < C.size.y; j += 1)  {	\
		BODY;						\
	}
#define KKK(BODY) for(int k = 0; k < B.size.x; k += 1)  {	\
		BODY;						\
	}


#define II(BODY) for(int ii = 0; ii < C.size.x; ii += p1)  {	\
		BODY;						\
	}
#define JJ(BODY) for(int jj = 0; jj < C.size.y; jj += p2)  {	\
		BODY;						\
	}
#define KK(BODY) for(int kk = 0; kk < B.size.x; kk += p3)  {	\
		BODY;						\
	}


//#define I for(int i = ii; i < C.size.x && i < ii + p1; i++)

#define SPLIT_BODY(XX, X, P, M, A, BODY)                                \
        if (P < 8) {                                                    \
                UNIFIED_BODY(XX,X,P,M,A, BODY);				\
	} else if(XX + (P/8*8) > M.size.A) {				\
                for(int X = XX; X < M.size.A; X++)  {			\
                        BODY;                                           \
                }                                                       \
        } else {                                                        \
                for(int X = XX; X < XX+(P/8*8); X++)  {                 \
                        BODY;                                           \
                }                                                       \
        }

#define SPLIT_BODY_UNROLL(XX, X, P, M, A, BODY)				\
        if (P < 8) {                                                    \
                UNIFIED_BODY(XX,X,P,M,A, BODY);				\
	} else if(XX + (P/8*8) > M.size.A) {				\
                for(int X = XX; X < M.size.A; X++)  {			\
                        BODY;                                           \
                }                                                       \
        } else {                                                       \
		for(int X = XX; X < XX+(P/8*8); X++)  {			\
			BODY;						\
			X++;						\
			BODY;						\
			X++;						\
                        BODY;                                           \
			X++;						\
                        BODY;                                           \
			X++;						\
                        BODY;                                           \
			X++;						\
                        BODY;                                           \
			X++;						\
                        BODY;                                           \
			X++;						\
                        BODY;                                           \
		}							\
	}

#define UNIFIED_BODY(XX, X, P, M, A, BODY)                      \
        for(int X = XX; X < M.size.A && X < XX+P; X++)  {       \
                BODY;                                           \
        }                                                       \

#define I(BODY) SPLIT_BODY(ii, i, p1, C, x, BODY)
#define J(BODY) SPLIT_BODY(jj, j, p2, C, y, BODY)
//#define K(BODY) SPLIT_BODY_UNROLL(kk, k, p3, B, x, BODY)
#define K(BODY) SPLIT_BODY(kk, k, p3, B, x, BODY)
//#define K(BODY) UNIFIED_BODY(kk, k, p3, B, x, BODY)


//#include "impls.hpp"



template<typename T>
void //__attribute__((optimize("unroll-loops")))
mult_fast(tensor_t<T> &C, const tensor_t<T> &A, const tensor_t<T> &B,
	  int64_t p1,
	  int64_t p2,
	  int64_t p3,
	  int64_t p4,
	  int64_t p5) 
{
	memset(C.start_address(), 0, (size_t)C.end_address() - (size_t)C.start_address());

	p1 = static_cast<int64_t>(floor(sqrt(C.size.x)))/8*8;
	if (p1 == 0) {
		p1 = static_cast<int64_t>(floor(sqrt(C.size.x)));
		assert(p1 < 8);
	}
	p2 = p1;
	p3 = p1;

	for (int ii = 0; ii < C.size.x; ii += p1) {
		for (int jj = 0; jj < C.size.y; jj += p2) {
			for (int kk = 0; kk < B.size.x; kk += p3) {
				for (int i = ii; i < C.size.x && i < ii + p1; i++) {
					for (int j = jj; j < C.size.y && j < jj + p2; j++) {
						K(C.get(i, j) += B.get(i, k) * A.get(k, j));
					}
				}
			}
		}
	}
}


template<typename T>
void __attribute__((optimize("Og")))
mult_fast_unopt(tensor_t<T> &C, const tensor_t<T> &A, const tensor_t<T> &B,
	  int64_t p1,
	  int64_t p2,
	  int64_t p3,
	  int64_t p4,
	  int64_t p5) 
{
	memset(C.start_address(), 0, (size_t)C.end_address() - (size_t)C.start_address());

	p1 = static_cast<int64_t>(floor(sqrt(C.size.x)))/8*8;
	if (p1 == 0) {
		p1 = static_cast<int64_t>(floor(sqrt(C.size.x)));
		assert(p1 < 8);
	}
	p2 = p1;
	p3 = p1;
	
	for (int ii = 0; ii < C.size.x; ii += p1) {
		for (int jj = 0; jj < C.size.y; jj += p2) {
			for (int kk = 0; kk < B.size.x; kk += p3) {
				for (int i = ii; i < C.size.x && i < ii + p1; i++) {
					for (int j = jj; j < C.size.y && j < jj + p2; j++) {
						K(C.get(i, j) += B.get(i, k) * A.get(k, j));
					}
				}
			}
		}
	}

}


template<typename T>
void do_mult(tensor_t<T> &dst, const tensor_t<T> &B, const tensor_t<T> &A,
	    int64_t p1,
	    int64_t p2,
	    int64_t p3,
	    int64_t p4,
	    int64_t p5) 
{
/*	if (p4 == 1) {
		if (p5 == 1) {
			mult_0(dst,B,A, p1, p2, p3, p4, p5);
		} else {
			t_mult_0(dst,B,A, p1, p2, p3, p4, p5);
		}
		
		} else if (p4 == 100) {*/
	if (p4 == 1) {
		mult_fast(dst,B,A, p1, p2, p3, p4, p5);
	} else if (p4 == 2) {
		mult_fast_unopt(dst,B,A, p1, p2, p3, p4, p5);
	} else {
		assert(0);
	}
	
//#include "options.hpp"
/*	} else {
		assert(0);
		}	*/
}

template<typename T>
void __attribute__((noinline)) matexp_solution(tensor_t<T> & dst, const tensor_t<T> & A, uint32_t power,
		      int64_t p1=0,
		      int64_t p2=0,
		      int64_t p3=0,
		      int64_t p4=0,
		      int64_t p5=0) {

	if (p4 == 0) { // Run the reference code
		return matexp_reference(dst,A, power, p1, p2, p3, p4, p5);
	}

	if (p5 == 0) { // reference code calling our multiplication funtion
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
			do_mult(dst,B,A, p1, p2, p3, p4, p5);
			TAG_STOP("B");
		}
	} else if (p5 == 1) {
		
		TAG_START("dst", dst.start_address(), dst.end_address(), true);
		TAG_START("A", A.start_address(), A.end_address(), true);


		tensor_t<T> product(A);
		bool first_product = true;
		TAG_START("B", product.start_address(), product.end_address(), false);
	
		for(unsigned int i = 0; i < 10; i++) {
			if (power & (1 << i)) {
				if (first_product) {
					dst = product;
					first_product = false;
				} else { 
					tensor_t<T> t2(dst);
					TAG_START("t2", t2.start_address(), t2.end_address(), false);
					do_mult(dst, t2, product, p1, p2, p3, p4, p5);
				}
			}
			if (power <= (1u << i)) {
				break;
			}
			tensor_t<T> t(product);
			do_mult(product, t, t, p1, p2, p3, p4, p5);
		}

		if (first_product) {
			for(int32_t x = 0; x < dst.size.x; x++) {
				for(int32_t y = 0; y < dst.size.y; y++) {
					if (x == y) {
						dst.get(x,y) = 1;
					} else {
						dst.get(x,y) = 0;
					}	
				}
			}
		}	

		TAG_STOP("dst");
		TAG_STOP("A");
		TAG_STOP("B");
	} else {
		assert(0);
	}

}


#endif
