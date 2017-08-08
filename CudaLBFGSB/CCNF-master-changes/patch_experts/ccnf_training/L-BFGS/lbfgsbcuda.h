/*************************************************************************
GPU Version:
Tsinghua University, Aug. 2012.

Written by Yun Fei in collaboration with
W. Wang and B. Wang

Original:
Optimization Technology Center.
Argonne National Laboratory and Northwestern University.

Written by Ciyou Zhu in collaboration with
R.H. Byrd, P. Lu-Chen and J. Nocedal.

Contributors:
    * Sergey Bochkanov (ALGLIB project). Translation from FORTRAN to
      pseudocode.
      
	  This software is freely available, but we  expect  that  all  publications
	  describing  work using this software, or all commercial products using it,
	  quote at least one of the references given below:
	  * R. H. Byrd, P. Lu and J. Nocedal.  A Limited  Memory  Algorithm  for
	  Bound Constrained Optimization, (1995), SIAM Journal  on  Scientific
	  and Statistical Computing , 16, 5, pp. 1190-1208.
	  * C. Zhu, R.H. Byrd and J. Nocedal. L-BFGS-B: Algorithm 778: L-BFGS-B,
	  FORTRAN routines for  large  scale  bound  constrained  optimization
	  (1997), ACM Transactions on Mathematical Software,  Vol 23,  Num. 4,
	  pp. 550 - 560.
*************************************************************************/

#pragma once

#include "cutil_inline.h"
#include <cublas_v2.h>

extern cublasHandle_t cublasHd;

#define MODU8

#ifdef MODU8
#define Modular(a, b) (a & 7)
#else
#define Modular(a, b) (a % b)
#endif

#define dynamicCall3D(f, bx, nblockx, nblocky, nblockz, st, var) \
{\
	switch(bx) { \
	case 9: \
	f<512><<<dim3(nblockx, nblocky, nblockz), dim3(512), 0, st>>>var;\
	break;\
	case 8: \
	f<256><<<dim3(nblockx, nblocky, nblockz), dim3(256), 0, st>>>var;\
	break;\
	case 7: \
	f<128><<<dim3(nblockx, nblocky, nblockz), dim3(128), 0, st>>>var;\
	break;\
	default: \
	f<64><<<dim3(nblockx, nblocky, nblockz), dim3(64), 0, st>>>var;\
	break;\
} \
}

#define dynamicCall(f, bx, nblockx, nblocky, st, var) dynamicCall3D(f, bx, nblockx, nblocky, 1, st, var)

#define inv9l2 0.36910312165415137198559104772104
#define invl2 3.3219280948873623478703194294894
#define LBFGSB_CUDA_DOUBLE_PRECISION

#ifdef LBFGSB_CUDA_DOUBLE_PRECISION
typedef double realreal;
#define machineepsilon 5E-16
#define machinemaximum 1e50
#define maxr(a, b) fmax(a, b)
#define minr(a, b) fmin(a, b)
#define absr(a) fabs(a)
#define sqrtr(a) sqrt(a)
#define rsqrtr(a) rsqrt(a)
#define cublasRtrsm cublasDtrsm
#define cublasRtrsv cublasDtrsv
#define cublasRdot cublasDdot
#define EPSG 1E-64
#define EPSF 1E-64
#define EPSX 1E-64
#define MAXITS 1000
#else
typedef float realreal;
#define machineepsilon 5E-7f
#define machinemaximum 1e20f
#define maxr(a, b) fmaxf(a, b)
#define minr(a, b) fminf(a, b)
#define absr(a) fabsf(a)
#define sqrtr(a) sqrtf(a)
#define rsqrtr(a) rsqrtf(a)
#define cublasRtrsm cublasStrsm
#define cublasRtrsv cublasStrsv
#define cublasRdot cublasSdot
#define EPSG 1e-37f
#define EPSF 1e-37f
#define EPSX 1e-37f
#define MAXITS 1000
#endif

#define SYNC_LEVEL 2

namespace lbfgsbcuda {
	inline void debugSync() {
#ifdef _DEBUG
		cutilSafeCall(cudaDeviceSynchronize());
#endif
	}
	inline int iDivUp(int a, int b)
	{
		return (a % b != 0) ? (a / b + 1) : (a / b);
	}
	inline int iDivUp2(int a, int b)
	{
		int c = a >> b;
		return (a > (c << b)) ? (c + 1) : c;
	}
	inline int log2Up(int n) {
		realreal lnb = log10((realreal)n);
		realreal nker = ceil(lnb * inv9l2);
		int m = ceil(lnb * invl2 / nker);
		m = __max(6, m);
		return m;
	}
	inline void CheckBuffer(const realreal* q, int stride, int total) {
#ifdef _DEBUG
		if(stride <= 0 || total <= 0)
			return;
		int h = iDivUp(total, stride);
		int wh = h * stride;
		realreal* hq = new realreal[wh];
		memset(hq, 0, sizeof(realreal) * wh);
		cutilSafeCall(cudaMemcpy(hq, q, total * sizeof(realreal), cudaMemcpyDeviceToHost));

/*
		char* pBufStr = new char[30 * wh];
		pBufStr[0] = '\0';
		char pbuft[30];

		for(int i = 0; i < h; i++) {
			for(int j = 0; j < stride; j++) {
				sprintf(pbuft, "%.9lf ", hq[i * stride + j]);
				strcat(pBufStr, pbuft);
			}
			strcat(pBufStr, "\n");
		}

		printf(pBufStr);*/
		delete[] hq;
/*		delete[] pBufStr;*/
#endif
	}

	inline void CheckBuffer_int(const int* q, int stride, int total) {
#ifdef _DEBUG
		if(stride <= 0 || total <= 0)
			return;
		int h = iDivUp(total, stride);
		int wh = h * stride;
		int* hq = new int[wh];
		memset(hq, 0, sizeof(int) * wh);
		cutilSafeCall(cudaMemcpy(hq, q, total * sizeof(int), cudaMemcpyDeviceToHost));
/*

		char* pBufStr = new char[30 * wh];
		pBufStr[0] = '\0';
		char pbuft[30];

		for(int i = 0; i < h; i++) {
			for(int j = 0; j < stride; j++) {
				sprintf(pbuft, "%d ", hq[i * stride + j]);
				strcat(pBufStr, pbuft);
			}
			strcat(pBufStr, "\n");
		}

		printf(pBufStr);*/
		delete[] hq;
/*		delete[] pBufStr;*/
#endif
	}

	namespace minimize {
		void vdot_vv(
			int n,
			const realreal* g,
			const realreal* d,
			realreal& gd,
			const cudaStream_t& stream = NULL
			);
		void vmul_v(
			const int n,
			realreal* d,
			const realreal stp,
			const cudaStream_t& stream = NULL
			);
		void vsub_v(
			const int n,
			const realreal* a, const realreal* b, realreal* c, const cudaStream_t& stream = NULL);
		void vdiffxchg_v(
			const int n,
			realreal* xdiff, realreal* xold, const realreal* x,
			const cudaStream_t& stream = NULL
			);
	};
	namespace active {
		void prog0(
			const int& n,
			const realreal* l,
			const realreal* u,
			const int* nbd,
			realreal* x,
			int* iwhere
			);
	};
	namespace projgr {
		void prog0(const int& n,
			const realreal* l,
			const realreal* u,
			const int* nbd,
			const realreal* x,
			const realreal* g,
			realreal* buf_n,
			realreal* sbgnrm,
			realreal* sbgnrm_dev,
			const cudaStream_t& stream);
	};
	namespace cauchy {
		void prog0
			(const int& n,
			const realreal* x,
			const realreal* l,
			const realreal* u,
			const int* nbd,
			const realreal* g,
			realreal* t,
			realreal* xcp,
			realreal* xcpb,
			const int& m,
			const realreal* wy,
			const realreal* ws,
			const realreal* sy,
			const int iPitch,
			realreal* wt,
			const realreal& theta,
			const int& col,
			const int& head,
			realreal* p,
			realreal* c,
			realreal* v,
			int& nint,
			const realreal& sbgnrm,
			realreal* buf_s_r,
			realreal* buf_array_p,
			int* iwhere,
			const int& iPitch_normal,
			const cudaStream_t* streamPool
			);
	};
	namespace freev {
		void prog0( 
			const int& n, 
			int& nfree, 
			int* index, 
			int& nenter, 
			int& ileave, 
			int* indx2, 
			const int* iwhere, 
			bool& wrk, 
			const bool& updatd, 
			const bool& cnstnd, 
			const int& iter,
			int* temp_ind1,
			int* temp_ind2,
			int* temp_ind3,
			int* temp_ind4
			);

	};
	namespace formk {
		void prog0(
			realreal* wn1,
			int m,
			int iPitch_wn,
			const cudaStream_t* streamPool
			);
		void prog1(
			const int n,
			const int nsub,
			const int ipntr,
			const int* ind,
			realreal* wn1,
			realreal* buf_array_p,
			const realreal* ws,
			const realreal* wy,
			const int head,
			const int m,
			const int col,
			const int iPitch_ws,
			const int iPitch_wn,
			const int iPitch_normal,
			const cudaStream_t* streamPool
			);
		void prog2(
			realreal* wn1,
			const int col,
			const int m,
			const int iPitch_wn,
			const cudaStream_t* streamPool
			);
		void prog3(
			const int* ind,
			const int jpntr,
			const int head,
			const int m,
			const int col,
			const int n,
			const int nsub,
			const int iPitch_ws,
			const int iPitch_wn,
			const int jy,
			const realreal* ws,
			const realreal* wy,
			realreal* buf_array_p,
			realreal* wn1,
			const int iPitch_normal,
			const cudaStream_t* streamPool);
		void prog31(
			const int* indx2, 
			const int head, 
			const int m, 
			const int upcl, 
			const int col, 
			const int nenter,
			const int ileave,
			const int n, 
			const int iPitch_ws, 
			const int iPitch_wn, 
			const realreal* wy, 
			realreal* buf_array_sup, 
			realreal* wn1,
			const realreal scal,
			const int iPitch_super,
			const cudaStream_t* streamPool		
			);
		void prog32( 
			const int* indx2, 
			const int head, 
			const int m, 
			const int upcl, 
			const int nenter,
			const int ileave,
			const int n, 
			const int iPitch_ws, 
			const int iPitch_wn, 
			const realreal* wy, 
			const realreal* ws, 
			realreal* buf_array_sup, 
			realreal* wn1,
			const int iPitch_super,
			const cudaStream_t* streamPool			
			);
		void prog4(
			const int col,
			const int iPitch_wn,
			const int iPitch_ws,
			const int m,
			const realreal* wn1,
			const realreal theta,
			const realreal* sy,
			realreal* wn,
			const cudaStream_t* streamPool);
		void prog5(
			const int col,
			const int iPitch_wn,
			realreal* wn,
			const cudaStream_t* streamPool);
	};
	namespace cmprlb {
		void prog0(
			int n,
			realreal* r,
			const realreal* g,
			const cudaStream_t& stream
			);
		void prog1(
			int nfree,
			const int* index,
			const int col,
			const int head,
			const int m,
			const int iPitch,
			const realreal* wa,
			const realreal* wy,
			const realreal* ws,
			const realreal theta,
			const realreal* z,
			const realreal* x,
			const realreal* g,
			realreal* r,
			const cudaStream_t& stream
			);
	};
	namespace subsm {
		void prog0(
			const int n,
			const int* ind,
			const int head,
			const int m,
			const int col,
			const int iPitch_ws,
			realreal* buf_array_p,
			const realreal* wy,
			const realreal* ws,
			const realreal* d,
			realreal* wv,
			const realreal theta,
			const int iPitch_normal,
			const cudaStream_t& stream
			);
		void prog1(
			realreal* wn,
			int col,
			int iPitch_wn,
			realreal* wv,
			const cudaStream_t& stream
			);
		void prog2(
			int nsub,
			const int* ind,
			const int col,
			const int head,
			const int m,
			const int iPitch,
			const realreal* wv,
			const realreal* wy,
			const realreal* ws,
			const realreal theta,
			realreal* d,
			const cudaStream_t& stream
			);
		void prog21
			( 
			int n,
			int nsub,
			const int* ind,
			const realreal* d,
			realreal* x,
			const realreal* l,
			const realreal* u,
			const int* nbd,
			const realreal* xx,
			const realreal* gg,
			realreal* buf_n_r,
			realreal* pddp,
			const cudaStream_t& stream);
		void prog3
			(int nsub,
			const int* ind,
			realreal* d,
			const int* nbd,
			realreal* buf_s_r,
			int* bufi_s_r,
			realreal* x,
			const realreal* u,
			const realreal* l,
			const cudaStream_t& stream
			);
	};
	namespace lnsrlb {
		void prog0(
			int n,
			const realreal* d,
			const int* nbd,
			const realreal* u,
			const realreal* x,
			const realreal* l,
			realreal* buf_s_r,
			realreal* stpmx_host,
			realreal* stpmx_dev,			
			const cudaStream_t& stream
			);
		void prog2(
			int n,
			realreal* x,
			realreal* d,
			const realreal* t,
			const realreal stp,
			const cudaStream_t& stream
			);
	};
	namespace matupd {
		void prog0(
			const int& n,
			const int& m,
			realreal* wy,
			realreal* sy,
			const realreal* r,
			const realreal* d,
			int& itail,
			const int& iupdat,
			int& col,
			int& head,
			const realreal& dr,
			const int& iPitch0,
			const int& iPitch_i,
			const int& iPitch_j,
			realreal* buf_array_p,
			const int& iPitch_normal,
			cudaStream_t st);
	};
	namespace formt {
		void prog01(
			const int col,
			const realreal* sy,
			const realreal* ss,
			realreal* wt,
			const int iPitch,
			const realreal theta,
			const cudaStream_t& stream
			);
	};
	namespace bmv {
		void prog0(
			const realreal* sy,
			const int& col,
			const int& iPitch,
			const realreal* v,
			realreal* p,
			const cudaStream_t& st);
		void prog1(
			const realreal* wt,
			const int& col,
			const int& iPitch,
			const realreal* v,
			realreal* p,
			const cudaStream_t& st
			);
		void prog2(
			const realreal* sy,
			realreal* wt,
			const int& col,
			const int& iPitch,
			const realreal* v,
			realreal* p,
			const cudaStream_t& st);
	};
	namespace dpofa {
		void prog0(realreal* m, int n, int pitch, int boffset, const cudaStream_t& st);
	};
};