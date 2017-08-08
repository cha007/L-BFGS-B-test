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

#include "lbfgsbcuda.h"

namespace lbfgsbcuda {
	namespace subsm {
		
		template<int bx>
		__global__
		void kernel00(
			const int nsub,
			const int* ind,
			const int head,
			const int m,
			const int col,
			const int iPitch_ws,
			const int oPitch,
			realreal* buf_array_p,
			const realreal* wy,
			const realreal* ws,
			const realreal* d,
			const realreal theta
			)
		{
			const int j = blockIdx.x * blockDim.x + threadIdx.x;
			const int i = blockIdx.y;
			const int tid = threadIdx.x;
			
			volatile __shared__ realreal sdata[bx];

			realreal mySum;

			if(j < nsub) {
				int pointr = Modular((head + i % col), m);
				const int k = ind[j];
				if(i >= col) {
					mySum = ws[k * iPitch_ws + pointr] * theta;
				} else {
					mySum = wy[k * iPitch_ws + pointr];
				}
				mySum *= d[j];
			} else {
				mySum = 0;
			}

			sdata[tid] = mySum;
			__syncthreads();
			if(bx > 512) {if (tid < 512) { sdata[tid] = mySum = (mySum + sdata[tid + 512]); } __syncthreads();}			
			if(bx > 256) {if (tid < 256) { sdata[tid] = mySum = (mySum + sdata[tid + 256]); } __syncthreads();}
			if(bx > 128) {if (tid < 128) { sdata[tid] = mySum = (mySum + sdata[tid + 128]); } __syncthreads();}
			if(bx > 64) {if (tid <  64) { sdata[tid] = mySum = (mySum + sdata[tid +  64]); } __syncthreads();}
    
			if (tid < __min(bx / 2, 32))
			{
				// now that we are using warp-synchronous programming (below)
				// we need to declare our shared memory volatile so that the compiler
				// doesn't reorder stores to it and induce incorrect behavior.
				volatile realreal* smem = sdata + tid;
				if(bx > 32) {*smem = mySum = mySum + smem[32];}
				if(bx > 16) {*smem = mySum = mySum + smem[16];}
				if(bx > 8) {*smem = mySum = mySum + smem[8];}
				if(bx > 4) {*smem = mySum = mySum + smem[4];}
				if(bx > 2) {*smem = mySum = mySum + smem[2];}
				if(bx > 1) {*smem = mySum = mySum + smem[1];}
			}

			if (tid == 0) 
				buf_array_p[i * oPitch + blockIdx.x] = mySum;
		}

		template<int bx>
		__global__
		void kernel01(
			const int n,
			const int iPitch,
			const int oPitch,
			const realreal* buf_in,
			realreal* buf_out)
		{
			const int i = blockIdx.x * blockDim.x + threadIdx.x;
			const int j = blockIdx.y;
			const int tid = threadIdx.x;
			
			volatile __shared__ realreal sdata[bx];

			realreal mySum;

			if(i < n)
				mySum = buf_in[j * iPitch + i];
			else
				mySum = 0;

			sdata[tid] = mySum;
			__syncthreads();
			if(bx > 512) {if (tid < 512) { sdata[tid] = mySum = (mySum + sdata[tid + 512]); } __syncthreads();}
			if(bx > 256) {if (tid < 256) { sdata[tid] = mySum = (mySum + sdata[tid + 256]); } __syncthreads();}
			if(bx > 128) {if (tid < 128) { sdata[tid] = mySum = (mySum + sdata[tid + 128]); } __syncthreads();}
			if(bx > 64) {if (tid <  64) { sdata[tid] = mySum = (mySum + sdata[tid +  64]); } __syncthreads();}
    
			if (tid < __min(bx / 2, 32))
			{
				// now that we are using warp-synchronous programming (below)
				// we need to declare our shared memory volatile so that the compiler
				// doesn't reorder stores to it and induce incorrect behavior.
				volatile realreal* smem = sdata + tid;
				if(bx > 32) {*smem = mySum = mySum + smem[32];}
				if(bx > 16) {*smem = mySum = mySum + smem[16];}
				if(bx > 8) {*smem = mySum = mySum + smem[8];}
				if(bx > 4) {*smem = mySum = mySum + smem[4];}
				if(bx > 2) {*smem = mySum = mySum + smem[2];}
				if(bx > 1) {*smem = mySum = mySum + smem[1];}
			}

			if(tid == 0) {
				buf_out[j * oPitch + blockIdx.x] = mySum;
			}
		}

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
			)
		{
			int nblock0 = n;
			int mi = log2Up(nblock0);
			int nblock1 = iDivUp2(nblock0, mi);

			realreal* output = (nblock1 == 1) ? wv : buf_array_p;
			int op20 = (nblock1 == 1) ? 1 : iPitch_normal;

			dynamicCall(kernel00, mi, nblock1, col * 2, stream, (n, ind, head, m, col, iPitch_ws, op20, output, wy, ws, d, theta));

			nblock0 = nblock1;
			while(nblock0 > 1) {

				nblock1 = iDivUp2(nblock0, mi);

				realreal* input = output;

				output = (nblock1 == 1) ? wv : (output + nblock0);

				int op20 = (nblock1 == 1) ? 1 : iPitch_normal;
				dynamicCall(kernel01, mi, nblock1, col * 2, stream, (nblock0, iPitch_normal, op20, input, output));

				nblock0 = nblock1;
			}
		}

		__global__
		void kernel1(
			realreal* wv) 
		{
			const int i = threadIdx.x;
			wv[i] = -wv[i];
		}

		void prog1(
			realreal* wn,
			int col,
			int iPitch_wn,
			realreal* wv,
			const cudaStream_t& stream
			)
		{
			int col2 = col * 2;
			lbfgsbcuda::CheckBuffer(wv, col * 2, col * 2);

			cublasSetStream(cublasHd, stream);
			cublasRtrsv(
				cublasHd, CUBLAS_FILL_MODE_LOWER, CUBLAS_OP_N, 
				CUBLAS_DIAG_NON_UNIT, col2, wn, iPitch_wn, wv, 1);
			lbfgsbcuda::CheckBuffer(wn, iPitch_wn, iPitch_wn * 7);
			lbfgsbcuda::CheckBuffer(wv, col * 2, col * 2);
			kernel1<<<1, col, 0, stream>>>
				(wv);
			lbfgsbcuda::CheckBuffer(wv, col * 2, col * 2);
			cublasRtrsv(cublasHd, CUBLAS_FILL_MODE_LOWER, CUBLAS_OP_T, 
				CUBLAS_DIAG_NON_UNIT, col2, wn, iPitch_wn, wv, 1);
			lbfgsbcuda::CheckBuffer(wv, col * 2, col * 2);
			cublasSetStream(cublasHd, NULL);
		}

		template<int bsize>
		__global__
		void kernel2(
		int nsub,
		const int* ind,
		const int col,
		const int head,
		const int m,
		const int iPitch,
		const realreal* wv,
		const realreal* wy,
		const realreal* ws,
		const realreal inv_theta,
		realreal* d
		)
		{
			const int i = blockIdx.x * blockDim.y + threadIdx.y;
			const int tidx = threadIdx.x; //8
			const int tidy = threadIdx.y; //64
			
			volatile __shared__ realreal sdata[(512 / bsize)][bsize + 1];

			__shared__ realreal a[2][bsize+1];

			realreal mySum;

			if(tidy == 0 && tidx < col) {
				a[0][tidx] = wv[tidx] * inv_theta;
				a[1][tidx] = wv[col + tidx];
			}

			if(i < nsub && tidx < col) {
				const int pointr = Modular((head + tidx), m);
				const int k = ind[i];
				__syncthreads();
				mySum = wy[k * iPitch + pointr] * a[0][tidx] + ws[k * iPitch + pointr] * a[1][tidx];
			} else
				mySum = 0;
			
			if(bsize > 1) {
				volatile realreal* smem = sdata[tidy] + tidx;
				*smem = mySum;

				__syncthreads();

				if(bsize > 4) {*smem = mySum = mySum + smem[4];}
				if(bsize > 2) {*smem = mySum = mySum + smem[2];}
				if(bsize > 1) {*smem = mySum = mySum + smem[1];}
			}

			if(tidx == 0 && i < nsub) {
				d[i] = (d[i] + mySum) * inv_theta;
			}
		}

		void prog2(
			const int nsub,
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
			)
		{
			realreal invtheta = 1.0 / theta;

			if(col > 4) {
				int nblocky = 512 / 8;
				kernel2<8><<<dim3(iDivUp(nsub, nblocky)), dim3(8, nblocky), 0, stream>>>
					(nsub, ind, col, head, m, iPitch, wv, wy, ws, invtheta, d);
			} else if(col > 2) {
				int nblocky = 512 / 4;
				kernel2<4><<<dim3(iDivUp(nsub, nblocky)), dim3(4, nblocky), 0, stream>>>
					(nsub, ind, col, head, m, iPitch, wv, wy, ws, invtheta, d);
			} else if(col > 1) {
				int nblocky = 512 / 2;
				kernel2<2><<<dim3(iDivUp(nsub, nblocky)), dim3(2, nblocky), 0, stream>>>
					(nsub, ind, col, head, m, iPitch, wv, wy, ws, invtheta, d);
			} else if(col == 1){
				int nblocky = 512 / 1;
				kernel2<1><<<dim3(iDivUp(nsub, nblocky)), dim3(1, nblocky), 0, stream>>>
					(nsub, ind, col, head, m, iPitch, wv, wy, ws, invtheta, d);
			}
		}

		__global__
		void kernel210(
			int nsub,
			const int* ind,
			const realreal* d,
			realreal* x,
			const realreal* l,
			const realreal* u,
			const int* nbd)
		{
			const int i = blockIdx.x * blockDim.x + threadIdx.x;

			if(i >= nsub)
				return;

			const int k = ind[i];
			realreal xk = x[k] + d[i];
			const int nbdk = nbd[k];

			if(nbdk == 1) {
				xk = maxr(l[k], xk);
			} else if(nbdk == 2) {
				xk = maxr(l[k], xk);
				xk = minr(u[k], xk);
			} else if(nbdk == 3) {
				xk = minr(u[k], xk);
			}

			x[k] = xk;
		}

		template<int bx>
		__global__
		void kernel211(
			const int n,
			realreal* buf_n_r,
			const realreal* x,
			const realreal* xx,
			const realreal* gg
			)
		{
			const int i = blockIdx.x * blockDim.x + threadIdx.x;
			const int tid = threadIdx.x;
			
			volatile __shared__ realreal sdata[bx];

			realreal mySum;

			if(i < n) {
				mySum = (x[i] - xx[i]) * gg[i];
			} else {
				mySum = 0;
			}

			sdata[tid] = mySum;
			__syncthreads();
			if(bx > 512) {if (tid < 512) { sdata[tid] = mySum = (mySum + sdata[tid + 512]); } __syncthreads();}			
			if(bx > 256) {if (tid < 256) { sdata[tid] = mySum = (mySum + sdata[tid + 256]); } __syncthreads();}
			if(bx > 128) {if (tid < 128) { sdata[tid] = mySum = (mySum + sdata[tid + 128]); } __syncthreads();}
			if(bx > 64) {if (tid <  64) { sdata[tid] = mySum = (mySum + sdata[tid +  64]); } __syncthreads();}
    
			if (tid < __min(bx / 2, 32))
			{
				// now that we are using warp-synchronous programming (below)
				// we need to declare our shared memory volatile so that the compiler
				// doesn't reorder stores to it and induce incorrect behavior.
				volatile realreal* smem = sdata + tid;
				if(bx > 32) {*smem = mySum = mySum + smem[32];}
				if(bx > 16) {*smem = mySum = mySum + smem[16];}
				if(bx > 8) {*smem = mySum = mySum + smem[8];}
				if(bx > 4) {*smem = mySum = mySum + smem[4];}
				if(bx > 2) {*smem = mySum = mySum + smem[2];}
				if(bx > 1) {*smem = mySum = mySum + smem[1];}
			}

			if (tid == 0) 
				buf_n_r[blockIdx.x] = mySum;
		}

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
			const cudaStream_t& stream)
		{
			kernel210<<<iDivUp(n, 512), 512, 0, stream>>>
				(nsub, ind, d, x, l, u, nbd);

			int nblock0 = n;
			int mi = log2Up(nblock0);
			int nblock1 = iDivUp2(nblock0, mi);

			realreal* output = (nblock1 == 1) ? pddp : buf_n_r;

			dynamicCall(kernel211, mi, nblock1, 1, stream, (n, output, x, xx, gg));

			nblock0 = nblock1;
			while(nblock0 > 1) {

				nblock1 = iDivUp2(nblock0, mi);

				realreal* input = output;

				output = (nblock1 == 1) ? pddp : (output + nblock0);

				dynamicCall(kernel01, mi, nblock1, 1, stream, (nblock0, n, 1, input, output));

				nblock0 = nblock1;
			}
		}

		__device__
		inline void minex(volatile realreal& a, volatile realreal& b, volatile int& ia, volatile int& ib)
		{
			if(a > b) {
				ia = ib, a = b;
			}
		}

		template<int bx>
		__global__
		void kernel30(
		const int nsub,
		const int* ind,
		realreal* d,
		const int* nbd,
		realreal* t,
		int* ti,
		realreal* x,
		const realreal* u,
		const realreal* l
		)
		{
			const int i = blockIdx.x * blockDim.x + threadIdx.x;

			const int tid = threadIdx.x;

			volatile __shared__ realreal sdata[bx];
			volatile __shared__ int sdatai[bx];

			realreal mySum = 1.0;

			if(i < nsub) {
				const int k = ind[i];
				const int nbdi = nbd[k];

				if(nbdi != 0) {
					realreal dk = d[i];
				    if( dk < 0 && nbdi <= 2 )
					{
						realreal temp2 = l[k] - x[k];
						if( temp2 >= 0 )
						{
							mySum = 0;
						}
						else
						{
							mySum = minr(1.0, temp2 / dk);
						}
					}
					else if( dk > 0 && nbdi >= 2 )
					{
						realreal temp2 = u[k] - x[k];
						if( temp2 <= 0 )
						{
							mySum = 0;
						}
						else
						{
							mySum = minr(1.0, temp2 / dk);
						}
					}
				}
			}


			sdata[tid] = mySum;
			sdatai[tid] = i;
			__syncthreads();

			t[i] = mySum;
			ti[i] = i;

			if(bx > 512) {if (tid < 512) { minex(sdata[tid], sdata[tid + 512], sdatai[tid], sdatai[tid + 512]); } __syncthreads();}			
			if(bx > 256) {if (tid < 256) { minex(sdata[tid], sdata[tid + 256], sdatai[tid], sdatai[tid + 256]); } __syncthreads();}
			if(bx > 128) {if (tid < 128) { minex(sdata[tid], sdata[tid + 128], sdatai[tid], sdatai[tid + 128]); } __syncthreads();}
			if(bx > 64) {if (tid <  64) { minex(sdata[tid], sdata[tid +  64], sdatai[tid], sdatai[tid + 64]); } __syncthreads();}
    
			if (tid < __min(bx / 2, 32))
			{
				// now that we are using warp-synchronous programming (below)
				// we need to declare our shared memory volatile so that the compiler
				// doesn't reorder stores to it and induce incorrect behavior.
				volatile realreal* smem = sdata + tid;
				volatile int* smemi = sdatai + tid;
				if(bx > 32) {minex(*smem, smem[32], *smemi, smemi[32]);}
				if(bx > 16) {minex(*smem, smem[16], *smemi, smemi[16]);}
				if(bx > 8) {minex(*smem, smem[8], *smemi, smemi[8]);}
				if(bx > 4) {minex(*smem, smem[4], *smemi, smemi[4]);}
				if(bx > 2) {minex(*smem, smem[2], *smemi, smemi[2]);}
				if(bx > 1) {minex(*smem, smem[1], *smemi, smemi[1]);}
								
				if (tid == 0) {
					t[blockIdx.x] = *smem;
					ti[blockIdx.x] = *smemi;

					if(gridDim.x == 1 && *smem < 1) {
						realreal dk = d[*smemi];
						const int k = ind[*smemi];
						if(dk > 0) {
							x[k] = u[k];
							d[*smemi] = 0;
						} else if(dk < 0)
						{
							x[k] = l[k];
							d[*smemi] = 0;
						}
					}
				}
			}
		}

		template<int bx>
		__global__
		void kernel31(
			const int n,
			const int* ind,
			const realreal* buf_in,
			const int* bufi_in,
			realreal* buf_out,
			int* bufi_out,
			realreal* d,
			realreal* x,
			const realreal* u,
			const realreal* l
			)
		{
			const int i = blockIdx.x * blockDim.x + threadIdx.x;
			const int tid = threadIdx.x;
			
			volatile __shared__ realreal sdata[bx];
			volatile __shared__ int sdatai[bx];

			realreal mySum;
			int mySumi;
			if(i < n) {
				mySum = buf_in[i];
				mySumi = bufi_in[i];
			} else {
				mySum = 1.0;
				mySumi = 0;
			}

			sdata[tid] = mySum;
			sdatai[tid] = mySumi;
			__syncthreads();
			if(bx > 512) {if (tid < 512) { minex(sdata[tid], sdata[tid + 512], sdatai[tid], sdatai[tid + 512]); } __syncthreads();}
			if(bx > 256) {if (tid < 256) { minex(sdata[tid], sdata[tid + 256], sdatai[tid], sdatai[tid + 256]); } __syncthreads();}
			if(bx > 128) {if (tid < 128) { minex(sdata[tid], sdata[tid + 128], sdatai[tid], sdatai[tid + 128]); } __syncthreads();}
			if(bx > 64) {if (tid <  64) { minex(sdata[tid], sdata[tid +  64], sdatai[tid], sdatai[tid + 64]); } __syncthreads();}
    
			if (tid < __min(bx / 2, 32))
			{
				// now that we are using warp-synchronous programming (below)
				// we need to declare our shared memory volatile so that the compiler
				// doesn't reorder stores to it and induce incorrect behavior.
				volatile realreal* smem = sdata + tid;
				volatile int* smemi = sdatai + tid;
				if(bx > 32) {minex(*smem, smem[32], *smemi, smemi[32]);}
				if(bx > 16) {minex(*smem, smem[16], *smemi, smemi[16]);}
				if(bx > 8) {minex(*smem, smem[8], *smemi, smemi[8]);}
				if(bx > 4) {minex(*smem, smem[4], *smemi, smemi[4]);}
				if(bx > 2) {minex(*smem, smem[2], *smemi, smemi[2]);}
				if(bx > 1) {minex(*smem, smem[1], *smemi, smemi[1]);}
								
				if (tid == 0) {
					buf_out[blockIdx.x] = *smem;
					bufi_out[blockIdx.x] = *smemi;
					
					if(gridDim.x == 1 && *smem < 1) {
						realreal dk = d[*smemi];
						const int k = ind[*smemi];
						if(dk > 0) {
							x[k] = u[k];
							d[*smemi] = 0;
						} else if(dk < 0)
						{
							x[k] = l[k];
							d[*smemi] = 0;
						}
					}
				}
			}
		}

		__global__
		void kernel32(
			const int nsub,
			const int* ind,
			realreal* x,
			const realreal* d,
			const realreal* alpha
			)
		{
			const int i = blockIdx.x * blockDim.x + threadIdx.x;

			__shared__ realreal salpha[1];

			if(i >= nsub)
				return;

			const int k = ind[i];

			if(threadIdx.x == 0) {
				*salpha = alpha[0];
			}
			realreal xi = x[k];
			realreal di = d[i];

			__syncthreads();
			
			x[k] = salpha[0] * di + xi;
		}

		void prog3
		(
			const int nsub,
			const int* ind,
			realreal* d,
			const int* nbd,
			realreal* buf_s_r,
			int* bufi_s_r,
			realreal* x,
			const realreal* u,
			const realreal* l,
			const cudaStream_t& stream
		)
		{
			//kernel30(nsub, d, nbd, buf_s_r, bufi_s_r, x, u, l, alpha);
			int nblock0 = nsub;
			int mi = log2Up(nblock0);
			int nblock1 = iDivUp2(nblock0, mi);

			realreal* output_r = buf_s_r;
			int* output_i = bufi_s_r;

			dynamicCall(kernel30, mi, nblock1, 1, stream, (nsub, ind, d, nbd, output_r, output_i, x, u, l));

/*
			kernel30<<<dim3(nblock1), dim3(512)>>>
				(nsub, d, nbd, output_r, output_i, x, u, l);*/
			
			CheckBuffer_int(output_i, nsub, nsub);
			CheckBuffer(output_r, nsub, nsub);
			nblock0 = nblock1;
			while(nblock0 > 1) {

				nblock1 = iDivUp2(nblock0, mi);

				realreal* input_r = output_r;
				int* input_i = output_i;

				output_r = output_r + nblock0;
				output_i = output_i + nblock0;

				dynamicCall(kernel31, mi, nblock1, 1, stream, (nblock0, ind, input_r, input_i, output_r, output_i, d, x, u, l));

/*
				kernel31<<<dim3(nblock1), dim3(512)>>>
					(nblock0, input_r, input_i, output_r, output_i, d, x, u, l);*/

				nblock0 = nblock1;
			}

			kernel32<<<dim3(iDivUp(nsub, 512)), dim3(512), 0, stream>>>
				(nsub, ind, x, d, output_r);

		}

	};
};