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
	namespace active {
		__global__
		void kernel0(
			const int n,
			const realreal* l,
			const realreal* u,
			const int* nbd,
			realreal* x,
			int* iwhere
			)
		{
			const int i = blockIdx.x * blockDim.x + threadIdx.x;
			if(i >= n)
				return;

			int nbdi = nbd[i];
			realreal xi = x[i];
			realreal li = l[i];
			realreal ui = u[i];
			int iwi = -1;

			if(nbdi > 0) {
				if( nbdi <= 2 )
				{
					xi = maxr(xi, li);
				}
				else
				{
					xi = minr(xi, ui);
				}
			} 

			if(nbdi == 2 && ui - li <= 0) {
				iwi = 3;
			} else if(nbdi != 0)
			{
				iwi = 0;
			}

			x[i] = xi;
			iwhere[i] = iwi;
		}


		void prog0(
			const int& n,
			const realreal* l,
			const realreal* u,
			const int* nbd,
			realreal* x,
			int* iwhere
			) 
		{

			kernel0<<<dim3(iDivUp(n, 512)), dim3(512)>>>
				(n, l, u, nbd, x, iwhere);
			cudaError_t err = cudaGetLastError();
			if (err != cudaSuccess)
				printf("Error: %s\n", cudaGetErrorString(err));
			CheckBuffer_int(iwhere, n, n);
		}
	};
};