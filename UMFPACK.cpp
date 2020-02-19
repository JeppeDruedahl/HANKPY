//////////////
// INCLUDES //
//////////////

#include <windows.h>
#include <stdio.h>
#include <omp.h>
#include "umfpack.h"

#define EXPORT extern "C" __declspec(dllexport)

///////////////
// FUNCTIONS //
///////////////

void* symbolic(int n, int *Ap, int *Ai, double *Ax){

	double *null = (double *) NULL;

	// a. make symbolic pre-ordering
	void *Symbolic;
	umfpack_di_symbolic(n,n,Ap,Ai,Ax,&Symbolic,null,null);
	
	// b. return symbolic
	return Symbolic;

}

void solve_with_symbolic(int n, int *Ap, int *Ai, double *Ax, double *b, double *x, 
						 void *Symbolic, int *Wi, double *W){

	double *null = (double *) NULL;
	
	// a. numeric scaling and factorization
	void *Numeric;
	umfpack_di_numeric(Ap,Ai,Ax,Symbolic,&Numeric,null,null);

	// c. solve
	umfpack_di_wsolve(UMFPACK_A,Ap,Ai,Ax,x,b,Numeric,null,null,Wi,W);

	// d. clear
	umfpack_di_free_numeric(&Numeric);

}

////////////
// EXPORT //
////////////

// solve single
EXPORT void solve(int n, int *Ap, int *Ai, double *Ax, double *b, double *x){
	
	void *Symbolic = symbolic(n,Ap,Ai,Ax);
	
	int *Wi = new int[n];
	double *W = new double [5*n];
	
	solve_with_symbolic(n,Ap,Ai,Ax,b,x,Symbolic,Wi,W);
	umfpack_di_free_symbolic(&Symbolic);

	delete[] Wi;
	delete[] W;

}

// solve many
EXPORT void solve_many(int n, int k, int **Ap, int **Ai, double **Ax, double **b, double **x, 
					   void **Symbolics, int **Wi, double **W, bool do_symbolic, bool do_solve, bool do_free, int cppthreads){
	
	#pragma omp parallel num_threads(cppthreads)
	{

	#pragma omp for
	for(int i = 0; i < k; i++){
	
		// a. make symbolic pre-ordering on first run
		if(do_symbolic){
			Symbolics[i] = symbolic(n,Ap[i],Ai[i],Ax[i]);
		}
	
		// b. solve with knwon symbolic pre-ordering
		if(do_solve){
			solve_with_symbolic(n,Ap[i],Ai[i],Ax[i],b[i],x[i],Symbolics[i],Wi[i],W[i]);
		}

		// c. free memory for symbolic pre-ordering
		if(do_free){
			umfpack_di_free_symbolic(&Symbolics[i]);
		}

	} // loop
	} // parallel

}

EXPORT void free_many(int k, void **Symbolics){
	
	for(int i = 0; i < k; i++){
		umfpack_di_free_symbolic(&Symbolics[i]);
	} // loop

}

EXPORT void setup_omp() {
	SetEnvironmentVariable("OMP_WAIT_POLICY", "passive");
}


//////////
// MAIN //
//////////

int main (void) { return 0; }