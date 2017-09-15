//#include <stdio.h>
#include <cuda.h>
#include <curand.h>
#include <curand_kernel.h>
//#include <cusolverDn.h>
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <float.h>
#include "../transportbase.h"
#include "knn_cuda_with_indexes.h"

// ERROR CHECKING MACROS //////////////////////////////////////////////////////

#define CUDA_CALL(x) { gpuAssert((x), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort
        =true)
{
    if (code != cudaSuccess) {
        fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file,
                line);
        cudaGetLastError();
//        cudaDeviceReset();
        cudaThreadExit();
        throw "CUDA Error.";
    }
}

//static const int max_shared_floats = 8000;

/**
 * Converts a CURAND error code from enum to text
 *
 * @param error as curandStatus_t
 * @return Error code as static const char
 */
static const char *curandGetErrorString(curandStatus_t error)
{
    switch (error)
    {
        case CURAND_STATUS_SUCCESS:
            return "CURAND_STATUS_SUCCESS";

        case CURAND_STATUS_VERSION_MISMATCH:
            return "CURAND_STATUS_VERSION_MISMATCH";

        case CURAND_STATUS_NOT_INITIALIZED:
            return "CURAND_STATUS_NOT_INITIALIZED";

        case CURAND_STATUS_ALLOCATION_FAILED:
            return "CURAND_STATUS_ALLOCATION_FAILED";

        case CURAND_STATUS_TYPE_ERROR:
            return "CURAND_STATUS_TYPE_ERROR";

        case CURAND_STATUS_OUT_OF_RANGE:
            return "CURAND_STATUS_OUT_OF_RANGE";

        case CURAND_STATUS_LENGTH_NOT_MULTIPLE:
            return "CURAND_STATUS_LENGTH_NOT_MULTIPLE";

        case CURAND_STATUS_DOUBLE_PRECISION_REQUIRED:
            return "CURAND_STATUS_DOUBLE_PRECISION_REQUIRED";

        case CURAND_STATUS_LAUNCH_FAILURE:
            return "CURAND_STATUS_LAUNCH_FAILURE";

        case CURAND_STATUS_PREEXISTING_FAILURE:
            return "CURAND_STATUS_PREEXISTING_FAILURE";

        case CURAND_STATUS_INITIALIZATION_FAILED:
            return "CURAND_STATUS_INITIALIZATION_FAILED";

        case CURAND_STATUS_ARCH_MISMATCH:
            return "CURAND_STATUS_ARCH_MISMATCH";

        case CURAND_STATUS_INTERNAL_ERROR:
            return "CURAND_STATUS_INTERNAL_ERROR";
    }

    return "<unknown>";
}

#define CURAND_CALL(x) { gpuAssert((x), __FILE__, __LINE__); }
inline void gpuAssert(curandStatus_t code, const char *file, int line, bool
        abort=true)
{
    if (code != CURAND_STATUS_SUCCESS) {
        fprintf(stderr,"GPUassert: %s %s %d\n",curandGetErrorString(code),
                file, line);
        cudaDeviceReset();

        throw "CURAND Error";
//        if (abort) throw std::exception();
    }
}

// ASSISTANT KERNELS //////////////////////////////////////////////////////////

// The generator is used for creating pseudo-random numbers for a given array
// of states (std_normal distribution)
__device__ float generateNormal(curandState* globalState, const unsigned int
        ind) {
    //copy state to local mem
    curandState localState = globalState[ind];
    //apply uniform distribution with calculated random
    float rndval = curand_normal( &localState );
    //update state
    globalState[ind] = localState;
    //return value
    return rndval;
}

// This kernel is used for finding the solution to a set of linear equations
// This is naive in that it assumes the coefficients matrix is non-singular.
// This should only be used for the linear regressions.
__device__ void solveLinearSystem(int dims, float *A, float *B, float *C) {
    // First generate upper triangular matrix for the augmented matrix
    float *swapRow;
    swapRow = (float*)malloc((dims+1)*sizeof(float));

    for (int ii = 0; ii < dims; ii++) {
        C[ii] = B[ii];
    }

    for (int ii = 0; ii < dims; ii++) {
        // Search for maximum in this column
        float maxElem = fabsf(A[ii*dims+ii]);
        int maxRow = ii;

        for (int jj = (ii+1); jj < dims; jj++) {
            if (fabsf(A[ii*dims+jj] > maxElem)) {
                maxElem = fabsf(A[ii*dims+jj]);
                maxRow = jj;
            }
        }

        // Swap maximum row with current row if needed
        if (maxRow != ii) {
            for (int jj = ii; jj < dims; jj++) {
                swapRow[jj] = A[jj*dims+ii];
                A[jj*dims+ii] = A[jj*dims+maxRow];
                A[jj*dims+maxRow] = swapRow[jj];
            }

            swapRow[dims] = C[ii];
            C[ii] = C[maxRow];
            C[maxRow] = swapRow[dims];
        }

        // Make all rows below this one 0 in current column
        for (int jj = (ii+1); jj < dims; jj++) {
            float factor = -A[ii*dims+jj]/A[ii*dims+ii];

            // Work across columns
            for (int kk = ii; kk < dims; kk++) {
                if (kk == ii) {
                    A[kk*dims+jj] = 0.0;
                } else {
                    A[kk*dims+jj] += factor*A[kk*dims+ii];
                }
            }

            // Results vector
            C[jj] += factor*C[ii];
        }
    }
    free(swapRow);

    // Solve equation for an upper triangular matrix
    for (int ii = dims-1; ii >= 0; ii--) {
        C[ii] = C[ii]/A[ii*dims+ii];

        for (int jj = ii-1; jj >= 0; jj--) {
            C[jj] -= C[ii]*A[ii*dims+jj];
        }
    }
}

// Initialise the states for curand on each kernel
__global__ void initialise_curand_on_kernels(curandState* state,
        unsigned long seed) {
    int idx = blockIdx.x*blockDim.x+threadIdx.x;
    curand_init(seed, idx, 0, &state[idx]);
}

// Generate the random numbers required for use in another function
__global__ void set_random_number_from_kernels(float* _ptr, curandState*
        globalState, const unsigned int _points, const unsigned int
        dimension1, const unsigned int dimension2 = 1, const unsigned int
        dimension3 = 1) {

    // Get the global index for the matrix
    unsigned int idx = blockIdx.x*blockDim.x+threadIdx.x;

    // Each generator operates on a different set of nYears * nPaths
    unsigned int stateNo = (unsigned int)(idx / (dimension2*dimension3));

    //only call gen on the kernels we have inited
    //(one per device container element)
    if (stateNo < dimension1) {
        if (idx < _points)
        {
            _ptr[idx] = generateNormal(&globalState[stateNo], idx);
        }
    }
}

// These kernels are used for knn search


// MAIN KERNELS ///////////////////////////////////////////////////////////////

// Kernel for computing a single path of an uncertain variable
__global__ void expPVPath(const int noPaths, const float gr, const int nYears,
        const float meanP, const float timeStep, const float rrr, float
        current, float reversion, float jumpProb, const float* brownian, const
        float* jumpSize, const float* jump, float* result) {

    // Get the global index for the matrix
    unsigned int idx = blockIdx.x*blockDim.x+threadIdx.x;

    if (idx < noPaths) {
        // Simulate a forward path
        float value = 0;
        float curr = current;

        for (int ii = 0; ii < nYears; ii++) {
            float jumped = (jump[idx+ii] < jumpProb)? 1.0f : 0.0f;

            curr += reversion*(meanP - curr)*timeStep + curr*brownian[idx+ii] +
                    (exp(jumpSize[idx+ii]) - 1)*curr*jumped;
            value += pow(1 + gr,ii)*curr/pow((1 + rrr),ii);
        }

        result[idx] = value;
    }
}

// The matrix multiplication kernel parallelises the multiplication of Eigen
// matrices
__global__ void matrixMultiplicationKernelNaive(const float* A, const float* B,
        float* C, int a, int b, int c, int d) {

    int ROW = blockIdx.y*blockDim.y+threadIdx.y;
    int COL = blockIdx.x*blockDim.x+threadIdx.x;

    float tmpSum = 0.0f;

    if (ROW < a && COL < d) {
        // each thread computes one element of the block sub-matrix
        for (int ii = 0; ii < b; ii++) {
            tmpSum += A[ROW * b + ii] * B[ii * b + COL];
        }
    }
    C[ROW * a + COL] = tmpSum;
}

// The matrix element-wise multiplication kernel parallelises the element-wise
// multiplication of Eigen matrices

__global__ void matrixElementWiseMultiplicationKernelNaive(const float* A,
        const float* B, float* C, int a, int b) {

    int ROW = blockIdx.y*blockDim.y+threadIdx.y;
    int COL = blockIdx.x*blockDim.x+threadIdx.x;

    if (ROW < a && COL < b) {
        C[ROW * a + COL] = A[ROW * b + COL]*B[ROW * b + COL];
    }
}

// The optimised matrix multiplication kernel that relies on efficient memory
// management
__global__ void matrixMultiplicationKernel(float *A, float* B, float* C, int a,
        int b, int d) {

    // Block index
    int bx = blockIdx.x;
    int by = blockIdx.y;

    // Thread index
    int tx = threadIdx.x;
    int ty = threadIdx.y;

    int ROW = by*blockDim.y+ty;
    int COL = bx*blockDim.x+tx;

    // First check if the thread exceeds the matrix dimensions
    if (ROW < a && COL < d) {

        // Declaration of the shared memory array As used to store the sub-
        // matrix of A
        __shared__ float As[BLOCK_SIZE * BLOCK_SIZE];
        __shared__ float As2[BLOCK_SIZE * BLOCK_SIZE];

        float *prefetch = As;
        float *prefetch2 = As2;

        // Declaration of the shared memory array Bs used to
        // store the sub-matrix of B
        // __shared__ float Bs[BLOCK_SIZE][BLOCK_SIZE];

        float cv[BLOCK_SIZE];

        for (int ii = 0; ii < BLOCK_SIZE; ii++) {
             cv[ii] = 0;
        }

        // Index of the first sub-matrix of A processed by the block
        int aBegin = a * BLOCK_SIZE * by;

        // Index of the last sub-matrix of A processed by the block
        int aEnd   = aBegin + a - 1;

        // Step size used to iterate through the sub-matrices of A
        int aStep  = BLOCK_SIZE;

        // Index of the first sub-matrix of B processed by the block
        int bBegin = BLOCK_SIZE * VECTOR_SIZE * bx;

        // Step size used to iterate through the sub-matrices of B
        int bStep  = BLOCK_SIZE * d;

        int cBegin = d * BLOCK_SIZE * by + VECTOR_SIZE * BLOCK_SIZE * bx;

        // Csub is used to store the element of the block sub-matrix
        // that is computed by the thread
        // float Csub = 0;
        float *Ap = &A[aBegin + a * ty +tx];
        float *ap = &prefetch[ty + BLOCK_SIZE * tx];
#pragma unroll
        for(int ii = 0; ii < BLOCK_SIZE; ii+=4){
          ap[ii] = Ap[a * ii];
        }
        __syncthreads();

        // Loop over all the sub-matrices of A and B
        // required to compute the block sub-matrix
        for (int a = aBegin, b = bBegin;
             a <= aEnd;
             a += aStep, b += bStep) {

            // Load the matrices from device memory
            // to shared memory; each thread loads
            // one element of each matrix
            Ap = &A[a + aStep + a * ty +tx];
            float *ap2 = &prefetch2[ty + BLOCK_SIZE * tx];
#pragma unroll
            for(int ii = 0; ii < BLOCK_SIZE; ii+=4){
                ap2[ii] = Ap[b * ii];
            }

            ap = &prefetch[0];
            float *bp = &B[b + BLOCK_SIZE * ty + tx];

#pragma unroll
            for (int ii = 0; ii < BLOCK_SIZE; ii++) {
                float bv = bp[0];
                for (int jj = 0; jj < BLOCK_SIZE; jj++) {
                    cv[jj] += ap[jj]*bv;
                    ap += BLOCK_SIZE;
                    bp += d;
                }
            }

            // Synchronize to make sure the matrices are loaded
            __syncthreads();

            // swap As and As2
            float *prefetch_temp = prefetch;
            prefetch = prefetch2;
            prefetch2 = prefetch_temp;
        }

        // Write the block sub-matrix to device memory;
        // each thread writes one element
        float *Cp = &C[cBegin];
        Cp += BLOCK_SIZE * ty + tx;
        int cStep = d;
#pragma unroll
        for(int ii=0; ii<BLOCK_SIZE; ii++){
          Cp[0] = cv[ii]; Cp += cStep;
        }
    }
}

// Element-wise matrix multiplication kernel
__global__ void matrixMultiplicationKernelEW(const float* A, const float*
        B, float* C, int a, int b) {

    int ROW = blockIdx.y*blockDim.y+threadIdx.y;
    int COL = blockIdx.x*blockDim.x+threadIdx.x;

    if (ROW < a && COL < b) {
        C[ROW * a + COL] = A[ROW * b + COL]*B[ROW * b + COL];
    }
}

// Element-wise matrix division kernel
__global__ void matrixDivisionKernelEW(const float* A, const float* B,
        float* C, int a, int b) {

    int ROW = blockIdx.y*blockDim.y+threadIdx.y;
    int COL = blockIdx.x*blockDim.x+threadIdx.x;

    if (ROW < a && COL < b) {
        C[ROW * a + COL] = A[ROW * b + COL]/B[ROW * b + COL];
    }
}

// Computes whether there is an intersection between line segements or not
__global__ void pathAdjacencyKernel(int noTransitions, int noSegments,
        float* XY1, float* XY2, float* X4_X3, float* Y4_Y3, float* X2_X1,
        float* Y2_Y1, int* adjacency) {

    int blockId = blockIdx.y * gridDim.x + blockIdx.x;
    int idx = blockId * blockDim.x + threadIdx.x;

    if (idx < noTransitions*noSegments) {
        int seg1 = idx/noSegments;
        int seg2 = idx - seg1*noSegments;

        float Y1_Y3 = XY1[seg1 + noTransitions] - XY2[seg2 + noSegments];
        float X1_X3 = XY1[seg1] - XY2[seg2];

        float numa = X4_X3[seg2]*Y1_Y3 - Y4_Y3[seg2]*X1_X3;
        float numb = X2_X1[seg1]*Y1_Y3 - Y2_Y1[seg1]*X1_X3;
        float deno = Y4_Y3[seg2]*X2_X1[seg1] - X4_X3[seg2]*Y2_Y1[seg1];

        float u_a = numa/deno;
        float u_b = numb/deno;

        adjacency[idx] = (int)((u_a >= 0.0) && (u_a <= 1.0) && (u_b >= 0.0)
                && (u_b <= 1.0));
    }
}

// Sums the line segments intersection values along the each row
__global__ void roadCrossingsKernel(int rows, int segs, int* adjacency,
        int* cross) {

    int idx = blockIdx.x*blockDim.x + threadIdx.x;

    if (idx < rows) {
        cross[idx] = 0;

        for (int ii = 0; ii < segs; ii++) {
            cross[idx] += adjacency[idx*segs + ii];
        }
    }
}

// The patch kernel represents a single cell for generating habitat patches
// The results matrix contains the following:
//
__global__ void patchComputation(int noCandidates, int W, int H, int skpx, int
        skpy, int xres, int yres, float subPatchArea, float xspacing, float
        yspacing, float capacity, int uniqueRegions, const int* labelledImage,
        const float* pops, float* results) {

    // Get global index of thread
    int idx = blockIdx.x*blockDim.x + threadIdx.x;

    if (idx < noCandidates) {
        // Dimensions arranged as X->Y->R
        int rem = idx;
        int blockIdxY = (int)(idx/(xres*uniqueRegions));
        rem = rem - blockIdxY*(xres*uniqueRegions);
        int blockIdxX = (int)(rem/uniqueRegions);
        rem = rem - blockIdxX*(uniqueRegions);
        // Valid region numbering starts at 1, not 0
        int regionNo = rem + 1;

        int blockSizeX;
        int blockSizeY;

        if ((blockIdxX+1)*skpx <= H) {
            blockSizeX = skpx;
        } else {
            blockSizeX = H-blockIdxX*skpx;
        }

        if ((blockIdxY+1)*skpy <= W) {
            blockSizeY = skpy;
        } else {
            blockSizeY = W-blockIdxY*skpy;
        }

        // Iterate through each sub patch for this large grid cell
        float area = 0.0f;
        float cap = 0.0f;
        float pop = 0.0f;
        float cx = 0.0f;
        float cy = 0.0f;

        for (int ii = 0; ii < blockSizeX; ii++) {
            for (int jj = 0; jj < blockSizeY; jj++) {
                int xCoord = blockIdxX*skpx+ii;
                int yCoord = blockIdxY*skpy+jj;

                area += (float)(labelledImage[xCoord + yCoord*W] == regionNo);
            }
        }

        if (area > 0) {
            for (int ii = 0; ii < blockSizeX; ii++) {
                for (int jj = 0; jj < blockSizeY; jj++) {
                    int xCoord = blockIdxX*skpx+ii;
                    int yCoord = blockIdxY*skpy+jj;

                    if (labelledImage[xCoord + yCoord*W] == regionNo) {
                        pop += (float)pops[xCoord + yCoord*W];
                        cx += ii;
                        cy += jj;
                    }
                }
            }
            cx = xspacing*(cx/area + blockIdxX*skpx);
            cy = yspacing*(cy/area + blockIdxY*skpy);
            area = area*subPatchArea;
            cap = area*capacity;
        }

        // Store results to output matrix
        results[5*idx] = area;
        results[5*idx+1] = cap;
        results[5*idx+2] = pop;
        results[5*idx+3] = cx;
        results[5*idx+4] = cy;
    }
}

// Computes the movement and mortality of a species from the forward path
// kernels
__global__ void mmKernel(float* popsIn, float* popsOut, float* mmm, int patches) {
    int ii = threadIdx.x;

    if (ii < patches) {
        extern __shared__ float s[];

        s[ii] = 0.0;

        for (int jj = 0; jj < patches; jj++) {
            s[ii] += popsIn[ii]*mmm[ii*patches + jj];
        }
        __syncthreads();

        popsOut[ii] = s[ii];
    }
}

// The mte kernel represents a single path for mte
__global__ void mteKernel(int noPaths, int nYears, int noPatches, float
        timeStep, float* rgr, float* brownians, float* jumpSizes, float* jumps,
        float* speciesParams, float *initPops, float* caps, float*mmm, int*
        rowIdx, int* elemsPerCol, float* pathPops, float* eps) {
    // Global index for finding the thread number
    int ii = blockIdx.x*blockDim.x + threadIdx.x;

    // Only perform matrix multiplication sequentially for now. Later, if
    // so desired, we can use dynamic parallelism because the card in the
    // machine has CUDA compute capability 3.5
    if (ii < noPaths) {
        //extern __shared__ float s[];

        // Initialise the prevailing population vector
        for (int jj = 0; jj < noPatches; jj++) {
            pathPops[(ii*2)*noPatches+jj] = initPops[jj];
        }

        float grMean = speciesParams[0];

        for (int jj = 0; jj < nYears; jj++) {
            // Movement and mortality. This component is very slow without
            // using shared memory. As we do not know the size of the patches
            // at compile time, we need to be careful how much shared memory we
            // allocate. For safety, we assume that we will have less than
            // 64KB worth of patch data in the mmm matrix. Using single
            // precision floating point numbers, this means that we can only
            // have up to 8,000 patches. As this number is extremely large, we
            // set a limit outside this routine to have at most 300 patches.
            for (int kk = 0; kk < noPatches; kk++) {
                pathPops[(ii*2+1)*noPatches+kk] = 0.0;
            }

            int iterator = 0;
            for (int kk = 0; kk < noPatches; kk++) {
                for (int ll = 0; ll < elemsPerCol[kk]; ll++) {
                    pathPops[(ii*2+1)*noPatches+kk] += pathPops[(ii*2)*
                            noPatches+rowIdx[iterator]]*mmm[iterator];
                    iterator++;
                }
            }

            // UPDATE: NEED TO IMPLEMENT SHARED MEMORY AS WELL

            // DEPRECATED - TO BE DELETED AT LATER STAGE
            // Load the correct slice of the mmm matrix for each
            // destination patch. Use the thread index as a helper to do
            // this. Wait for all information to be loaded in before
            // proceeding. We need to tile the mmm matrix here to obtain
            // a sufficient speed up.

//            for (int kk = 0; kk < noTiles; kk++) {
//                int currDim = tileDim;

//                if (threadIdx.x < noPatches) {
//                    // First, allocate the memory for this tile
//                    if (kk == noTiles-1) {
//                        currDim = (int)(noTiles*tileDim == noPatches) ?
//                                (int)tileDim : (int)(noPatches - kk*tileDim);
//                    }

//                    for (int ll = 0; ll < currDim; ll++) {
//                        s[ll*noPatches + threadIdx.x] = mmm[kk*noPatches*
//                                tileDim + ll*noPatches + threadIdx.x];
//                    }
//                }
//                __syncthreads();

//                // Now increment the populations for this path
//                for (int kk = 0; kk < currDim; kk++) {
//                    for (int ll = 0; ll < noPatches; ll++) {
//                        pathPops[(ii*2+1)*noPatches+kk] += pathPops[(ii*2)*
//                                noPatches+ll]*s[kk*noPatches + ll];
//                    }
//                }
//            }

//            for (int kk = 0; kk < noPatches; kk++) {
//                for (int ll = 0; ll < noPatches; ll++) {
////                    pathPops[(ii*2+1)*noPatches+kk] += pathPops[(ii*2)*
////                            noPatches+ll]*s[ll];
//                    pathPops[(ii*2+1)*noPatches+kk] += pathPops[(ii*2)*
//                            noPatches+ll]*mmm[kk*noPatches+ll];
//                }
//            }

//            matrixMultiplicationKernel<<<noBlocks,noThreadsPerBlock>>>(pathPops
//                    + (ii*2)*noPatches, mmm, pathPops + (ii*2+1)*noPatches, 1,
//                    noPatches, noPatches);
//            cudaDeviceSynchronize();
//            __syncthreads();

            // Natural birth and death

            // Adjust the global growth rate mean for this species at this
            // time step for this path.
            float jump = (jumps[ii*nYears + jj] < speciesParams[6]) ? 1.0f :
                    0.0f;
            float meanP = speciesParams[1];
            float reversion = speciesParams[4];

            float brownian = brownians[ii*nYears + jj]*speciesParams[2];
            float jumpSize = jumpSizes[ii*nYears + jj]*pow(speciesParams[5],2)
                    - pow(speciesParams[5],2)/2;

            grMean = grMean + reversion*(meanP - grMean)*timeStep + grMean
                    *brownian + (exp(jumpSize) - 1)*grMean*jump;

            for (int kk = 0; kk < noPatches; kk++) {
                float gr = speciesParams[7]*rgr[ii*(nYears*noPatches) + jj*
                        noPatches + kk]*grMean + grMean;
                pathPops[(ii*2)*noPatches+kk] = pathPops[(ii*2+1)*noPatches+kk]
                        *(1.0f + gr*(caps[kk]-pathPops[(ii*2+1)*noPatches+kk])/
                        caps[kk]);
            }
        }

        eps[ii] = 0.0f;
        for (int jj = 0; jj < noPatches; jj++) {
            eps[ii] += pathPops[(ii*2+1)*noPatches+jj];
        }
    }
}

// The kernel for converting random floats to corresponding controls
__global__ void randControls(int noPaths, int nYears, int noControls, float*
        randCont, int* control) {
    int idx = blockIdx.x*blockDim.x + threadIdx.x;

    if (idx < noPaths*nYears) {
        control[idx] = (int)(randCont[idx]*noControls);
        if (control[idx] == noControls) {
            control[idx]--;
        }
    }
}

// Test routine for debugging purposes.
__global__ void printControls(int noPaths, int path, int nYears, int*
        controls) {
    for (int ii = 0; ii < nYears; ii++) {
        printf("%d %d\n",ii,controls[path*nYears + ii]);
    }
}

// Test routine for debugging purposes
__global__ void printAverages(int nYears, int noSpecies, int noControls,
        int noPaths, float* totalPops, float* aars) {
    int idx = blockIdx.x*blockDim.x + threadIdx.x;

    if (idx < nYears) {
        float* totals, *aar;
        totals = (float*)malloc(noSpecies*sizeof(float));

        aar = (float*)malloc(noSpecies*noControls*sizeof(float));

        for (int ii = 0; ii < noSpecies; ii++) {
            totals[ii] = 0.0f;
            for (int kk = 0; kk < noControls; kk++) {
                aar[ii*noControls + kk] = 0;
            }
        }

        for (int ii = 0; ii < noPaths; ii++) {
            for (int jj = 0; jj < noSpecies; jj++) {
                totals[jj] += totalPops[ii*noSpecies*(nYears+1) + (idx+1)*
                        noSpecies + jj];
                for (int kk = 0; kk < noControls; kk++) {
                    aar[jj*noControls + kk] += aars[ii*(nYears+1)*noControls*
                            noSpecies + idx*noControls*noSpecies + jj*
                            noControls + kk];
                }
            }
        }

        for (int ii = 0; ii < noSpecies; ii++) {
            totals[ii] = totals[ii]/(float)noPaths;
            for (int jj = 0; jj < noControls; jj++) {
                aar[ii*noControls + jj] = aar[ii*noControls + jj]/(float)
                        noPaths;
            }
        }

        printf("Year: %d Total: %f C1: %f C2: %f C3: %f\n", idx,totals[0],aar[0],aar[1],aar[2]);

        free(totals);
        free(aar);
    }
}

// The kernel for computing forward paths in ROV. This routine considers
// each patch as containing a certain number of each species. We call this
// routine at the start of the ROV routine. This routine uses the randomised
// controls.
__global__ void forwardPathKernel(int noPaths, int nYears, int noSpecies,
        int noPatches, int noControls, int noUncertainties, float timeStep,
        float* initPops, float* pops, float*mmm, int* rowIdx, int* elemsPerCol,
        int maxElems, float* speciesParams, float* caps, float* aars, float*
        uncertParams, int* controls, float* uJumps, float* uBrownian, float*
        uJumpSizes, float* uJumpsSpecies, float* uBrownianSpecies, float*
        uJumpSizesSpecies, float* rgr, float* uResults, float* totalPops) {

    // Global thread index
    int idx = blockIdx.x*blockDim.x + threadIdx.x;

    // Only perform matrix multiplication sequentially for now. Later, if so
    // desired, we can use dynamic parallelism because the card in the
    // machine has CUDA compute compatability 3.5

    if (idx < noPaths) {
        // Initialise the population data at time t=0
        for (int ii = 0; ii < noSpecies; ii++) {
            float population = 0;
            for (int jj = 0; jj < noPatches; jj++) {
                pops[idx*(nYears+1)*noSpecies*noPatches + ii*noPatches + jj] =
                        initPops[jj];
                population += pops[idx*(nYears+1)*noSpecies*noPatches + ii*
                        noPatches + jj];
            }
            totalPops[idx*(nYears+1)*noSpecies + ii] = population;

            // The aars are computed in the next for loop.
        }

        // Carry over the initial value for all uncertainties
        for (int ii = 0; ii < noUncertainties; ii++) {
            uResults[idx*noUncertainties*nYears + ii] = uncertParams[ii*6];
        }

        float* grMean;
        grMean = (float*)malloc(noSpecies*sizeof(float));

        for (int ii = 0; ii < noSpecies; ii++) {
            grMean[ii] = speciesParams[ii*8];
        }

        // All future time periods
        for (int ii = 0; ii <= nYears; ii++) {
            // Control to pick
            int control = controls[idx*nYears + ii];

            for (int jj = 0; jj < noSpecies; jj++) {
                totalPops[idx*(nYears+1)*noSpecies + (ii+1)*noSpecies + jj] =
                        0;

                // Adjust the global growth rate mean for this species at this
                // time step for this path.
                float jump = (uJumpsSpecies[idx*noSpecies*nYears +
                        ii*noSpecies + jj] < speciesParams[jj*8 + 5]) ?
                        1.0f : 0.0f;
                float meanP = speciesParams[jj*8 + 1];
                float reversion = speciesParams[jj*8 + 4];

                float brownian = uBrownianSpecies[idx*noSpecies*nYears +
                        ii*noSpecies + jj]*speciesParams[jj*8 + 2];
                float jumpSize = uJumpSizesSpecies[idx*noSpecies*nYears
                        + ii*noSpecies + jj]*pow(speciesParams[
                        jj*8 + 5],2) - pow(speciesParams[jj*8 + 5],2)/2;

                grMean[jj] = grMean[jj] + reversion*(meanP - grMean[jj])*
                        timeStep + grMean[jj]*brownian + (exp(jumpSize) - 1)*
                        grMean[jj]*jump;

                // Initialise temporary populations
                float initialPopulation = 0.0f;

                for (int kk = 0; kk < noPatches; kk++) {
                    initialPopulation += pops[idx*(nYears+1)*noSpecies*
                            noPatches + ii*noSpecies*noPatches + jj*noPatches
                            + kk];
                }

                // For each patch, update the population for the next time
                // period by using the movement and mortality matrix for the
                // correct species/control combination. We use registers due
                // to their considerably lower latency over global memory.
                for (int kk = 0; kk < noControls; kk++) {
                    // Overall population at this time period
                    float totalPop = 0.0f;

                    int iterator = 0;
                    for (int ll = 0; ll < noPatches; ll++) {
                        // Population for this patch
                        float population = 0.0f;

                        // Transfer animals from each destination patch to
                        // this one for the next period.
                        for (int mm = 0; mm < elemsPerCol[(jj*noControls + kk)*
                                noPatches + ll]; mm++) {

                            float value = pops[idx*(nYears+1)*noSpecies*
                                    noPatches + ii*noSpecies*noPatches + jj*
                                    noPatches + rowIdx[iterator + (jj*
                                    noControls + kk)*maxElems]]*mmm[iterator +
                                    (jj*noControls + kk)*maxElems];

                            population += value;

                            iterator++;
                        }

                        totalPop += population;

                        // We only update the actual populations if we are in
                        // the control that was selected.
                        if (kk == control && ii < nYears) {
                            // Population growth based on a mean-reverting process
                            rgr[idx*noSpecies*noPatches*nYears + ii*noSpecies*
                                    noPatches + jj*noPatches + ll] = grMean[jj]
                                    + rgr[idx*noSpecies*noPatches*nYears + ii*
                                    noSpecies*noPatches + jj*noPatches + ll]*
                                    speciesParams[jj*8 + 7];

                            float gr = rgr[idx*noSpecies*noPatches*nYears + ii*
                                    noSpecies*noPatches + jj*noPatches + ll];

                            pops[idx*(nYears+1)*noSpecies*noPatches + (ii+1)*
                                    noSpecies*noPatches + jj*noPatches + ll] =
                                    population*(1.0f + gr*(caps[jj*noPatches +
                                    ll] - population)/caps[jj*noPatches + ll]/
                                    100.0);
                            totalPops[idx*noSpecies*(nYears+1) + (ii+1)*
                                    noSpecies + jj] += pops[idx*(nYears+1)*
                                    noSpecies*noPatches + (ii+1)*noSpecies*
                                    noPatches + jj*noPatches + ll];
                        }
                    }
                    // Save AAR for this control
                    aars[idx*(nYears+1)*noControls*noSpecies + ii*noControls*
                            noSpecies + jj*noControls + kk] = totalPop/
                            initialPopulation;
                }
            }

            // Other uncertainties

            for (int jj = 0; jj < noUncertainties; jj++) {
                float jump = (uJumps[idx*noUncertainties*nYears +
                        ii*noUncertainties + jj] < uncertParams[jj*6 + 5]) ?
                        1.0f : 0.0f;

                float curr = uResults[idx*noUncertainties*nYears +
                        ii*noUncertainties + jj];
                float meanP = uncertParams[jj*6 + 1];
                float reversion = uncertParams[jj*6 + 3];

                float brownian = uBrownian[idx*noUncertainties*nYears +
                        ii*noUncertainties + jj]*uncertParams[jj*6 + 2];
                float jumpSize = uJumpSizes[idx*noUncertainties*nYears +
                        ii*noUncertainties + jj]*pow(uncertParams[jj*6 + 4],2)
                        - pow(uncertParams[jj*6 + 4],2)/2;

                uResults[idx*noUncertainties*nYears+(ii+1)*noUncertainties+jj]
                        = curr + reversion*(meanP - curr)*timeStep +
                        curr*brownian + (exp(jumpSize) - 1)*curr*jump;
            }
        }
        free(grMean);
    }
}

// Routine for evaluating the state variables for each path for ROV.
__global__ void computePathStates(int noPaths, int noDims, int nYears, int
        noControls, int year, float unitCost, float unitRevenue, int* controls,
        int noFuels, float *fuelCosts, float *uResults, float *uComposition,
        int noUncertainties, int *fuelIdx, int noCommodities, float* aars,
        float* totalPops, float* xin, int* currControls) {

    // Global thread index
    int idx = blockIdx.x*blockDim.x + threadIdx.x;

    if (idx < noPaths) {

        // 1. Adjusted population for each species
        for (int ii = 0; ii < noDims-1; ii++) {
            xin[idx*noDims + ii] = totalPops[idx*(noDims-1)*(nYears+1) + year*
                    (noDims-1) + ii]*aars[idx*(nYears+1)*noControls*(noDims-1)
                    + year*noControls*(noDims-1) + ii*noControls + controls[
                    idx*nYears + year]];
        }

        // 2. Unit profit
        float unitFuel = 0.0;
        float orePrice = 0.0;

        // Compute the unit fuel cost component
        for (int ii = 0; ii < noFuels; ii++) {
            unitFuel += fuelCosts[ii]*uResults[idx*nYears*noUncertainties +
                    (year)*noUncertainties + fuelIdx[ii]];
        }
        // Compute the unit revenue from ore
        for (int ii = 0; ii < noCommodities; ii++) {
            orePrice += uComposition[idx*nYears*noCommodities + (year)*
                    noCommodities + ii]*uResults[idx*nYears*noUncertainties +
                    (year)*noUncertainties + noFuels + ii];
        }

        xin[idx*noDims + noDims-1] = unitCost + unitFuel - unitRevenue*
                orePrice;
        currControls[idx] = controls[idx*nYears + year];

//        printf("%f %f\n",unitFuel,orePrice);
    }
}

// Allocates X (predictors) and corresponding response variables for multiple
// local linear regression.
__global__ void allocateXYRegressionData(int noPaths, int noControls, int
        noDims, int nYears, float* speciesParams, int year, int* controls,
        float* xin, float *condExp, int *dataPoints, float *xvals, float
        *yvals) {

    for (int ii = 0; ii < noControls; ii++) {
        dataPoints[ii] = 0;
    }

    // For each path
    for (int ii = 0; ii < noPaths; ii++) {
        if (controls[ii] >= noControls) {
            printf("Invalid control %d\n",controls[ii]);
        }

        yvals[noPaths*controls[ii] + dataPoints[controls[ii]]] = condExp[(
                year + 1)*noPaths + ii];

        // Save the input dimension values to the corresponding data group
        for (int jj = 0; jj < noDims; jj++) {
            xvals[controls[ii]*noPaths*noDims + jj*noPaths + dataPoints[
                    controls[ii]]] = xin[ii*noDims + jj];
        }

        // Increment the number of data points for this control
        dataPoints[controls[ii]] += 1;

//        // First check that the path is in-the-money. If it isn't we do not use
//        // it
//        bool valid = true;
//        for (int jj = 0; jj < (noDims-1); jj++) {
//            if (xin[ii*noDims + jj] < speciesParams[8*jj + 3]) {
//                valid = false;
//                break;
//            }
//        }

//        if (valid || controls[ii] == 0) {
//            // Save the input dimension values to the corresponding data group
//            for (int jj = 0; jj < noDims; jj++) {
//                xvals[controls[ii]*noPaths*noDims + jj*noPaths + dataPoints[
//                        controls[ii]]] = xin[ii*noDims + jj];
//            }

//            // Increment the number of data points for this control
//            dataPoints[controls[ii]] += 1;
//        }
//        printf("%6d | %3d: %6.0f %15.0f %15.0f\n",ii,controls[ii],xin[ii*noDims],xin[ii*noDims + 1],condExp[year*noPaths + ii]);
    }
}

// Find the minimum and maximum values for each state for each control from the
// X predictors for the regression data.
__global__ void computeStateMinMax(int noControls, int noDims, int noPaths,
        int* dataPoints, float* xvals, float* xmins, float* xmaxes) {

    for (int ii = 0; ii < noControls; ii++) {
        float *xmin, *xmax;
        xmin = (float*)malloc(noDims*sizeof(float));
        xmax = (float*)malloc(noDims*sizeof(float));

        for (int jj = 0; jj < noDims; jj++) {
            xmin[jj] = xvals[ii*noDims*noPaths + jj*noPaths];
            xmax[jj] = xmin[jj];
        }

        for (int jj = 0; jj < noDims; jj++) {
            for (int kk = 0; kk < dataPoints[ii]; kk++) {
                float xtemp = xvals[ii*noDims*noPaths + jj*noPaths + kk];
                if (xmin[jj] > xtemp) {
                    xmin[jj] = xtemp;
                } else if (xmax[jj] < xtemp) {
                    xmax[jj] = xtemp;
                }
            }
        }

//        for (int jj = 0; jj < noDims; jj++) {
//            xmin[jj] = xvals[ii*noDims*noPaths + jj];
//            xmax[jj] = xmin[jj];
//        }

//        for (int jj = 0; jj < dataPoints[ii]; jj++) {
//            for (int kk = 0; kk < noDims; kk ++) {
//                float xtemp = xvals[ii*noDims*noPaths + jj*noDims + kk];
//                if (xmin[kk] > xtemp) {
//                    xmin[kk] = xtemp;
//                } else if (xmax[kk] < xtemp) {
//                    xmax[kk] = xtemp;
//                }
//            }
//        }

        for (int jj = 0; jj < noDims; jj++) {
            xmins[ii*noDims + jj] = xmin[jj];
            xmaxes[ii*noDims + jj] = xmax[jj];
//            printf("Control %d: Xmin = %f Xmax = %f\n",ii,xmin[jj],xmax[jj]);
        }

        free(xmin);
        free(xmax);
    }
}

// Computes regularly-spaced query points for a multiple local linear
// regression
__global__ void createQueryPoints(int noPoints, int noDims, int dimRes, int
        control, int noControls, int year, float* xmins, float* xmaxes, float*
        regression, float* queryPts) {

    // Global thread index
    int idx = blockIdx.x*blockDim.x + threadIdx.x;

    if (idx < noPoints) {

        // First, deconstruct the index into the index along each dimension
        int *dimIdx;
        dimIdx = (int*)malloc(noDims*sizeof(int));

        int rem = idx;

        for (int ii = 0; ii < noDims; ii++) {
            int div = (int)(rem/pow(dimRes,noDims-ii-1));
            dimIdx[ii] = div;
            rem = rem - div*pow(dimRes,noDims-ii-1);
        }

        // Get the query point coordinates
        for (int ii = 0; ii < noDims; ii++) {
            queryPts[idx + ii*noPoints] = ((float)dimIdx[ii])*(xmaxes[
                    control*noDims + ii] - xmins[control*noDims + ii])/(
                    float)(dimRes-1) + xmins[control*noDims + ii];

            // Save the X value for the query point
            regression[year*noControls*(dimRes*noDims + (int)pow(dimRes,
                    noDims)*2) + control*(dimRes*noDims + (int)pow(dimRes,
                    noDims)*2) + ii*dimRes + dimIdx[ii]] = queryPts[idx + ii*
                    noPoints];
        }

        free(dimIdx);
    }
}

// Recomputes the optimal forward paths for each time period in the backward
// induction in the ROV with endogenous uncertainty.
__global__ void optimalForwardPaths(int start, int noPaths, int nYears, int
        noSpecies, int noPatches, int noControls, int noUncertainties, float
        timeStep, float unitCost, float unitRevenue, float rrr, int noFuels,
        int noCommodities, int dimRes, float* Q, float* fuelCosts, float* pops,
        float* totalPops,float*mmm, int* rowIdx, int* elemsPerCol, int
        maxElems, float* speciesParams, float* rgr, float* caps, int* controls,
        float* aars, float* regression, float* uComposition, float* uResults,
        int* fuelIdx, float* condExp, int* optCont, float* adjPops, float
        *unitProfits) {

    // Global thread index (path number)
    int idx = blockIdx.x*blockDim.x + threadIdx.x;

    // We do not need to recompute the exogenous uncertainties (everything
    // else other than animal populations)
    if (idx < noPaths) {
        // Shared memory declaration for patch populations used in path
        // recomputations and optimal controls/conditional expectations
        extern __shared__ float s[];

        float* payoffs, *currPayoffs;
        payoffs = (float*)malloc(noControls*sizeof(float));
        currPayoffs = (float*)malloc(noControls*sizeof(float));
        bool* valid;
        valid = (bool*)malloc(noControls*sizeof(float));

        float unitFuel = 0.0;
        float orePrice = 0.0;

        float conditionalExp = 0.0;
        int bestCont = 0;

        // Compute the unit fuel cost component
        for (int ii = 0; ii < noFuels; ii++) {
            unitFuel += fuelCosts[ii]*uResults[idx*nYears*noUncertainties +
                    start*noUncertainties + fuelIdx[ii]];
        }
        // Compute the unit revenue from ore
        for (int ii = 0; ii < noCommodities; ii++) {
            orePrice += uComposition[idx*nYears*noCommodities + start*
                    noCommodities + ii]*uResults[idx*nYears*noUncertainties +
                    start*noUncertainties + noFuels + ii];
        }

        if (start == nYears) {
            // At the last period we run the road if the adjusted population
            // for a particular control (pop-pop*aar_jj) is greater than the
            // minimum permissible population. This becomes the optimal
            // payoff, which we then regress onto the predictors to determine
            // the expected payoff at time T given a prevailing end adjusted
            // population. Everything is considered deterministic at this
            // stage. Therefore, we do not need to compute this section.
            for (int ii = 0; ii < noControls; ii++) {
                // Compute the single period financial payoff for each control
                // for this period and the adjusted profit. If any adjusted
                // population is below the threshold, then the payoff is
                // invalid.
                valid[ii] = true;
                for (int jj = 0; jj < noSpecies; jj++) {
                    float adjPop = totalPops[idx*noSpecies*(nYears+1) + start*
                            noSpecies + jj]*aars[idx*(nYears+1)*noSpecies*
                            noControls + start*noControls*noSpecies + jj*
                            noControls + ii];

                    // Zero flow control is always valid
                    if (adjPop < speciesParams[noSpecies*jj + 3] && ii > 0) {
                        valid[ii] = false;
                        break;
                    }
                }

                // Compute the payoff for the control if valid.
                if (valid[ii]) {
                    // Now compute the overall period profit for this control
                    // given the prevailing stochastic factors (undiscounted).
                    payoffs[ii] = Q[ii]*(unitCost + unitFuel - unitRevenue*
                            orePrice);
                } else {
                    payoffs[ii] = NAN;
                }
            }

            // Save the optimal control, conditional expectation and state to
            // the policy map variables

            // The optimal value is the one with the lowest net present cost.
            // As the zero flow rate option is always available, we can
            // initially set the optimal control to this before checking the
            // other controls.

            float bestExp = payoffs[0];
            int bestCont = 0;

            for (int ii = 1; ii < noControls; ii++) {
                if (isfinite(payoffs[ii])) {
                    if (payoffs[ii] < bestExp) {
                        bestExp = payoffs[ii];
                        bestCont = ii;
                    }
                }
            }

            conditionalExp = bestExp;
            optCont[nYears*noPaths + idx] = bestCont;

            // INITIAL STATES //
            // The states are the adjusted populations per unit traffic for
            // each species and the current period unit profit. We use the
            // aar of the selected control to compute this. AdjPops is only for
            // the current year. We only compute the values here for
            // completeness. They have no bearing on the results.
            for (int ii = 0; ii < noSpecies; ii++) {
                adjPops[ii*noPaths+idx] = totalPops[idx*noSpecies*(nYears+1) +
                        start*noSpecies + ii]*aars[idx*(nYears+1)*noSpecies*
                        noControls + start*noControls*noSpecies + ii*noControls
                        + controls[optCont[nYears*noPaths + idx]]];
            }

            // The prevailing unit profit
            unitProfits[start*noPaths + idx] = unitCost + unitFuel -
                    unitRevenue*orePrice;

//            float adjPop = totalPops[idx*noSpecies*(nYears+1) + start*
//                    noSpecies]*aars[idx*(nYears+1)*noSpecies*
//                    noControls + start*noControls*noSpecies + bestCont];

//            printf("%d %f %f %f %d\n",idx,condExp[nYears*noPaths + idx],unitProfits[start*noPaths + idx], adjPop,bestCont);

        } else {
            // For all other time periods, we need to recompute the forward
            // paths and add the present values of the expected payoffs to the
            // current period payoff using the regression functions that were
            // computed outside of this kernel.

            // As the original points were developed with linear regression,
            // we use linear interpolation as a reasonable approximation.
            // Furthermore, speed is an issue, so we need a faster approach
            // than a more accurate one such as cubic spline interpolation.

            // Find the current state through multilinear interpolation. The
            // state consists of the current period unit profit and the
            // adjusted population for each species under the chosen control.
            // As it is endogenous, the adjusted populations components of the
            // state are different for each control and must be dealt with
            // accordingly.
            float *state;
            state = (float*)malloc((noSpecies*noControls+1)*sizeof(float));

            // For each control
            for (int ii = 0; ii < noControls; ii++) {
                // 1. Adjusted populations
                for (int jj = 0; jj <noSpecies; jj++) {
                    state[ii*noSpecies + jj] = totalPops[idx*noSpecies*(nYears+
                            1) + start*noSpecies + jj]*aars[idx*(nYears+1)*
                            noControls*noSpecies + start*noControls*noSpecies +
                            jj*noControls + ii];
                }
            }

            // 2. Unit profit is the same for each control
            unitFuel = 0.0;
            orePrice = 0.0;

            // Compute the unit fuel cost component
            for (int ii = 0; ii < noFuels; ii++) {
                unitFuel += fuelCosts[ii]*uResults[idx*nYears*noUncertainties
                        + start*noUncertainties + fuelIdx[ii]];
            }
            // Compute the unit revenue from ore
            for (int ii = 0; ii < noCommodities; ii++) {
                orePrice += uComposition[idx*nYears*noCommodities + start*
                        noCommodities + ii]*uResults[idx*nYears*noUncertainties +
                        start*noUncertainties + noFuels + ii];
            }
            state[noSpecies*noControls] = unitCost + unitFuel - unitRevenue*
                    orePrice;
            unitProfits[start*noPaths + idx] = unitCost + unitFuel -
                    unitRevenue*orePrice;

//            if (idx == noPaths-1) {
//                printf("Test: %f %f %f %f\n",state[0],state[1],state[2],state[3]);
//            }

            // Determine the current period payoffs to select the optimal
            // control for this period.
            if (start > 0) {
                for (int ii = 0; ii < noControls; ii++) {
                    // Compute the single period financial payoff for each
                    // control for this period and the adjusted profit. If any
                    // adjusted population is below the threshold, then the
                    // payoff is invalid.
                    valid[ii] = true;
                    for (int jj = 0; jj < noSpecies; jj++) {
                        float adjPop = state[ii*noSpecies + jj];

                        if (adjPop < speciesParams[noSpecies*jj + 3] && ii >
                                0) {
                            valid[ii] = false;
                            break;
                        }
                    }

                    // Compute the payoff for the control if valid using the
                    // regressions. We keep track of the overall temporary cost
                    // -to-go in order to pick the optimal control as well as
                    // the current period payoffs in order to compute the
                    // adjusted cost-to-go that accounts for endogenous
                    // uncertainty.
                    if (valid[ii]) {
                        // Now compute the overall period profit for this
                        // control given the prevailing stochastic factors
                        // (undiscounted).
                        currPayoffs[ii] = Q[ii]*state[noSpecies*noControls];

                        // First find the global upper and lower bounds in each
                        // dimension as well as the index of the lower bound of
                        // the regressed value in each dimension.
                        float *lower, *upper, *coeffs;
                        int *lowerInd;
                        lower = (float*)malloc((noSpecies+1)*sizeof(float));
                        upper = (float*)malloc((noSpecies+1)*sizeof(float));
                        coeffs = (float*)malloc(((int)pow(2,noSpecies))*
                                sizeof(float));
                        lowerInd = (int*)malloc((noSpecies+1)*sizeof(float));

                        // Indices for species state variables
                        for (int jj = 0; jj < noSpecies; jj++) {
                            lower[jj] = regression[start*noControls*(dimRes*(
                                    noSpecies+1) + (int)pow(dimRes,noSpecies+1)
                                    *2) + ii*(dimRes*(noSpecies+1) + (int)pow(
                                    dimRes,(noSpecies+1))*2) + jj*dimRes];
                            upper[jj] = regression[start*noControls*(dimRes*(
                                    noSpecies+1) + (int)pow(dimRes,noSpecies+1)
                                    *2) + ii*(dimRes*(noSpecies+1) + (int)pow(
                                    dimRes,(noSpecies+1))*2) + (jj+1)*dimRes -
                                    1];

                            lowerInd[jj] = (int)(dimRes-1)*(state[ii*noSpecies
                                    + jj] - lower[jj])/(upper[jj] - lower[jj]);

                            if (lowerInd[jj] < 0) {
                                lowerInd[jj] = 0;
                            } else if (lowerInd[jj] >= dimRes) {
                                lowerInd[jj] = dimRes-2;
                            }
                        }

                        // Index for unit profit state variable
                        lower[noSpecies] = regression[start*noControls*(dimRes
                                *(noSpecies+1) + (int)pow(dimRes,noSpecies+1)*
                                2) + ii*(dimRes*(noSpecies+1) + (int)pow(
                                dimRes,(noSpecies+1))*2) + noSpecies*dimRes];
                        upper[noSpecies] = regression[start*noControls*(dimRes*
                                (noSpecies+1) + (int)pow(dimRes,noSpecies+1)*2)
                                + ii*(dimRes*(noSpecies+1) + (int)pow(dimRes,
                                (noSpecies+1))*2) + (noSpecies+1)*dimRes - 1];

                        lowerInd[noSpecies] = (int)(dimRes-1)*(state[noSpecies*
                                noControls] - lower[noSpecies])/(upper[
                                noSpecies] - lower[noSpecies]);

                        if (lowerInd[noSpecies] < 0) {
                            lowerInd[noSpecies] = 0;
                        } else if (lowerInd[noSpecies] >= dimRes) {
                            lowerInd[noSpecies] = dimRes-2;
                        }

    //                    if (ii == 0) {
    //                        printf("%d : %8f %8f | %8f %8f | %8f %8f | %4d %4d\n",idx,lower[0],upper[0],lower[1],upper[1],state[0],state[1],lowerInd[0],lowerInd[1]);
    //                    }

                        // Now that we have all the index requirements, let's
                        // interpolate.
                        // Get the uppermost dimension x value
                        float x0 = regression[start*noControls*(dimRes*(
                                noSpecies + 1) + (int)pow(dimRes,noSpecies+1)*2
                                ) + ii*(dimRes*(noSpecies+1) + (int)pow(dimRes,
                                (noSpecies+1))*2) + lowerInd[0]];
                        float x1 = regression[start*noControls*(dimRes*(
                                noSpecies + 1) + (int)pow(dimRes,noSpecies+1)*2
                                ) + ii*(dimRes*(noSpecies+1) + (int)pow(dimRes,
                                (noSpecies+1))*2) + lowerInd[0] + 1];

                        float xd;

                        if ((fabs(x1 - x0) < FLT_EPSILON) || x0 == x1) {
                            xd = 0.0;
                        } else {
                            xd = (state[ii*noSpecies] - x0)/(x1-x0);
                        }

                        // First, assign the yvalues to the coefficients matrix
                        for (int jj = 0; jj < (int)pow(2,noSpecies); jj++) {
                            // Get the indices of the yvalues of the lower and
                            // upper bounding values on this dimension.
                            int idxL = start*noControls*(dimRes*(noSpecies + 1)
                                    + (int)pow(dimRes,(noSpecies+1))*2) + ii*(
                                    dimRes*(noSpecies + 1) + (int)pow(dimRes,(
                                    noSpecies+1))*2) + dimRes*(noSpecies + 1);

                            // OLD
    //                        for (int kk = 0; kk < noSpecies; kk++) {
    //                            int rem = ((int)(jj/((int)pow(2,noSpecies - kk))) +
    //                                    1) - 2*(int)(((int)(jj/((int)pow(2,
    //                                    noSpecies - kk))) + 1)/2);
    //                            if (rem > 0) {
    //                                idxL += lowerInd[kk]*(int)pow(dimRes,noSpecies
    //                                        - kk);
    //                            } else {
    //                                idxL += (lowerInd[kk]+1)*(int)pow(dimRes,noSpecies
    //                                        - kk);
    //                            }
    //                        }

    //                        int idxU = idxL + (lowerInd[0]+1)*(int)pow(dimRes,
    //                                noSpecies);

    //                        idxL += lowerInd[0]*(int)pow(dimRes,noSpecies);

    //                        coeffs[jj] = regression[idxL]*(1 - xd) +
    //                                regression[idxU]*xd;

                            int div = jj;
                            for (int kk = 0; kk < noSpecies; kk++) {
                                int rem = div - 2*((int)(div/2));
                                div = (int)(div/2);

                                if (rem == 0) {
                                    idxL += lowerInd[noSpecies - 1 - kk]*(int)
                                            pow(dimRes,kk + 1);
                                } else {
                                    idxL += (lowerInd[noSpecies - 1 - kk] + 1)*
                                            (int)pow(dimRes,kk + 1);
                                }
                            }

                            coeffs[jj] = regression[idxL]*(1 - xd) +
                                    regression[idxL + 1]*xd;
                        }

                        // Now we work our way down the dimensions using our
                        // computed coefficients to get the interpolated value.
                        for (int jj = 1; jj <= noSpecies; jj++) {
                            // Get the current dimension x value
                            x0 = regression[start*noControls*(dimRes*(noSpecies
                                    +1) + (int)pow(dimRes,noSpecies+1)*2) + ii*
                                    (dimRes*(noSpecies+1) + (int)pow(dimRes,(
                                    noSpecies+1))*2) + jj*dimRes + lowerInd[
                                    jj]];
                            x1 = regression[start*noControls*(dimRes*(noSpecies
                                    + 1) + (int)pow(dimRes,noSpecies+1)*2) + ii
                                    *(dimRes*(noSpecies+1) + (int)pow(dimRes,(
                                    noSpecies+1))*2) + jj*dimRes + lowerInd[jj]
                                    + 1];

                            if ((fabs(x1 - x0) < FLT_EPSILON) || x0 == x1) {
                                xd = 0.0;
                            } else {
                                xd = (state[ii*noSpecies + jj] - x0)/(x1-x0);
                            }

                            for (int kk = 0; kk < (int)pow(2,jj); kk++) {
                                int jump = (int)pow(2,noSpecies-jj-1);
                                coeffs[kk] = coeffs[kk]*(1 - xd) + coeffs[kk+
                                        jump]*xd;
                            }
                        }

                        payoffs[ii] = currPayoffs[ii] + coeffs[0]/(1+rrr/(100*
                                timeStep));

                        free(lower);
                        free(upper);
                        free(coeffs);
                        free(lowerInd);
                    } else {
                        currPayoffs[ii] = NAN;
                        payoffs[ii] = NAN;
                    }
                }
                // Initialise the conditional expectations for this path at this
                // stage using the optimal control. Again, the first control of
                // no traffic flow will have a finite payoff as it is always a
                // valid option. We select the control with the lowest overall
                // payoff.

                float bestExp = payoffs[0];
                int bestCont = 0;

                for (int ii = 1; ii < noControls; ii++) {
                    if (isfinite(payoffs[ii])) {
                        if (payoffs[ii] < bestExp) {
                            bestExp = payoffs[ii];
                            bestCont = ii;
                        }
                    }
                }

//                condExp[start*noPaths + idx] = bestExp;
                optCont[start*noPaths + idx] = bestCont;

            } else {
                // We know the optimal control already from the
                // "firstPeriodInduction" kernel that was just called so we do
                // nothing here.
            }

//            if (idx == noPaths-1) {
//                printf("Test: %f %f %f\n",currPayoffs[0],currPayoffs[1],currPayoffs[2]);
//                printf("Test: %f %f %f\n",payoffs[0],payoffs[1],payoffs[2]);
//            }

            ///////////////////////////////////////////////////////////////////
            // Now recompute the optimal forward path and add the discounted
            // optimal payoff at each period to this path's conditional
            // expectation.
            // First, we must read in the shared memory for use in the
            // updated populations.
            for (int jj = 0; jj < noSpecies; jj++) {
                for (int kk = 0; kk < noPatches; kk++) {
                    s[2*threadIdx.x*noSpecies*noPatches + jj*noPatches + kk] =
                            pops[idx*(nYears+1)*noSpecies*noPatches + start*
                            noSpecies*noPatches + jj*noPatches + kk];
//                    s[2*idx*noSpecies*noPatches + jj*noPatches + kk] = pops[idx
//                            *(nYears+1)*noSpecies*noPatches + start*noSpecies*
//                            noPatches + jj*noPatches + kk];
                }
            }

            for (int ii = start+1; ii <= nYears; ii++) {
                bestCont = optCont[(ii-1)*noPaths + idx];

                // We must keep track of the population(s) over time as well as
                // the optimal choice taken. This means computing the current
                // state.

                // First, update the population given the optimal control at
                // the previous stage.
                //int control = optCont[(ii-1)*noPaths + idx];

                for (int jj = 0; jj < noSpecies; jj++) {
                    // For each patch, update the population for the next time
                    // period by using the movement and mortality matrix for
                    // the correct species/control combination. We use
                    // registers due to their considerably lower latency over
                    // global memory.
                    int iterator = 0;

                    // Movement and mortality
                    for (int ll = 0; ll < noPatches; ll++) {
                        // Population for this patch
                        float population = 0.0f;

                        // Transfer animals from each destination patch to
                        // this one for the next period.
                        for (int mm = 0; mm < elemsPerCol[(jj*noControls +
                                bestCont)*noPatches + ll]; mm++) {

                            population += s[2*threadIdx.x*noSpecies*noPatches +
                                    jj*noPatches + rowIdx[iterator + (jj*
                                    noControls + bestCont)*maxElems]]*mmm[
                                    iterator + (jj*noControls + bestCont)*
                                    maxElems];

//                            float value = pops[idx*(nYears+1)*noSpecies*
//                                    noPatches + (ii-1)*noSpecies*noPatches
//                                    + jj*noPatches + rowIdx[iterator + (jj*
//                                    noControls + control)*maxElems]]*mmm[
//                                    iterator + (jj*noControls + control)*
//                                    maxElems];

//                            population += value;

                            iterator++;
                        }

                        s[2*threadIdx.x*noSpecies*noPatches + jj*noPatches +
                                ll + noSpecies*noPatches] = population;
                    }

                    // Update the actual populations. We only care about the
                    // end result, therefore we do not alter the global
                    // population vectors; we update the shared memory
                    // population vector.
                    for (int ll = 0; ll < noPatches; ll++) {
                        // Population growth based on a mean-reverting process
                        float gr = rgr[idx*noSpecies*noPatches*nYears +
                                (ii-1)*noSpecies*noPatches + jj*
                                noPatches + ll];

                        // Use shared memory here
//                                pops[idx*(nYears+1)*noSpecies*noPatches + ii*
//                                        noSpecies*noPatches + jj*noPatches +
//                                        ll] = population*(1.0f + gr*(caps[jj*
//                                        noPatches + ll] - population)/caps[jj*
//                                        noPatches + ll]/100.0);
                        s[2*threadIdx.x*noPatches*noSpecies + jj*noPatches +
                                ll] = s[2*threadIdx.x*noSpecies*noPatches + jj*
                                noPatches + ll + noSpecies*noPatches]*(1.0f +
                                gr*(caps[jj*noPatches + ll] - s[2*threadIdx.x*
                                noSpecies*noPatches + noSpecies*noPatches + jj*
                                noPatches + ll])/caps[jj*noPatches + ll]/100.0);
                    }
                }

                __syncthreads();

                ///////////////////////////////////////////////////////////////
                // Now, as before, compute the current state and the optimal
                // control to pick using the regressions.
                ///////////////////////////////////////////////////////////////
                for (int jj = 0; jj < noSpecies; jj++) {

                    float initialPopulation = 0.0f;

                    for (int kk = 0; kk < noPatches; kk++) {
//                        initialPopulation += pops[idx*(nYears+1)*noSpecies*
//                                noPatches + ii*noSpecies*noPatches + jj*
//                                noPatches + kk];
                        initialPopulation += s[2*threadIdx.x*noPatches*
                                noSpecies + jj*noPatches + kk];
                    }

                    // Compute the aar under each control to determine the
                    // states
                    for (int kk = 0; kk < noControls; kk++) {
                        // Overall population at this time period
                        float totalPop = 0.0f;

                        int iterator = 0;
                        for (int ll = 0; ll < noPatches; ll++) {
                            // Population for this patch
                            float population = 0.0f;

                            // Transfer animals from each destination patch to
                            // this one for the next period.
                            for (int mm = 0; mm < elemsPerCol[(jj*noControls +
                                    kk)*noPatches + ll]; mm++) {

//                                float value = pops[idx*(nYears+1)*noSpecies*
//                                        noPatches + ii*noSpecies*noPatches +
//                                        jj*noPatches + rowIdx[iterator + (jj*
//                                        noControls + kk)*maxElems]]*mmm[
//                                        iterator + (jj*noControls + kk)*
//                                        maxElems];

//                                population += value;

                                population += s[2*threadIdx.x*noPatches*
                                        noSpecies + jj*noPatches + rowIdx[
                                        iterator + (jj*noControls + kk)*
                                        maxElems]]*mmm[iterator + (jj*
                                        noControls + kk)*maxElems];
                                iterator++;
                            }

                            totalPop += population;

                            state[jj*noControls + kk] = totalPop/
                                    initialPopulation;
                        }
                    }
                }

//                for (int jj = 0; jj <noSpecies; jj++) {
//                    state[jj] = totalPops[ii*noPaths + idx]*aars[ii*noPaths
//                            + idx];
//                }

                unitFuel = 0.0;
                orePrice = 0.0;

                // Compute the unit fuel cost component
                for (int jj = 0; jj < noFuels; jj++) {
                    unitFuel += fuelCosts[jj]*uResults[idx*nYears*
                            noUncertainties + ii*noUncertainties +
                            fuelIdx[jj]];
                }
                // Compute the unit revenue from ore
                for (int jj = 0; jj < noCommodities; jj++) {
                    orePrice += uComposition[idx*nYears*noCommodities +
                            ii*noCommodities + jj]*uResults[idx*nYears*
                            noUncertainties + ii*noUncertainties +
                            noFuels + jj];
                }

                // Current period unit profit (state used by all controls)
                state[noSpecies*noControls] = unitCost + unitFuel - unitRevenue
                        *orePrice;

                // Determine the current period payoffs to select the optimal
                // control for this period.
                for (int jj = 0; jj < noControls; jj++) {
                    // Compute the single period financial payoff for each control
                    // for this period and the adjusted profit. If any adjusted
                    // population is below the threshold, then the payoff is
                    // invalid.
                    valid[jj] = true;
                    for (int kk = 0; kk < noSpecies; kk++) {
//                        float adjPop = totalPops[ii*noSpecies*noPaths + idx*
//                                noSpecies + kk]*aars[kk*nYears*noPaths*noControls
//                                + idx*noControls + jj];

                        if (state[jj*noSpecies + kk] < speciesParams[noSpecies
                                *kk + 2]) {
                            valid[jj] = false;
                            break;
                        }
                    }

                    // Compute the payoff for the control if valid using the
                    // regressions. We keep track of the overall temporary cost
                    // -to- go in order to pick the optimal control as well as
                    // the current period payoffs in order to compute the
                    // adjusted cost-to-go that accounts for endogenous
                    // uncertainty.
                    // If we are at the last time period, we simply use the
                    // highest valid payoff (there are no regressions at this
                    // stage).
                    if (valid[jj]) {
                        // Now compute the overall period profit for this
                        // control given the prevailing stochastic factors
                        // (undiscounted).
                        currPayoffs[jj] = Q[jj]*state[noSpecies*noControls];

                        if (ii == nYears) {
                            // At the last time period, the payoff is simply
                            // the single period payoff (no conditional
                            // expectation to compute).
                            payoffs[jj] = currPayoffs[jj];

                        } else {
                            // First find global the upper and lower bounds in
                            // each dimension as well as the index of the lower
                            // bound of the regressed value in each dimension.
                            float *lower, *upper, *coeffs;
                            int *lowerInd;
                            lower = (float*)malloc((noSpecies+1)*sizeof(
                                    float));
                            upper = (float*)malloc((noSpecies+1)*sizeof(
                                    float));
                            coeffs = (float*)malloc(((int)pow(2,noSpecies))*
                                    sizeof(float));
                            lowerInd = (int*)malloc((noSpecies+1)*sizeof(
                                    float));

                            // Indices for species state variables
                            for (int kk = 0; kk < noSpecies; kk++) {
                                lower[kk] = regression[ii*noControls*(dimRes*(
                                        noSpecies+1) + (int)pow(dimRes,
                                        noSpecies+1)*2) + jj*(dimRes*(noSpecies
                                        +1) + (int)pow(dimRes,(noSpecies+1))*2)
                                        + kk*dimRes];
                                upper[kk] = regression[ii*noControls*(dimRes*(
                                        noSpecies+1) + (int)pow(dimRes,
                                        noSpecies+1)*2) + jj*(dimRes*(noSpecies
                                        +1) + (int)pow(dimRes,(noSpecies+1))*2)
                                        + (kk+1)*dimRes - 1];

                                lowerInd[kk] = (int)(dimRes-1)*(state[jj*
                                        noSpecies + kk] - lower[kk])/(upper[kk]
                                        - lower[kk]);
                            }

                            // Index for unit profit state variable
                            lower[noSpecies] = regression[ii*noControls*(dimRes
                                    *(noSpecies+1) + (int)pow(dimRes,noSpecies
                                    + 1)*2) + jj*(dimRes*(noSpecies+1) + (int)
                                    pow(dimRes,(noSpecies+1))*2) + noSpecies*
                                    dimRes];
                            upper[noSpecies] = regression[ii*noControls*(dimRes
                                    *(noSpecies+1) + (int)pow(dimRes,noSpecies
                                    + 1)*2) + jj*(dimRes*(noSpecies+1) + (int)
                                    pow(dimRes,(noSpecies+1))*2) + (noSpecies
                                    + 1)*dimRes - 1];

                            lowerInd[noSpecies] = (int)(dimRes-1)*(state[
                                    noSpecies*noControls] - lower[noSpecies])/(
                                    upper[noSpecies] - lower[noSpecies]);

                            if (lowerInd[noSpecies] < 0) {
                                lowerInd[noSpecies] = 0;
                            } else if (lowerInd[noSpecies] >= dimRes) {
                                lowerInd[noSpecies] = dimRes-2;
                            }

                            // Now that we have all the index requirements,
                            // let's interpolate.
                            // Get the uppermost dimension x value
                            float x0 = regression[ii*noControls*(dimRes*(
                                    noSpecies + 1) + (int)pow(dimRes,noSpecies
                                    + 1)*2) + jj*(dimRes*(noSpecies+1) + (int)
                                    pow(dimRes,(noSpecies+1))*2) + lowerInd[
                                    0]];
                            float x1 = regression[ii*noControls*(dimRes*(
                                    noSpecies + 1) + (int)pow(dimRes,noSpecies
                                    + 1)*2) + jj*(dimRes*(noSpecies+1) + (int)
                                    pow(dimRes,(noSpecies+1))*2) + lowerInd[0]
                                    + 1];
                            float xd;

                            if ((fabs(x1 - x0) < FLT_EPSILON) || x0 == x1) {
                                xd = 0.0;
                            } else {
                                xd = (state[jj*noSpecies] - x0)/(x1-x0);
                            }

                            // First, assign the yvalues to the coefficients
                            // matrix
                            for (int kk = 0; kk < (int)pow(2,noSpecies);
                                    kk++) {
                                // Get the indices of the yvales of the lower
                                // and upper bounding values on this dimension.
                                int idxL = ii*noControls*(dimRes*(noSpecies +
                                        1) + (int)pow(dimRes,(noSpecies+1))*2)
                                        + jj*(dimRes*(noSpecies + 1) + (int)
                                        pow(dimRes,(noSpecies+1))*2) + dimRes*
                                        (noSpecies + 1);

                                int div = kk;
                                for (int ll = 0; ll < noSpecies; ll++) {
                                    int rem = div - 2*((int)(div/2));
                                    div = (int)(div/2);

                                    if (rem == 0) {
                                        idxL += lowerInd[noSpecies - 1 - ll]*
                                                (int)pow(dimRes,ll + 1);
                                    } else {
                                        idxL += (lowerInd[noSpecies - 1 - ll]
                                                + 1)*(int)pow(dimRes,ll + 1);
                                    }
                                }

                                coeffs[kk] = regression[idxL]*(1 - xd) +
                                        regression[idxL + 1]*xd;

                                // OLD
    //                            for (int ll = 1; ll <= noSpecies; ll++) {
    //                                int rem = ((int)(kk/((int)pow(2,noSpecies -
    //                                        ll))) + 1) - 2*(int)(((int)(kk/((int)
    //                                        pow(2,noSpecies - ll))) + 1)/2);
    //                                if (rem > 0) {
    //                                    idxL += lowerInd[ll]*(int)pow(dimRes,
    //                                            noSpecies - ll);
    //                                } else {
    //                                    idxL += (lowerInd[ll] + 1)*(int)pow(dimRes,
    //                                            noSpecies - ll)*2;
    //                                }
    //                            }

    //                            int idxU = idxL + (lowerInd[0] + 1)*(int)pow(
    //                                    dimRes,noSpecies);

    //                            idxL += lowerInd[0]*(int)pow(dimRes,
    //                                    noSpecies);

    //                            coeffs[kk] = regression[idxL]*(1 - xd) +
    //                                    regression[idxU]*xd;
                            }

                            // Now we work our way down the dimensions using our
                            // computed coefficients to get the interpolated value.
                            for (int kk = 1; kk <= noSpecies; kk++) {
                                // Get the current dimension x value
                                x0 = regression[ii*noControls*(dimRes*(noSpecies +
                                        1) + (int)pow(dimRes,noSpecies+1)*2) + jj*(dimRes*(
                                        noSpecies+1) + (int)pow(dimRes,(noSpecies+1))*2) +
                                        kk*dimRes + lowerInd[kk]];
                                x1 = regression[ii*noControls*(dimRes*(noSpecies +
                                        1) + (int)pow(dimRes,noSpecies+1)*2) + jj*(dimRes*(
                                        noSpecies+1) + (int)pow(dimRes,(noSpecies+1))*2) +
                                        kk*dimRes + lowerInd[kk] + 1];

                                if ((fabs(x1 - x0) < FLT_EPSILON) || x0 == x1) {
                                    xd = 0.0;
                                } else {
                                    xd = (state[jj*noSpecies + kk] - x0)/(x1-x0);
                                }

                                for (int ll = 0; ll < (int)pow(2,kk); ll++) {
                                    int jump = (int)pow(2,noSpecies-kk-1);
                                    coeffs[ll] = coeffs[ll]*(1 - xd) + coeffs[ll+jump]
                                            *xd;
                                }
                            }

                            payoffs[jj] = currPayoffs[jj] + coeffs[0];

                            free(lower);
                            free(upper);
                            free(coeffs);
                            free(lowerInd);
                        }
                    } else {
                        currPayoffs[jj] = NAN;
                        payoffs[jj] = NAN;
                    }
                }

                // Initialise the conditional expectations for this path at this
                // stage using the optimal control. Again, the first control of
                // no traffic flow will have a finite payoff as it is always a
                // valid option. We select the control with the lowest overall
                // payoff.
                float currMax = currPayoffs[0];
                bestCont = 0;

                for (int jj = 1; jj < noControls; jj++) {
                    if (isfinite(payoffs[jj])) {
                        if (payoffs[jj] < currMax) {
                            currMax = payoffs[jj];
                            bestCont = jj;
                        }
                    }
                }

                // Now add the discounted cash flow for the current period for
                // the control with the optimal payoff to the retained values
                // for the optimal path value at this time step.
                conditionalExp += currMax/(1+rrr/(100*timeStep));
            }

            free(state);
        }
        // We don't need to keep the optimal control at this stage but can
        // easily store it later if we wish.

        condExp[start*noPaths + idx] = conditionalExp;

        // Free memory
        free(payoffs);
        free(valid);
        free(currPayoffs);

//        if (idx == noPaths-1) {
//            printf("\n Got here: %d\n",start);
//        }
    }
}

// Computes the optimal control to pick at time period 0. Not implemented in
// parallel.
__global__ void firstPeriodInduction(int noPaths, int nYears, int noSpecies,
        int noControls, float timeStep, float unitCost, float unitRevenue,
        float rrr, int noFuels, int noCommodities, float* Q, float* fuelCosts,
        float* totalPops, float* speciesParams, int* controls, float* aars,
        float* uComposition, float* uResults, int* fuelIdx, float* condExp,
        int* optCont, float* stats) {

    float *payoffs, *dataPoints;
    payoffs = (float*)malloc(noControls*sizeof(float));
    dataPoints = (float*)malloc(noControls*sizeof(float));
    bool* valid;
    valid = (bool*)malloc(noControls*sizeof(bool));

    float unitFuel = 0.0;
    float orePrice = 0.0;

    // Compute the unit fuel cost component
    for (int ii = 0; ii < noFuels; ii++) {
        unitFuel += fuelCosts[ii]*uResults[fuelIdx[ii]];
    }
    // Compute the unit revenue from ore
    for (int ii = 0; ii < noCommodities; ii++) {
        orePrice += uComposition[ii]*uResults[noFuels + ii];
    }

    for (int ii = 0; ii < noControls; ii++) {
        dataPoints[ii] = 0.0;
        payoffs[ii] = 0.0;
    }

    // Now get the average payoff across all paths of the same control for
    // each control

    for (int ii = 0; ii < noPaths; ii++) {
        int control = controls[ii*nYears];

        payoffs[control] += condExp[ii+noPaths];
        dataPoints[control]++;
    }

    for (int ii = 0; ii < noControls; ii++) {
        // Compute the single period financial payoff for each control
        // for this period and the adjusted profit. If any adjusted
        // population is below the threshold, then the payoff is
        // invalid.
        if (dataPoints[ii] > 0) {
            payoffs[ii] = payoffs[ii]/(dataPoints[ii]*(1+rrr/(100*
                    timeStep)));
        } else {
            break;
        }

        valid[ii] = true;
        for (int jj = 0; jj < noSpecies; jj++) {
            float adjPop = totalPops[jj]*aars[jj*noControls + ii];

            // Zero flow control is always valid
            if (adjPop < speciesParams[noSpecies*jj + 3] && ii > 0) {
                valid[ii] = false;
                break;
            }
        }

        // Compute the payoff for the control if valid.
        if (valid[ii]) {
            // Now compute the overall period profit for this control
            // given the prevailing stochastic factors (undiscounted).
            payoffs[ii] += Q[ii]*(unitCost + unitFuel - unitRevenue*
                    orePrice);
        } else {
            payoffs[ii] = NAN;
        }
    }

    // The optimal value is the one with the lowest net present cost.
    // As the zero flow rate option is always available, we can
    // initially set the optimal control to this before checking the
    // other controls.
    float bestExp = payoffs[0];
    int bestCont = 0;

    for (int ii = 1; ii < noControls; ii++) {
        if (isfinite(payoffs[ii])) {
            if (payoffs[ii] < bestExp) {
                bestExp = payoffs[ii];
                bestCont = ii;
            }
        }
    }

    // Assign the optimal control and payoff to all paths at time period 0

    // Standard deviation
    stats[2] = 0;

    // Assign values and prepare standard deviation
    for (int ii = 0; ii < noPaths; ii++) {
        condExp[ii] = bestExp;
        optCont[ii] = bestCont;

        if (controls[ii*nYears] == bestCont) {
            stats[2] += (condExp[ii+noPaths] - payoffs[bestCont])*(condExp[ii
                    +noPaths] - payoffs[bestCont]);
        }
    }

    stats[0] = condExp[0];
    stats[1] = (float)optCont[0];
    stats[2] = sqrt(stats[2]/(dataPoints[bestCont]*(1+rrr/(100*timeStep))));

    free(valid);
    free(payoffs);
    free(dataPoints);
}

// Performs backward induction for the optimal control problem
__global__ void backwardInduction(int start, int noPaths, int nYears, int
        noSpecies, int noControls, int noUncertainties, float timeStep, float
        unitCost, float unitRevenue, float rrr, int noFuels, int noCommodities,
        int dimRes, float* Q, float* fuelCosts, float* totalPops, float*
        speciesParams, int* controls, float* aars, float* regression, float*
        uComposition, float* uResults, int* fuelIdx, float* condExp, int*
        optCont, float* adjPops, float *unitProfits) {

    // Global thread index (path number)
    int idx = blockIdx.x*blockDim.x + threadIdx.x;

    // We do not need to recompute the exogenous uncertainties (everything
    // else other than animal populations)
    if (idx < noPaths) {
        float* payoffs, *currPayoffs;
        payoffs = (float*)malloc(noControls*sizeof(float));
        currPayoffs = (float*)malloc(noControls*sizeof(float));
        bool* valid;
        valid = (bool*)malloc(noControls*sizeof(float));

        float unitFuel = 0.0;
        float orePrice = 0.0;

        // Compute the unit fuel cost component
        for (int ii = 0; ii < noFuels; ii++) {
            unitFuel += fuelCosts[ii]*uResults[idx*nYears*noUncertainties +
                    start*noUncertainties + fuelIdx[ii]];
        }
        // Compute the unit revenue from ore
        for (int ii = 0; ii < noCommodities; ii++) {
            orePrice += uComposition[idx*nYears*noCommodities + start*
                    noCommodities + ii]*uResults[idx*nYears*noUncertainties +
                    start*noUncertainties + noFuels + ii];
        }

        float* grMean;
        grMean = (float*)malloc(noSpecies*sizeof(float));

        for (int ii = 0; ii < noSpecies; ii++) {
            grMean[ii] = speciesParams[ii*8];
        }

        if (start == nYears) {
            // At the last period we run the road if the adjusted population
            // for a particular control (pop-pop*aar_jj) is greater than the
            // minimum permissible population. This becomes the optimal
            // payoff, which we then regress onto the predictors to determine
            // the expected payoff at time T given a prevailing end adjusted
            // population. Everything is considered deterministic at this
            // stage. Therefore, we do not need to compute this section.
            for (int ii = 0; ii < noControls; ii++) {
                // Compute the single period financial payoff for each control
                // for this period and the adjusted profit. If any adjusted
                // population is below the threshold, then the payoff is
                // invalid.
                valid[ii] = true;
                for (int jj = 0; jj < noSpecies; jj++) {
                    float adjPop = totalPops[idx*noSpecies*(nYears+1) + start*
                            noSpecies + jj]*aars[idx*(nYears+1)*noSpecies*
                            noControls + start*noControls*noSpecies + jj*
                            noControls + ii];

                    // Zero flow control is always valid
                    if (adjPop < speciesParams[noSpecies*jj + 3] && ii > 0) {
                        valid[ii] = false;
                        break;
                    }
                }

                // Compute the payoff for the control if valid.
                if (valid[ii]) {
                    // Now compute the overall period profit for this control
                    // given the prevailing stochastic factors (undiscounted).
                    payoffs[ii] = Q[ii]*(unitCost + unitFuel - unitRevenue*
                            orePrice);
                } else {
                    payoffs[ii] = NAN;
                }
            }

            // Save the optimal control, conditional expectation and state to
            // the policy map variables

            // The optimal value is the one with the lowest net present cost.
            // As the zero flow rate option is always available, we can
            // initially set the optimal control to this before checking the
            // other controls.

            float bestExp = payoffs[0];
            int bestCont = 0;

            for (int ii = 1; ii < noControls; ii++) {
                if (isfinite(payoffs[ii])) {
                    if (payoffs[ii] < bestExp) {
                        bestExp = payoffs[ii];
                        bestCont = ii;
                    }
                }
            }

            condExp[nYears*noPaths + idx] = bestExp;
            optCont[nYears*noPaths + idx] = bestCont;

            // INITIAL STATES //
            // The states are the adjusted populations per unit traffic for
            // each species and the current period unit profit. We use the
            // aar of the selected control to compute this. AdjPops is only for
            // the current year. We only compute the values here for
            // completeness. They have no bearing on the results.
            for (int ii = 0; ii < noSpecies; ii++) {
                adjPops[ii*noPaths+idx] = totalPops[idx*noSpecies*(nYears+1) +
                        start*noSpecies + ii]*aars[idx*(nYears+1)*noSpecies*
                        noControls + start*noControls*noSpecies + ii*noControls
                        + controls[optCont[nYears*noPaths + idx]]];
            }

            // The prevailing unit profit
            unitProfits[start*noPaths + idx] = unitCost + unitFuel -
                    unitRevenue*orePrice;
        } else {
            // As the original points were developed with linear regression,
            // we use linear interpolation as a reasonable approximation.
            // Furthermore, speed is an issue, so we need a faster approach
            // than a more accurate one such as cubic spline interpolation.

            // Find the current state through multilinear interpolation. The
            // state consists of the current period unit profit and the
            // adjusted population for each species under the chosen control.
            // As it is endogenous, the adjusted populations components of the
            // state are different for each control and must be dealt with
            // accordingly.
            float *state;
            state = (float*)malloc((noSpecies*noControls+1)*sizeof(float));

            // For each control
            for (int ii = 0; ii < noControls; ii++) {
                // 1. Adjusted populations
                for (int jj = 0; jj <noSpecies; jj++) {
                    state[ii*noSpecies + jj] = totalPops[idx*noSpecies*(nYears+
                            1) + start*noSpecies + jj]*aars[idx*(nYears+1)*
                            noControls*noSpecies + start*noControls*noSpecies +
                            jj*noControls + ii];
                }
            }

            // 2. Unit profit is the same for each control
            unitFuel = 0.0;
            orePrice = 0.0;

            // Compute the unit fuel cost component
            for (int ii = 0; ii < noFuels; ii++) {
                unitFuel += fuelCosts[ii]*uResults[idx*nYears*noUncertainties
                        + start*noUncertainties + fuelIdx[ii]];
            }
            // Compute the unit revenue from ore
            for (int ii = 0; ii < noCommodities; ii++) {
                orePrice += uComposition[idx*nYears*noCommodities + start*
                        noCommodities + ii]*uResults[idx*nYears*noUncertainties +
                        start*noUncertainties + noFuels + ii];
            }
            state[noSpecies*noControls] = unitCost + unitFuel - unitRevenue*
                    orePrice;

            unitProfits[start*noPaths + idx] = unitCost + unitFuel -
                    unitRevenue*orePrice;

            // Determine the current period payoffs to select the optimal
            // control for this period.
            for (int ii = 0; ii < noControls; ii++) {
                // Compute the single period financial payoff for each control
                // for this period and the adjusted profit. If any adjusted
                // population is below the threshold, then the payoff is
                // invalid.
                valid[ii] = true;
                for (int jj = 0; jj < noSpecies; jj++) {
                    float adjPop = state[ii*noSpecies + jj];

                    if (adjPop < speciesParams[noSpecies*jj + 3] && ii > 0) {
                        valid[ii] = false;
                        break;
                    }
                }

                // Compute the payoff for the control if valid using the
                // regressions. We keep track of the overall temporary cost-to-
                // go in order to pick the optimal control as well as the
                // current period payoffs in order to compute the adjusted
                // cost-to-go that accounts for endogenous uncertainty.
                if (valid[ii]) {
                    // Now compute the overall period profit for this control
                    // given the prevailing stochastic factors (undiscounted).
                    currPayoffs[ii] = Q[ii]*state[noSpecies*noControls];

                    // First find the global upper and lower bounds in each
                    // dimension as well as the index of the lower bound of the
                    // regressed value in each dimension.
                    float *lower, *upper, *coeffs;
                    int *lowerInd;
                    lower = (float*)malloc((noSpecies+1)*sizeof(float));
                    upper = (float*)malloc((noSpecies+1)*sizeof(float));
                    coeffs = (float*)malloc(((int)pow(2,noSpecies))*
                            sizeof(float));
                    lowerInd = (int*)malloc((noSpecies+1)*sizeof(float));

                    // Indices for species state variables
                    for (int jj = 0; jj < noSpecies; jj++) {
                        lower[jj] = regression[start*noControls*(dimRes*(
                                noSpecies+1) + (int)pow(dimRes,noSpecies+1)*2)
                                + ii*(dimRes*(noSpecies+1) + (int)pow(dimRes,
                                (noSpecies+1))*2) + jj*dimRes];
                        upper[jj] = regression[start*noControls*(dimRes*(
                                noSpecies+1) + (int)pow(dimRes,noSpecies+1)*2)
                                + ii*(dimRes*(noSpecies+1) + (int)pow(dimRes,
                                (noSpecies+1))*2) + (jj+1)*dimRes - 1];

                        lowerInd[jj] = (int)(dimRes-1)*(state[ii*noSpecies +
                                jj] - lower[jj])/(upper[jj] - lower[jj]);

                        if (lowerInd[jj] < 0) {
                            lowerInd[jj] = 0;
                        } else if (lowerInd[jj] >= dimRes) {
                            lowerInd[jj] = dimRes-2;
                        }
                    }

                    // Index for unit profit state variable
                    lower[noSpecies] = regression[start*noControls*(dimRes*(
                            noSpecies+1) + (int)pow(dimRes,noSpecies+1)*2)
                            + ii*(dimRes*(noSpecies+1) + (int)pow(dimRes,
                            (noSpecies+1))*2) + noSpecies*dimRes];
                    upper[noSpecies] = regression[start*noControls*(dimRes*(
                            noSpecies+1) + (int)pow(dimRes,noSpecies+1)*2)
                            + ii*(dimRes*(noSpecies+1) + (int)pow(dimRes,
                            (noSpecies+1))*2) + (noSpecies+1)*dimRes - 1];

                    lowerInd[noSpecies] = (int)(dimRes-1)*(state[noSpecies*
                            noControls] - lower[noSpecies])/(upper[noSpecies]
                            - lower[noSpecies]);

                    if (lowerInd[noSpecies] < 0) {
                        lowerInd[noSpecies] = 0;
                    } else if (lowerInd[noSpecies] >= dimRes) {
                        lowerInd[noSpecies] = dimRes-2;
                    }

                    // Now that we have all the index requirements, let's
                    // interpolate.
                    // Get the uppermost dimension x value
                    float x0 = regression[start*noControls*(dimRes*(noSpecies +
                            1) + (int)pow(dimRes,noSpecies+1)*2) + ii*(dimRes*(
                            noSpecies+1) + (int)pow(dimRes,(noSpecies+1))*2) +
                            lowerInd[0]];
                    float x1 = regression[start*noControls*(dimRes*(noSpecies +
                            1) + (int)pow(dimRes,noSpecies+1)*2) + ii*(dimRes*(
                            noSpecies+1) + (int)pow(dimRes,(noSpecies+1))*2) +
                            lowerInd[0] + 1];

                    float xd;

                    if ((fabs(x1 - x0) < FLT_EPSILON) || x0 == x1) {
                        xd = 0.0;
                    } else {
                        xd = (state[ii*noSpecies] - x0)/(x1-x0);
                    }

                    // First, assign the yvalues to the coefficients matrix
                    for (int jj = 0; jj < (int)pow(2,noSpecies); jj++) {
                        // Get the indices of the yvalues of the lower and upper
                        // bounding values on this dimension.
                        int idxL = start*noControls*(dimRes*(noSpecies + 1) +
                                (int)pow(dimRes,(noSpecies+1))*2) + ii*(dimRes*
                                (noSpecies + 1) + (int)pow(dimRes,(noSpecies+1))
                                *2) + dimRes*(noSpecies + 1);

                        int div = jj;
                        for (int kk = 0; kk < noSpecies; kk++) {
                            int rem = div - 2*((int)(div/2));
                            div = (int)(div/2);

                            if (rem == 0) {
                                idxL += lowerInd[noSpecies - 1 - kk]*(int)pow(
                                        dimRes,kk + 1);
                            } else {
                                idxL += (lowerInd[noSpecies - 1 - kk] + 1)*
                                        (int)pow(dimRes,kk + 1);
                            }
                        }

                        coeffs[jj] = regression[idxL]*(1 - xd) +
                                regression[idxL + 1]*xd;
                    }

                    // Now we work our way down the dimensions using our
                    // computed coefficients to get the interpolated value.
                    for (int jj = 1; jj <= noSpecies; jj++) {
                        // Get the current dimension x value
                        x0 = regression[start*noControls*(dimRes*(noSpecies +
                                1) + (int)pow(dimRes,noSpecies+1)*2) + ii*(dimRes*(
                                noSpecies+1) + (int)pow(dimRes,(noSpecies+1))*2) +
                                jj*dimRes + lowerInd[jj]];
                        x1 = regression[start*noControls*(dimRes*(noSpecies +
                                1) + (int)pow(dimRes,noSpecies+1)*2) + ii*(dimRes*(
                                noSpecies+1) + (int)pow(dimRes,(noSpecies+1))*2) +
                                jj*dimRes + lowerInd[jj] + 1];

                        if ((fabs(x1 - x0) < FLT_EPSILON) || x0 == x1) {
                            xd = 0.0;
                        } else {
                            xd = (state[ii*noSpecies + jj] - x0)/(x1-x0);
                        }

                        for (int kk = 0; kk < (int)pow(2,jj); kk++) {
                            int jump = (int)pow(2,noSpecies-jj-1);
                            coeffs[kk] = coeffs[kk]*(1 - xd) + coeffs[kk+jump]
                                    *xd;
                        }
                    }

                    payoffs[ii] = currPayoffs[ii] + coeffs[0]/(1+rrr/(100*
                            timeStep));

                    free(lower);
                    free(upper);
                    free(coeffs);
                    free(lowerInd);
                } else {
                    currPayoffs[ii] = NAN;
                    payoffs[ii] = NAN;
                }
            }

            // Initialise the conditional expectations for this path at this
            // stage using the optimal control. Again, the first control of
            // no traffic flow will have a finite payoff as it is always a
            // valid option. We select the control with the lowest overall
            // payoff.

            float bestExp = currPayoffs[0];
            int bestCont = 0;

            for (int ii = 1; ii < noControls; ii++) {
                if (isfinite(payoffs[ii])) {
                    if (payoffs[ii] < bestExp) {
                        bestExp = currPayoffs[ii];
                        bestCont = ii;
                    }
                }
            }

            condExp[start*noPaths + idx] = payoffs[bestCont];
            optCont[start*noPaths + idx] = bestCont;

            free(state);
        }

        // Free memory
        free(payoffs);
        free(valid);
        free(currPayoffs);
        free(grMean);
    }
}

// If using cuBlas, we need the following two routines.
// Create cuBlas handles for each thread
__global__ void createHandles(cublasHandle_t* handles, int noThreads) {

    // Global thread index
    int idx = blockIdx.x*blockDim.x + threadIdx.x;

    if (idx < noThreads) {
        cublasStatus_t status = cublasCreate_v2(&handles[idx]);
    }
}

// Destroy cuBlas handles for each thread
__global__ void destroyHandles(cublasHandle_t* handles, int noThreads) {

    // Global thread index
    int idx = blockIdx.x*blockDim.x + threadIdx.x;

    if (idx < noThreads) {
        cublasStatus_t status = cublasDestroy_v2(handles[idx]);
    }
}

// Multiple local linear regression. Makes use of LU decomposition for
// solving the linear equations. Does not use cuBlas.
__global__ void multiLocLinReg(int noPoints, int noDims, int dimRes, int nYears,
        int noControls, int year, int control, int k, int* dataPoints, float
        *xvals, float *yvals, float *regression, float* xmins, float* xmaxes,
        float *dist, int *ind) {

    // Global thread index
    int idx = blockIdx.x*blockDim.x + threadIdx.x;

    if (idx < noPoints) {
        // First, deconstruct the index into the index along each dimension
        int *dimIdx;
        dimIdx = (int*)malloc(noDims*sizeof(int));

        int rem = idx;

        for (int ii = 0; ii < noDims; ii++) {
            int div = (int)(rem/pow(dimRes,noDims-ii-1));
            dimIdx[ii] = div;
            rem = rem - div*pow(dimRes,noDims-ii-1);
        }

        // Get the query point coordinates
        float *xQ;
        xQ = (float*)malloc(noDims*sizeof(float));

        for (int ii = 0; ii < noDims; ii++) {
            xQ[ii] = ((float)dimIdx[ii])*(xmaxes[control*noDims + ii] -
                    xmins[control*noDims + ii])/(float)dimRes +
                    xmins[control*noDims + ii];
        }

        // 1. First find the k nearest neighbours to the query point (already)
        // computed prior).

        // 2. Build the matrices used in the calculation
        // A - Input design matrix
        // B - Input known matrix
        // C - Output matrix of coefficients
        float *A, *B, *X;

        A = (float*)malloc(pow(noDims+1,2)*sizeof(float));
        B = (float*)malloc((noDims+1)*sizeof(float));
        X = (float*)malloc((noDims+1)*sizeof(float));

        // Bandwidth for kernel
        float h = dist[noPoints*(k-1) + idx];

        for (int ii = 0; ii <= noDims; ii++) {
            // We will use a kernel and normalise by the distance of
            // the furthest point of the nearest k neighbours.

            // Initialise values to zero
            B[ii] = 0.0;

            for (int kk = 0; kk < k; kk++) {
                float d = dist[noPoints*kk + idx];
                // Gaussian kernel (Not used for now)
//                float z = exp(-(d/h)*(d/h)/2)/sqrt(2*M_PI);
                // Epanechnikov kernel
                float z = 0.75*(1-pow(d/h,2));

                if (ii == 0) {
                    B[ii] += yvals[ind[noPoints*kk + idx] - 1]*z;
                } else {
                    B[ii] += yvals[ind[noPoints*kk + idx] - 1]*(xvals[(ind[noPoints
                            *kk + idx] - 1)*noDims + ii - 1] - xQ[ii-1])*z;
                }
            }

            for (int jj = 0; jj <= noDims; jj++) {
                A[jj*(noDims+1)+ii] = 0.0;

                for (int kk = 0; kk < k; kk++) {
//                    float h = d_h[ind[kk]];
                    float d = dist[noPoints*kk + idx];
//                    For Gaussian kernel. Not used.
//                    float z = exp(-(d/h)*(d/h)/2)/sqrt(2*M_PI);
                    float z = 0.75*(1-pow(d/h,2));

                    if ((ii == 0) && (jj == 0)) {
                        A[jj*(noDims+1)+ii] += 1.0*z;
                    } else if (ii == 0) {
                        A[jj*(noDims+1)+ii] += (xvals[(ind[noPoints*kk + idx] - 1
                                )*noDims + jj - 1] - xQ[jj - 1])*z;
                    } else if (jj == 0) {
                        A[jj*(noDims+1)+ii] += (xvals[(ind[noPoints*kk + idx] - 1
                                )*noDims + ii - 1] - xQ[ii - 1])*z;
                    } else {
                        A[jj*(noDims+1)+ii] += (xvals[(ind[noPoints*kk + idx] - 1
                                )*noDims + jj - 1] - xQ[jj-1])*(xvals[(ind[
                                noPoints*kk + idx] - 1)*noDims + ii - 1] - xQ[ii
                                - 1])*z;
                    }
                }
            }
        }

        // Solve the linear system using LU decomposition.
        solveLinearSystem(noDims+1,A,B,X);

        // 4. Compute the y value at the x point of interest using the just-
        //    found regression coefficients. This is simply the y intercept we
        //    just computed and save to the regression matrix.
        regression[year*noControls*(dimRes*noDims + (int)pow(dimRes,noDims)*2)
                + control*(dimRes*noDims + (int)pow(dimRes,noDims)*2) + dimRes*
                noDims + idx] = /*yvals[ind[idx] - 1]*/ X[0];

        // Free memory
        free(A);
        free(B);
        free(X);
        free(xQ);
        free(dimIdx);
    }
}

__global__ void rovCorrection(int noPoints, int noDims, int dimRes, int nYears,
        int noControls, int year, int control, float* regression) {

    // Global thread index
    int idx = blockIdx.x*blockDim.x + threadIdx.x;

    if (idx < noPoints) {
        float currVal = regression[year*noControls*(dimRes*noDims +
                (int)pow(dimRes,noDims)*2) + control*(dimRes*noDims +
                (int)pow(dimRes,noDims)*2) + dimRes*noDims + idx];

        // The surrogate value cannot be greater than zero by definition
        if (currVal > 0) {
            regression[year*noControls*(dimRes*noDims + (int)pow(dimRes,
                    noDims)*2) + control*(dimRes*noDims + (int)pow(dimRes,
                    noDims)*2) + dimRes*noDims + idx] = 0.0;
        }
    }
}

// Interpolation routine for multiple regression. Uses linear interpolation
// for now for speed.
__global__ void interpolateMulti(int points, int noDims, int dimRes, float*
        surrogate, float* predictors, float* results) {

    // Global thread index
    int idx = blockIdx.x*blockDim.x + threadIdx.x;

    if (idx < points) {
        float *lower, *upper, *coeffs;
        int *lowerInd;
        lower = (float*)malloc((noDims)*sizeof(float));
        upper = (float*)malloc((noDims)*sizeof(float));
        coeffs = (float*)malloc(((int)pow(2,noDims-1))*sizeof(float));
        lowerInd = (int*)malloc((noDims)*sizeof(float));

        for (int jj = 0; jj < noDims; jj++) {
            lower[jj] = surrogate[jj*dimRes];
            upper[jj] = surrogate[(jj+1)*dimRes - 1];
            lowerInd[jj] = (int)((dimRes-1)*(predictors[noDims*idx+jj] -
                    lower[jj])/(upper[jj] - lower[jj]));

            if (lowerInd[jj] >= (dimRes-1)) {
                lowerInd[jj] = dimRes-2;
            } else if (lowerInd[jj] < 0){
                lowerInd[jj] = 0;
            }
        }

        // Let's interpolate
        // Uppermost dimensions x value
        float x0 = surrogate[lowerInd[0]];
        float x1 = surrogate[lowerInd[0]+1];
        float xd = (predictors[noDims*idx] - x0)/(x1-x0);

        // First, assign the yvalues to the coefficients matrix
        for (int jj = 0; jj < (int)pow(2,noDims-1); jj++) {
            // Get the indices of the yvalues of the lower and upper bounding
            // values on this dimension.
            int idxL = dimRes*noDims;

            for (int kk = 1; kk < noDims; kk++) {
                int rem = ((int)(jj/((int)pow(2,noDims - kk - 1))) + 1) - 2*
                        (int)(((int)(jj/((int)pow(2,noDims - kk - 1))) + 1)/2);
                if(rem > 0) {
                    idxL += lowerInd[kk]*(int)pow(dimRes,noDims - kk - 1);
                } else {
                    idxL += (lowerInd[kk]+1)*(int)pow(dimRes,noDims - kk - 1);
                }
            }

            int idxU = idxL + (lowerInd[0]+1)*(int)pow(dimRes,noDims-1);

            idxL += lowerInd[0]*(int)pow(dimRes,noDims-1);

            coeffs[jj] = surrogate[idxL]*(1 - xd) + surrogate[idxU]*xd;
        }

        // Now we work our way down the dimensions using our computed
        // coefficients to get the interpolated value.
        for (int jj = 1; jj < noDims; jj++) {
            // Get the current dimension x value
            x0 = surrogate[jj*dimRes + lowerInd[jj]];
            x1 = surrogate[jj*dimRes + lowerInd[jj] + 1];
            xd = (predictors[jj] - x0)/(x1-x0);

            for (int kk = 0; kk < (int)pow(2,jj); kk++) {
                int jump = (int)pow(2,noDims - jj - 2);
                coeffs[kk] = coeffs[kk]*(1 - xd) + coeffs[kk + jump]*xd;
            }
        }

        // Free variables
        free(lowerInd);
        free(coeffs);
        free(upper);
        free(lower);
        // Output the result
        results[idx] = coeffs[0];
    }
}

// WRAPPERS ///////////////////////////////////////////////////////////////////
// This section documents the wrappers for the CUDA kernels that can be called
// by external C++ routines (i.e. the external routines do not need to be
// compiled with nvcc).

// Computes the expected present value of an uncertain price.
void SimulateGPU::expPV(UncertaintyPtr uncertainty) {
    // Get device properties
    int device = 0;
    struct cudaDeviceProp properties;
    CUDA_CALL(cudaGetDeviceProperties(&properties, device));
    int maxThreadsPerBlock = properties.maxThreadsPerBlock;

    OptimiserPtr optimiser = uncertainty->getOptimiser();
    EconomicPtr economic = optimiser->getEconomic();
    unsigned int nYears = economic->getYears();
    double timeStep = economic->getTimeStep();
    unsigned int noPaths = optimiser->getOtherInputs()->getNoPaths();
    double total = 0.0;
    double gr = optimiser->getTraffic()->getGR()*economic->getTimeStep();

    // Get experimental scenario multipliers
    ExperimentalScenarioPtr sc = optimiser->getScenario();
    VariableParametersPtr vp = optimiser->getVariableParams();

    // Uncertain components of Brownian motion
    float *d_brownian, *d_jumpSizes, *d_jumps, *d_results, *results;
    curandGenerator_t gen;
    srand(time(NULL));
    int _seed = rand();
    results = (float*)malloc(noPaths*sizeof(float));

    try {
        // Allocate GPU memory
        CUDA_CALL(cudaMalloc((void **)&d_brownian, sizeof(float)*nYears*
                noPaths));
        CUDA_CALL(cudaMalloc((void **)&d_jumpSizes, sizeof(float)*nYears*
                noPaths));
        CUDA_CALL(cudaMalloc((void **)&d_jumps, sizeof(float)*nYears*noPaths));
        CUDA_CALL(cudaMalloc((void **)&d_results, sizeof(float)*nYears*
                noPaths));

        CURAND_CALL(curandCreateGenerator(&gen, CURAND_RNG_PSEUDO_DEFAULT));
        CURAND_CALL(curandSetPseudoRandomGeneratorSeed(gen, _seed));

        CURAND_CALL(curandGenerateNormal(gen, d_brownian, nYears*noPaths, 0.0f,
                uncertainty->getNoiseSD()*timeStep*vp->
                getCommoditySDMultipliers()(sc->getCommoditySD())));
        CURAND_CALL(curandGenerateNormal(gen, d_jumpSizes, nYears*noPaths,
                -pow(uncertainty->getPoissonJump()*vp->
                getCommoditySDMultipliers()(sc->getCommoditySD()),2)/2,pow(
                uncertainty->getPoissonJump()*vp->getCommoditySDMultipliers()(
                sc->getCommoditySD()),2)));
        CURAND_CALL(curandGenerateUniform(gen, d_jumps, nYears*noPaths));

        CURAND_CALL(curandDestroyGenerator(gen));

        // Compute path values
        int noBlocks = (noPaths % maxThreadsPerBlock) ? (int)(
                noPaths/maxThreadsPerBlock + 1) : (int)
                (noPaths/maxThreadsPerBlock);
        int noThreadsPerBlock = min(maxThreadsPerBlock,nYears*noPaths);

        // Call CUDA kernel
        expPVPath<<<noBlocks,noThreadsPerBlock>>>(noPaths, gr, nYears,
                uncertainty->getMean()*vp->getCommodityMultipliers()(
                sc->getCommodity()), timeStep, economic->getRRR(),
                uncertainty->getCurrent(), uncertainty->getMRStrength()*vp->
                getCommoditySDMultipliers()(sc->getCommoditySD()),
                uncertainty->getJumpProb()*vp->getCommoditySDMultipliers()(
                sc->getCommoditySD()), d_brownian, d_jumpSizes, d_jumps,
                d_results);
        CUDA_CALL(cudaPeekAtLastError());
        CUDA_CALL(cudaDeviceSynchronize());

        CUDA_CALL(cudaMemcpy(results,d_results,noPaths*sizeof(float),
                cudaMemcpyDeviceToHost));

        for (int ii = 0; ii < noPaths; ii++) {
            total += results[ii];
        }

        uncertainty->setExpPV((double)total/((double)noPaths));

        total = 0.0;
        for (int ii = 0; ii < noPaths; ii++) {
            total += pow(results[ii] - uncertainty->getExpPV(),2);
        }

        uncertainty->setExpPVSD(sqrt(total));

        CUDA_CALL(cudaFree(d_brownian));
        CUDA_CALL(cudaFree(d_jumpSizes));
        CUDA_CALL(cudaFree(d_jumps));
        CUDA_CALL(cudaFree(d_results));

    } catch (const char* err) {
        throw err;
    }
    free(results);
}

// Matrix multiplication (Naive)
void SimulateGPU::eMMN(const Eigen::MatrixXd& A, const Eigen::MatrixXd& B,
        Eigen::MatrixXd& C) {

    int device = 0;
    struct cudaDeviceProp properties;
    cudaGetDeviceProperties(&properties, device);
    int maxThreadsPerBlock = properties.maxThreadsPerBlock;

    if (A.cols() != B.rows()) {
        throw "SimulateGPU: matrixMultiplication: Inner dimensions do not match!";
    }

    float *Af, *Bf, *Cf, *d_A, *d_B, *d_C;

    int a = A.rows();
    int b = A.cols();
    int c = B.rows();
    int d = B.cols();

    Af = (float*)malloc(a*b*sizeof(float));
    Bf = (float*)malloc(c*d*sizeof(float));
    Cf = (float*)malloc(a*d*sizeof(float));

    CUDA_CALL(cudaMalloc(&d_A,a*b*sizeof(float)));
    CUDA_CALL(cudaMemcpy(d_A,Af,a*b*sizeof(float),cudaMemcpyHostToDevice));

    CUDA_CALL(cudaMalloc(&d_B,c*d*sizeof(float)));
    CUDA_CALL(cudaMemcpy(d_B,Bf,c*d*sizeof(float),cudaMemcpyHostToDevice));

    CUDA_CALL(cudaMalloc(&d_C,a*d*sizeof(float)));

    // Declare the number of blocks per grid and the number of threads per block
    dim3 threadsPerBlock(a, d);
    dim3 blocksPerGrid(1, 1);
    if (a*d > maxThreadsPerBlock){
        threadsPerBlock.x = maxThreadsPerBlock;
        threadsPerBlock.y = maxThreadsPerBlock;
        blocksPerGrid.x = ceil(double(a)/double(threadsPerBlock.x));
        blocksPerGrid.y = ceil(double(d)/double(threadsPerBlock.y));
    }

    // Call the naive kernel
    matrixMultiplicationKernelNaive<<<blocksPerGrid,threadsPerBlock>>>(d_A,d_B,
            d_C,a,b,c,d);
    CUDA_CALL(cudaPeekAtLastError());
    CUDA_CALL(cudaDeviceSynchronize());

    // Retrieve result and free data
    CUDA_CALL(cudaMemcpy(C.data(),d_C,a*d*sizeof(float),
            cudaMemcpyDeviceToHost));

    free(Af);
    free(Bf);
    free(Cf);
    CUDA_CALL(cudaFree(d_A));
    CUDA_CALL(cudaFree(d_B));
    CUDA_CALL(cudaFree(d_C));
}

// Performs matrix multiplication using shared memory
void SimulateGPU::eMM(const Eigen::MatrixXd& A, const Eigen::MatrixXd& B,
        Eigen::MatrixXd& C) {

    int device = 0;
    struct cudaDeviceProp properties;
    cudaGetDeviceProperties(&properties, device);

    if (A.cols() != B.rows()) {
        throw "SimulateGPU: matrixMultiplication: Inner dimensions do not match!";
    }

    float *d_A, *d_B, *d_C;

    int a = A.rows();
    int b = A.cols();
    int c = B.rows();
    int d = B.cols();

    Eigen::MatrixXf Af = A.cast<float>();
    Eigen::MatrixXf Bf = B.cast<float>();
    Eigen::MatrixXf Cf = C.cast<float>();

    CUDA_CALL(cudaMalloc(&d_A,a*b*sizeof(float)));
    CUDA_CALL(cudaMemcpy(d_A,Af.data(),a*b*sizeof(float),
            cudaMemcpyHostToDevice));

    CUDA_CALL(cudaMalloc(&d_B,c*d*sizeof(float)));
    CUDA_CALL(cudaMemcpy(d_B,Bf.data(),c*d*sizeof(float),
            cudaMemcpyHostToDevice));

    CUDA_CALL(cudaMalloc(&d_C,a*d*sizeof(float)));

    // declare the number of blocks per grid and the number of threads per block
    dim3 threads(BLOCK_SIZE,VECTOR_SIZE);
    dim3 grid(d/(BLOCK_SIZE*VECTOR_SIZE), a/BLOCK_SIZE);

    // Call the CUDA kernel
    matrixMultiplicationKernel<<<grid,threads>>>(d_A,d_B,d_C,a,b,d);
    CUDA_CALL(cudaPeekAtLastError());
    CUDA_CALL(cudaDeviceSynchronize());

    // Retrieve result and free data
    CUDA_CALL(cudaMemcpy(Cf.data(),d_C,a*d*sizeof(float),
            cudaMemcpyDeviceToHost));

    C = Cf.cast<double>();
    CUDA_CALL(cudaFree(d_A));
    CUDA_CALL(cudaFree(d_B));
    CUDA_CALL(cudaFree(d_C));

}

// Performs element-wise matrix multiplication. Does not require shared
// memory.
void SimulateGPU::ewMM(const Eigen::MatrixXd& A, const Eigen::MatrixXd &B,
        Eigen::MatrixXd &C) {

    int device = 0;
    struct cudaDeviceProp properties;
    cudaGetDeviceProperties(&properties, device);

    if ((A.cols() != B.cols()) || (A.rows() != B.rows())) {
        throw "SimulateGPU: matrixMultiplication: Matrix dimensions do not match!";
    }

    float *d_A, *d_B, *d_C;

    int a = A.rows();
    int b = A.cols();

    Eigen::MatrixXf Af = A.cast<float>();
    Eigen::MatrixXf Bf = B.cast<float>();
    Eigen::MatrixXf Cf = C.cast<float>();

    CUDA_CALL(cudaMalloc(&d_A,a*b*sizeof(float)));
    CUDA_CALL(cudaMemcpy(d_A,Af.data(),a*b*sizeof(float),
            cudaMemcpyHostToDevice));

    CUDA_CALL(cudaMalloc(&d_B,a*b*sizeof(float)));
    CUDA_CALL(cudaMemcpy(d_B,Bf.data(),a*b*sizeof(float),
            cudaMemcpyHostToDevice));

    CUDA_CALL(cudaMalloc(&d_C,a*b*sizeof(float)));

    // declare the number of blocks per grid and the number of threads per
    // block
    dim3 dimBlock(32,32);
    dim3 dimGrid(b/dimBlock.x,a/dimBlock.y);

    // Call the CUDA kernel
    matrixMultiplicationKernelEW<<<dimGrid,dimBlock>>>(d_A,d_B,d_C,a,b);
    CUDA_CALL(cudaPeekAtLastError());
    CUDA_CALL(cudaDeviceSynchronize());

    // Retrieve result and free data
    CUDA_CALL(cudaMemcpy(Cf.data(),d_C,a*b*sizeof(float),
            cudaMemcpyDeviceToHost));

    C = Cf.cast<double>();
    CUDA_CALL(cudaFree(d_A));
    CUDA_CALL(cudaFree(d_B));
    CUDA_CALL(cudaFree(d_C));
}

// Performs element-wise matrix division
void SimulateGPU::ewMD(const Eigen::MatrixXd& A, const Eigen::MatrixXd& B,
        Eigen::MatrixXd& C) {

    int device = 0;
    struct cudaDeviceProp properties;
    cudaGetDeviceProperties(&properties, device);

    if ((A.cols() != B.cols()) || (A.rows() != B.rows())) {
        throw "SimulateGPU: matrixMultiplication: Matrix dimensions do not match!";
    }

    float *d_A, *d_B, *d_C;

    int a = A.rows();
    int b = A.cols();

    Eigen::MatrixXf Af = A.cast<float>();
    Eigen::MatrixXf Bf = B.cast<float>();
    Eigen::MatrixXf Cf = C.cast<float>();

    CUDA_CALL(cudaMalloc(&d_A,a*b*sizeof(float)));
    CUDA_CALL(cudaMemcpy(d_A,Af.data(),a*b*sizeof(float),
            cudaMemcpyHostToDevice));

    CUDA_CALL(cudaMalloc(&d_B,a*b*sizeof(float)));
    CUDA_CALL(cudaMemcpy(d_B,Bf.data(),a*b*sizeof(float),
            cudaMemcpyHostToDevice));

    CUDA_CALL(cudaMalloc(&d_C,a*b*sizeof(float)));

    // declare the number of blocks per grid and the number of threads per
    // block
    dim3 dimBlock(32,32);
    dim3 dimGrid(b/dimBlock.x,a/dimBlock.y);

    matrixDivisionKernelEW<<<dimGrid,dimBlock>>>(d_A,d_B,d_C,a,b);
    CUDA_CALL(cudaPeekAtLastError());
    CUDA_CALL(cudaDeviceSynchronize());

    // Retrieve result and free data
    CUDA_CALL(cudaMemcpy(Cf.data(),d_C,a*b*sizeof(float),
            cudaMemcpyDeviceToHost));

    C = Cf.cast<double>();
    CUDA_CALL(cudaFree(d_A));
    CUDA_CALL(cudaFree(d_B));
    CUDA_CALL(cudaFree(d_C));
}

// Computes the number of times each line in XY2 crosses the parametrised line
// XY1.
void SimulateGPU::lineSegmentIntersect(const Eigen::MatrixXd& XY1, const
        Eigen::MatrixXd& XY2, Eigen::VectorXi& crossings) {

    try {
        int device = 0;
        struct cudaDeviceProp properties;
        cudaGetDeviceProperties(&properties, device);
        int maxThreadsPerBlock = properties.maxThreadsPerBlock;
        int maxBlocksPerGrid = 65536;

        // Precompute necessary vectors to be shared across threads
        Eigen::VectorXf X4_X3 = (XY2.block(0,2,XY2.rows(),1) -
                XY2.block(0,0,XY2.rows(),1)).cast<float>();
        Eigen::VectorXf Y4_Y3 = (XY2.block(0,3,XY2.rows(),1) -
                XY2.block(0,1,XY2.rows(),1)).cast<float>();
        Eigen::VectorXf X2_X1 = (XY1.block(0,2,XY1.rows(),1) -
                XY1.block(0,0,XY1.rows(),1)).cast<float>();
        Eigen::VectorXf Y2_Y1 = (XY1.block(0,3,XY1.rows(),1) -
                XY1.block(0,1,XY1.rows(),1)).cast<float>();

        Eigen::MatrixXf XY1f = XY1.cast<float>();
        Eigen::MatrixXf XY2f = XY2.cast<float>();

        // Allocate space on the GPU
        float *d_XY1, *d_XY2, *d_X4_X3, *d_Y4_Y3, *d_X2_X1, *d_Y2_Y1;
        int *d_adjacency, *d_cross;

        CUDA_CALL(cudaMalloc(&d_XY1,XY1.rows()*XY1.cols()*sizeof(float)));
        CUDA_CALL(cudaMemcpy(d_XY1,XY1f.data(),XY1.rows()*XY1.cols()*sizeof(
                float),cudaMemcpyHostToDevice));
        CUDA_CALL(cudaMalloc(&d_XY2,XY2.rows()*XY2.cols()*sizeof(float)));
        CUDA_CALL(cudaMemcpy(d_XY2,XY2f.data(),XY2.rows()*XY2.cols()*sizeof(
                float),cudaMemcpyHostToDevice));

        CUDA_CALL(cudaMalloc(&d_X4_X3,XY2.rows()*sizeof(float)));
        CUDA_CALL(cudaMemcpy(d_X4_X3,X4_X3.data(),XY2.rows()*sizeof(float),
                cudaMemcpyHostToDevice));
        CUDA_CALL(cudaMalloc(&d_Y4_Y3,XY2.rows()*sizeof(float)));
        CUDA_CALL(cudaMemcpy(d_Y4_Y3,Y4_Y3.data(),XY2.rows()*sizeof(float),
                cudaMemcpyHostToDevice));
        CUDA_CALL(cudaMalloc(&d_X2_X1,XY1.rows()*sizeof(float)));
        CUDA_CALL(cudaMemcpy(d_X2_X1,X2_X1.data(),XY1.rows()*sizeof(float),
                cudaMemcpyHostToDevice));
        CUDA_CALL(cudaMalloc(&d_Y2_Y1,XY1.rows()*sizeof(float)));
        CUDA_CALL(cudaMemcpy(d_Y2_Y1,Y2_Y1.data(),XY1.rows()*sizeof(float),
                cudaMemcpyHostToDevice));
        CUDA_CALL(cudaMalloc(&d_adjacency,XY1.rows()*XY2.rows()*sizeof(int)));
        CUDA_CALL(cudaMalloc(&d_cross,XY1.rows()*sizeof(int)));

        // Compute the road crossings for each transition
        int noCombos = XY1.rows()*XY2.rows();
        int noBlocks = (noCombos % maxThreadsPerBlock) ?
                (noCombos/maxThreadsPerBlock + 1) : (noCombos/
                maxThreadsPerBlock);
        double number = (double)(noBlocks)/(((double)maxBlocksPerGrid)*
                ((double)maxBlocksPerGrid));
        int blockYDim = ((number - floor(number)) > 0 ) ? (int)number + 1 :
                (int)number;
        int blockXDim = (int)min(maxBlocksPerGrid,noBlocks);

        dim3 dimGrid(blockXDim,blockYDim);
        pathAdjacencyKernel<<<dimGrid,maxThreadsPerBlock>>>(XY1.rows(),XY2.
                rows(),d_XY1,d_XY2,d_X4_X3,d_Y4_Y3,d_X2_X1,d_Y2_Y1,
                d_adjacency);
        CUDA_CALL(cudaPeekAtLastError());
        CUDA_CALL(cudaDeviceSynchronize());

        // Sum the number
        noBlocks = (XY1.rows() % maxThreadsPerBlock)? (int)(XY1.rows()/
                maxThreadsPerBlock + 1) : (int)(XY1.rows()/maxThreadsPerBlock);
        roadCrossingsKernel<<<noBlocks,maxThreadsPerBlock>>>(XY1.rows(),
                XY2.rows(),d_adjacency,d_cross);
        CUDA_CALL(cudaPeekAtLastError());
        CUDA_CALL(cudaDeviceSynchronize());

        // Retrieve results
        CUDA_CALL(cudaMemcpy(crossings.data(),d_cross,XY1.rows()*sizeof(int),
                cudaMemcpyDeviceToHost));
        // Free memory
        CUDA_CALL(cudaFree(d_XY1));
        CUDA_CALL(cudaFree(d_XY2));
        CUDA_CALL(cudaFree(d_X4_X3));
        CUDA_CALL(cudaFree(d_Y4_Y3));
        CUDA_CALL(cudaFree(d_X2_X1));
        CUDA_CALL(cudaFree(d_Y2_Y1));
        CUDA_CALL(cudaFree(d_adjacency));
        CUDA_CALL(cudaFree(d_cross));
    } catch (const char* err) {
        throw err;
    }
}

// Splits a 2-dimensional labelled region into discrete patches by finding
// the intersection of grid cells with contiguous regions with the same label.
// Saves the resulting patches to HabitatPatch objects for later use.
void SimulateGPU::buildPatches(int W, int H, int skpx, int skpy, int xres,
        int yres, int noRegions, double xspacing, double yspacing, double
        subPatchArea, HabitatTypePtr habTyp, const Eigen::MatrixXi&
        labelledImage, const Eigen::MatrixXd& populations,
        std::vector<HabitatPatchPtr>& patches, double& initPop,
        Eigen::VectorXd& initPops, Eigen::VectorXd& capacities, int&
        noPatches) {

    try {
        // Get device properties
        int device = 0;
        struct cudaDeviceProp properties;
        cudaGetDeviceProperties(&properties, device);
        int maxThreadsPerBlock = properties.maxThreadsPerBlock;

        Eigen::MatrixXf popsFloat = populations.cast<float>();

        float *results, *d_results, *d_populations;
        int *d_labelledImage;

        results = (float*)malloc(xres*yres*noRegions*5*sizeof(float));
        CUDA_CALL(cudaMalloc((void **)&d_results,xres*yres*noRegions*5*
                sizeof(float)));

        CUDA_CALL(cudaMalloc((void **)&d_labelledImage,H*W*sizeof(int)));
        CUDA_CALL(cudaMemcpy(d_labelledImage,labelledImage.data(),H*W*sizeof(
                int),cudaMemcpyHostToDevice));

        CUDA_CALL(cudaMalloc((void **)&d_populations,H*W*sizeof(float)));
        CUDA_CALL(cudaMemcpy(d_populations,popsFloat.data(),H*W*sizeof(float),
                cudaMemcpyHostToDevice));

        int noBlocks = ((xres*yres*noRegions) % maxThreadsPerBlock)? (int)(
                xres*yres*noRegions/maxThreadsPerBlock + 1) : (int)(xres*yres*
                noRegions/maxThreadsPerBlock);
        int noThreadsPerBlock = min(maxThreadsPerBlock,xres*yres*noRegions);

        patchComputation<<<noBlocks,noThreadsPerBlock>>>(xres*yres*noRegions,
                W, H, skpx, skpy, xres,yres,(float)subPatchArea,(float)
                xspacing,(float)yspacing,(float)habTyp->getMaxPop(),noRegions,
                d_labelledImage,d_populations,d_results);
        CUDA_CALL(cudaPeekAtLastError());
        CUDA_CALL(cudaDeviceSynchronize());

        CUDA_CALL(cudaMemcpy(results,d_results,xres*yres*noRegions*5*sizeof(
                float),cudaMemcpyDeviceToHost));

        // Now turn the results into patches
        for (int ii = 0; ii < xres*yres*noRegions; ii++) {
            if (results[5*ii] > 0) {
                // Create new patch to add to patches vector
                HabitatPatchPtr hab(new HabitatPatch());
                hab->setArea((double)results[5*ii]);
                hab->setCX((double)results[5*ii+3]);
                hab->setCY((double)results[5*ii+4]);
                hab->setPopulation((double)results[5*ii+2]);
                hab->setCapacity((double)results[5*ii+1]);
                hab->setType(habTyp);
                initPop += (double)results[5*ii+2];
                initPops(noPatches) = (double)results[5*ii+2];
                capacities(noPatches) = (double)results[5*ii+1];
                patches[noPatches++] = hab;
            }
        }

        CUDA_CALL(cudaFree(d_populations));
        CUDA_CALL(cudaFree(d_labelledImage));
        CUDA_CALL(cudaFree(d_results));
        free(results);
    } catch (const char* err) {
        throw err;
    }
}

// Computes the expected end population for a single species in a design region
// by performing Monte Carlo simulation. This uses the animal movement and
// road mortality model of Rhodes et al. (2014).
void SimulateGPU::simulateMTECUDA(SimulatorPtr sim,
        std::vector<SpeciesRoadPatchesPtr>& srp,
        std::vector<Eigen::VectorXd>& initPops,
        std::vector<Eigen::VectorXd>& capacities,
        Eigen::MatrixXd& endPops) {

    try {
        // Get device properties
        int device = 0;
        struct cudaDeviceProp properties;
        cudaGetDeviceProperties(&properties, device);
        int maxThreadsPerBlock = properties.maxThreadsPerBlock;

        // We convert all inputs to floats from double as CUDA is much faster
        // in single precision than double precision

        // Get important values for computation
        int nYears = sim->getRoad()->getOptimiser()->getEconomic()->getYears();
        int noPaths = sim->getRoad()->getOptimiser()->getOtherInputs()->
                getNoPaths();

        // Get experimental scenario multipliers
        ExperimentalScenarioPtr scenario = sim->getRoad()->getOptimiser()->
                getScenario();
        VariableParametersPtr varParams = sim->getRoad()->getOptimiser()->
                getVariableParams();

        // Get the important values for the road first and convert them to
        // formats that the kernel can use

        for (int ii = 0; ii < srp.size(); ii++) {

            // Species parameters
            float stepSize = (float)sim->getRoad()->getOptimiser()->
                    getEconomic()->getTimeStep();
            int nPatches = capacities[ii].size();

            float *speciesParams, *d_speciesParams, *eps, *d_initPops, *d_eps,
                    *d_caps;

            speciesParams = (float*)malloc(8*sizeof(float));
            CUDA_CALL(cudaMalloc((void**)&d_speciesParams,8*sizeof(float)));

            // Read in the information into the correct format
            speciesParams[0] = srp[ii]->getSpecies()->getGrowthRate()->
                    getCurrent()*varParams->getGrowthRatesMultipliers()(
                    scenario->getPopGR());
            speciesParams[1] = srp[ii]->getSpecies()->getGrowthRate()->
                    getMean()*varParams->getGrowthRatesMultipliers()(scenario->
                    getPopGR());
            speciesParams[2] = srp[ii]->getSpecies()->getGrowthRate()->
                    getNoiseSD()*varParams->getGrowthRateSDMultipliers()(
                    scenario->getPopGRSD());
            speciesParams[3] = srp[ii]->getSpecies()->getThreshold()*varParams
                    ->getPopulationLevels()(scenario->getPopLevel());
            speciesParams[4] = srp[ii]->getSpecies()->getGrowthRate()->
                    getMRStrength()*varParams->getGrowthRateSDMultipliers()(
                    scenario->getPopGRSD());
            speciesParams[5] = srp[ii]->getSpecies()->getGrowthRate()->
                    getPoissonJump()*varParams->getGrowthRateSDMultipliers()(
                    scenario->getPopGRSD());
            speciesParams[6] = srp[ii]->getSpecies()->getGrowthRate()->
                    getJumpProb()*varParams->getGrowthRateSDMultipliers()(
                    scenario->getPopGRSD());
            speciesParams[7] = srp[ii]->getSpecies()->getLocalVariability();

            CUDA_CALL(cudaMemcpy(d_speciesParams,speciesParams,8*sizeof(float),
                    cudaMemcpyHostToDevice));

            // RANDOM MATRICES
            float *d_growthRates, *d_uBrownianSpecies, *d_uJumpSizesSpecies,
                    *d_uJumpsSpecies;
            //allocate space for 100 floats on the GPU
            CUDA_CALL(cudaMalloc((void**)&d_growthRates,nYears*noPaths*nPatches
                    *sizeof(float)));
            CUDA_CALL(cudaMalloc((void**)&d_uBrownianSpecies,nYears*noPaths*
                    sizeof(float)));
            CUDA_CALL(cudaMalloc((void**)&d_uJumpSizesSpecies,nYears*noPaths*
                    sizeof(float)));
            CUDA_CALL(cudaMalloc((void**)&d_uJumpsSpecies,nYears*noPaths*
                    sizeof(float)));

            curandGenerator_t gen;
            srand(time(NULL));
            int _seed = rand();
            curandCreateGenerator(&gen, CURAND_RNG_PSEUDO_DEFAULT);
            curandSetPseudoRandomGeneratorSeed(gen, _seed);

            // Random matrices for growth rate parameter for species
            CURAND_CALL(curandGenerateNormal(gen, d_growthRates, nYears*noPaths
                    *nPatches,0.0f,1.0f));

            CURAND_CALL(curandGenerateNormal(gen, d_uBrownianSpecies, nYears*
                    noPaths, 0.0f,1.0f));

            CURAND_CALL(curandGenerateNormal(gen, d_uJumpSizesSpecies, nYears*
                    noPaths, 0.0f,1.0f));

            CURAND_CALL(curandGenerateUniform(gen, d_uJumpsSpecies, nYears*
                    noPaths));

            CURAND_CALL(curandDestroyGenerator(gen));

            // INITIAL POPULATIONS
            Eigen::VectorXf initPopsF = initPops[ii].cast<float>();
            CUDA_CALL(cudaMalloc((void**)&d_initPops,initPops[ii].size()*
                    sizeof(float)));
            CUDA_CALL(cudaMemcpy(d_initPops,initPopsF.data(),initPops[ii].
                    size()*sizeof(float),cudaMemcpyHostToDevice));

            // END POPULATIONS
            eps = (float*)malloc(noPaths*sizeof(float));
            CUDA_CALL(cudaMalloc((void**)&d_eps, noPaths*sizeof(float)));

            // TEMPORARY KERNEL POPULATIONS
            float *d_pathPops;
            CUDA_CALL(cudaMalloc((void**)&d_pathPops, noPaths*2*initPops[ii].
                    size()*sizeof(float)));

            // CAPACITIES
            Eigen::VectorXf capsF = capacities[ii].cast<float>();
            CUDA_CALL(cudaMalloc((void**)&d_caps,capacities[ii].size()*
                    sizeof(float)));
            CUDA_CALL(cudaMemcpy(d_caps,capsF.data(),capacities[ii].size()*
                    sizeof(float),cudaMemcpyHostToDevice));

            // MOVEMENT AND MORTALITY MATRIX
            // Convert the movement and mortality matrix to a sparse matrix for
            // use in the kernel efficiently.
            float *d_sparseOut;
            int *d_elemsPerCol, *d_rowIdx;

            {
                const Eigen::MatrixXd& transProbs = srp[ii]->getTransProbs();
                const Eigen::MatrixXd& survProbs = srp[ii]->getSurvivalProbs()[
                        srp[ii]->getSurvivalProbs().size()-1];
                Eigen::MatrixXf mmm = (transProbs.array()*survProbs.array()).
                        cast<float>();

                Eigen::MatrixXf sparseOut(mmm.rows(),mmm.cols());
                Eigen::VectorXi elemsPerCol(capacities[ii].size());
                Eigen::VectorXi rowIdx(mmm.rows()*mmm.cols());

                int totalElements;
                SimulateGPU::dense2Sparse(mmm.data(),capacities[ii].size(),
                        capacities[ii].size(),sparseOut.data(),elemsPerCol.
                        data(),rowIdx.data(),totalElements);

                // Allocate GPU memory for sparse matrix
                CUDA_CALL(cudaMalloc((void**)&d_sparseOut,totalElements*
                        sizeof(float)));
                CUDA_CALL(cudaMalloc((void**)&d_rowIdx,totalElements*sizeof(
                        int)));
                CUDA_CALL(cudaMalloc((void**)&d_elemsPerCol,capacities[ii].
                        size()*sizeof(int)));

                CUDA_CALL(cudaMemcpy(d_sparseOut,sparseOut.data(),
                        totalElements*sizeof(float),cudaMemcpyHostToDevice));
                CUDA_CALL(cudaMemcpy(d_rowIdx,rowIdx.data(),totalElements*
                        sizeof(int),cudaMemcpyHostToDevice));
                CUDA_CALL(cudaMemcpy(d_elemsPerCol,elemsPerCol.data(),
                        capacities[ii].size()*sizeof(int),
                        cudaMemcpyHostToDevice));
            }

            ///////////////////////////////////////////////////////////////////
            // Perform N simulation paths. Currently, there is no species
            // interaction, so we run each kernel separately and do not need to
            // use the Thrust library.

            // Modify the below code the run the kernel multiple times
            // depending on how many paths are required.

            // Blocks and threads for each path
            int noBlocks = (int)(noPaths % maxThreadsPerBlock)?
                    (int)(noPaths/maxThreadsPerBlock + 1) :
                    (int)(noPaths/maxThreadsPerBlock);
            int noThreadsPerBlock = min(noPaths,maxThreadsPerBlock);

            mteKernel<<<noBlocks,noThreadsPerBlock>>>(noPaths,nYears,
                    capacities[ii].size(),stepSize,d_growthRates,
                    d_uBrownianSpecies,d_uJumpSizesSpecies,d_uJumpsSpecies,
                    d_speciesParams,d_initPops,d_caps,d_sparseOut,d_rowIdx,
                    d_elemsPerCol,d_pathPops,d_eps);
            CUDA_CALL(cudaPeekAtLastError());
            CUDA_CALL(cudaDeviceSynchronize());

            // Retrieve results
            CUDA_CALL(cudaMemcpy(eps,d_eps,noPaths*sizeof(float),
                    cudaMemcpyDeviceToHost));

            for (int jj = 0; jj < noPaths; jj++) {
                endPops(jj,ii) = eps[jj];
            }

            // Free memory
            CUDA_CALL(cudaFree(d_growthRates));
            CUDA_CALL(cudaFree(d_uBrownianSpecies));
            CUDA_CALL(cudaFree(d_uJumpSizesSpecies));
            CUDA_CALL(cudaFree(d_uJumpsSpecies));
            CUDA_CALL(cudaFree(d_speciesParams));
            CUDA_CALL(cudaFree(d_initPops));
            CUDA_CALL(cudaFree(d_pathPops));
            CUDA_CALL(cudaFree(d_eps));
            CUDA_CALL(cudaFree(d_caps));
            CUDA_CALL(cudaFree(d_sparseOut));
            CUDA_CALL(cudaFree(d_rowIdx));
            CUDA_CALL(cudaFree(d_elemsPerCol));
            free(eps);
            free(speciesParams);
        }
    } catch (const char* err) {
        throw err;
    }
}

// Code for second paper. Performs optimal control of the traffic flow over
// time so as to maximise the expected future value of a road subject to
// ecological (animal populations) and economic (commodity and fuel prices)
// uncertainty over the design horizon. Uses Monte Carlo simulation.
void SimulateGPU::simulateROVCUDA(SimulatorPtr sim,
        std::vector<SpeciesRoadPatchesPtr>& srp,
        std::vector<Eigen::MatrixXd> &adjPops, Eigen::MatrixXd& unitProfits,
        Eigen::MatrixXd& condExp, Eigen::MatrixXi& optCont, Eigen::VectorXd&
        regressions, bool plotResults) {
    // Currently there is no species interaction. This can be a future question
    // and would be an interesting extension on how it can be implemented,
    // what the surrogate looks like and how the patches are formed.

    // The predictor variables are the adjusted population and current unit
    // profit. To determine the optimal control, we find the adjusted
    // populations for each species for each control and check against the
    // policy map.

    ///////////////////////////////////////////////////////////////////////////
    time_t begin;
    time_t end;
    double regressionsTime = 0;
    double pathsTime = 0;
    ///////////////////////////////////////////////////////////////////////////

    try {
        // Get device properties
        int device = 0;
        struct cudaDeviceProp properties;
        cudaGetDeviceProperties(&properties, device);
        int maxThreadsPerBlock = properties.maxThreadsPerBlock;

        // Scaling for relative importance of species population to unit profit
        // used in regressions.
        float scaling = 2.0;

        // Get general properties
        OptimiserPtr optimiser = sim->getRoad()->getOptimiser();
        ExperimentalScenarioPtr scenario = optimiser->getScenario();
        VariableParametersPtr varParams = optimiser->getVariableParams();
        int sc = scenario->getProgram();
        TrafficProgramPtr program = optimiser->getPrograms()[sc];
        std::vector<CommodityPtr> commodities = optimiser->getEconomic()->
                getCommodities();
        std::vector<CommodityPtr> fuels = optimiser->getEconomic()->
                getFuels();

        /////////////////////////////////////////
        // Plot the surrogate
//        GnuplotPtr plotPtr(new Gnuplot);
        /////////////////////////////////////////

        // Get important values for computation
        int nYears = sim->getRoad()->getOptimiser()->getEconomic()->getYears();
        int noPaths = sim->getRoad()->getOptimiser()->getOtherInputs()->
                getNoPaths();
        int noControls = program->getFlowRates().size();
        int noUncertainties = commodities.size() + fuels.size();

        // Fixed cost per unit traffic
        double unitCost = sim->getRoad()->getAttributes()->getUnitVarCosts();
        // Fuel consumption per vehicle class per unit traffic (L)
        Eigen::VectorXf fuelCosts = sim->getRoad()->getCosts()->
                getUnitFuelCost().cast<float>();
        // Load per unit traffic
        float unitRevenue = (float)sim->getRoad()->getCosts()->
                getUnitRevenue();
        float stepSize = (float)optimiser->getEconomic()->getTimeStep();
        float rrr = (float)optimiser->getEconomic()->getRRR();

        // Get the important values for the road first and convert them to
        // formats that the kernel can use

        // Initialise CUDA memory /////////////////////////////////////////////

        // 1. Transition and survival matrices for each species and each
        // control
        float *speciesParams, *uncertParams, *d_initPops, *d_tempPops,
                *d_capacities, *d_speciesParams, *d_uncertParams, *d_fuelCosts;

        int *noPatches, *d_noPatches;

        noPatches = (int*)malloc(srp.size()*sizeof(int));

        int patches = 0;
        int transition = 0;

        for (int ii = 0; ii < srp.size(); ii++) {
            noPatches[ii] = srp[ii]->getHabPatches().size();
            patches += noPatches[ii];
            transition += pow(noPatches[ii],2);
        }

        Eigen::VectorXf initPops(patches);
        Eigen::VectorXf capacities(patches);
        speciesParams = (float*)malloc(srp.size()*8*sizeof(float));
        uncertParams = (float*)malloc(noUncertainties*6*sizeof(float));

        CUDA_CALL(cudaMalloc((void**)&d_noPatches,srp.size()*sizeof(int)));
        CUDA_CALL(cudaMalloc((void**)&d_initPops,patches*sizeof(float)));
        CUDA_CALL(cudaMalloc((void**)&d_capacities,patches*sizeof(float)));
        CUDA_CALL(cudaMalloc((void**)&d_speciesParams,srp.size()*8*sizeof(
                float)));
        CUDA_CALL(cudaMalloc((void**)&d_uncertParams,noUncertainties*6*sizeof(
                float)));
        CUDA_CALL(cudaMalloc((void**)&d_tempPops,noPaths*(nYears+1)*patches*srp
                .size()*sizeof(float)));
        CUDA_CALL(cudaMalloc((void**)&d_fuelCosts,fuelCosts.size()*sizeof(
                float)));

        int counter0 = 0;
        int counter1 = 0;

        // Read in the information into the correct format
        for (int ii = 0; ii < srp.size(); ii++) {
            // This routine requires all species to have the same patches so
            // that there can be species interactions etc. Therefore, the
            // number of patches is the same.
            initPops.segment(counter0,srp[ii]->getHabPatches().size()) =
                    srp[ii]->getInitPops().cast<float>();
            capacities.segment(counter0,srp[ii]->getHabPatches().size()) =
                    srp[ii]->getCapacities().cast<float>();

            counter0 += srp[ii]->getHabPatches().size();

            speciesParams[counter1] = (float)srp[ii]->getSpecies()->
                    getGrowthRate()->getCurrent()*varParams->
                    getGrowthRatesMultipliers()(scenario->getPopGR());
            speciesParams[counter1+1] = (float)srp[ii]->getSpecies()->
                    getGrowthRate()->getMean()*varParams->
                    getGrowthRatesMultipliers()(scenario->getPopGR());
            speciesParams[counter1+2] = (float)srp[ii]->getSpecies()->
                    getGrowthRate()->getNoiseSD()*varParams->
                    getGrowthRateSDMultipliers()(scenario->getPopGRSD());
            speciesParams[counter1+3] = (float)srp[ii]->getSpecies()->
                    getThreshold()*varParams->getPopulationLevels()(scenario->
                    getPopLevel());
            speciesParams[counter1+4] = (float)srp[ii]->getSpecies()->
                    getGrowthRate()->getMRStrength()*varParams->
                    getGrowthRateSDMultipliers()(scenario->getPopGRSD());
            speciesParams[counter1+5] = (float)srp[ii]->getSpecies()->
                    getGrowthRate()->getPoissonJump()*varParams->
                    getGrowthRateSDMultipliers()(scenario->getPopGRSD());
            speciesParams[counter1+6] = (float)srp[ii]->getSpecies()->
                    getGrowthRate()->getJumpProb()*varParams->
                    getGrowthRateSDMultipliers()(scenario->getPopGRSD());
            speciesParams[counter1+7] = (float)srp[ii]->getSpecies()->
                    getLocalVariability();

            counter1 += 8;
        }

        for (int ii = 0; ii < fuels.size(); ii++) {
            uncertParams[6*ii] = (float)fuels[ii]->getCurrent();
            uncertParams[6*ii+1] = (float)fuels[ii]->getMean()*varParams->
                    getCommodityMultipliers()(scenario->getCommodity());
            uncertParams[6*ii+2] = (float)fuels[ii]->getNoiseSD()*varParams->
                    getCommoditySDMultipliers()(scenario->getCommoditySD());
            uncertParams[6*ii+3] = (float)fuels[ii]->getMRStrength()*varParams
                    ->getCommoditySDMultipliers()(scenario->getCommoditySD());
            uncertParams[6*ii+4] = (float)fuels[ii]->getPoissonJump()*varParams
                    ->getCommoditySDMultipliers()(scenario->getCommoditySD());
            uncertParams[6*ii+5] = (float)fuels[ii]->getJumpProb()*varParams->
                    getCommoditySDMultipliers()(scenario->getCommoditySD());
        }

        // Set the fuel indices for the vehicle classes corresponding to the
        // fuels order above
        Eigen::VectorXi fuelIdx(optimiser->getTraffic()->getVehicles().size());

        for (int ii = 0; ii < fuelIdx.size(); ii++) {
            CommodityPtr fuel = (optimiser->getTraffic()->getVehicles())[ii]->
                    getFuel();

            for (int jj = 0; jj < fuels.size(); jj++) {
                if (fuel == fuels[jj]) {
                    fuelIdx(ii) = jj;
                    break;
                } else if ((jj+1) == fuels.size()) {
                    fuelIdx(ii) = 0;
                }
            }
        }

        for (int ii = 0; ii < commodities.size(); ii++) {
            uncertParams[fuels.size()*6 + 6*ii] = (float)commodities[ii]->
                    getCurrent();
            uncertParams[fuels.size()*6 + 6*ii+1] = (float)commodities[ii]->
                    getMean()*varParams->getCommodityMultipliers()(scenario->
                    getCommodity());
            uncertParams[fuels.size()*6 + 6*ii+2] = (float)commodities[ii]->
                    getNoiseSD()*varParams->getCommoditySDMultipliers()(
                    scenario->getCommoditySD());
            uncertParams[fuels.size()*6 + 6*ii+3] = (float)commodities[ii]->
                    getMRStrength()*varParams->getCommoditySDMultipliers()(
                    scenario->getCommoditySD());;
            uncertParams[fuels.size()*6 + 6*ii+4] = (float)commodities[ii]->
                    getPoissonJump()*varParams->getCommoditySDMultipliers()(
                    scenario->getCommoditySD());
            uncertParams[fuels.size()*6 + 6*ii+5] = (float)commodities[ii]->
                    getJumpProb()*varParams->getCommoditySDMultipliers()(
                    scenario->getCommoditySD());
        }

        // Transfer the data to the device
        CUDA_CALL(cudaMemcpy(d_noPatches,noPatches,srp.size()*sizeof(int),
                cudaMemcpyHostToDevice));
        CUDA_CALL(cudaMemcpy(d_initPops,initPops.data(),patches*sizeof(float),
                cudaMemcpyHostToDevice));
        CUDA_CALL(cudaMemcpy(d_capacities,capacities.data(),patches*sizeof(
                float),cudaMemcpyHostToDevice));
        CUDA_CALL(cudaMemcpy(d_speciesParams,speciesParams,srp.size()*8*
                sizeof(float),cudaMemcpyHostToDevice));
        CUDA_CALL(cudaMemcpy(d_uncertParams,uncertParams,noUncertainties*6*
                sizeof(float),cudaMemcpyHostToDevice));
        CUDA_CALL(cudaMemcpy(d_fuelCosts,fuelCosts.data(),fuelCosts.size()*
                sizeof(float),cudaMemcpyHostToDevice));

        // Free the host memory
        free(speciesParams);
        free(uncertParams);

        // MOVEMENT AND MORTALITY MATRICES
        // Convert the movement and mortality matrix to a sparse matrix for
        // use in the kernel efficiently.
        float *d_sparseOut, *sparseOutAll;
        int *d_elemsPerCol, *d_rowIdx, *elemsPerColAll, *rowIdxAll;

        sparseOutAll = (float*)malloc(srp.size()*patches*patches*noControls*
                sizeof(float));
        rowIdxAll = (int*)malloc(srp.size()*patches*patches*noControls*
                sizeof(int));
        elemsPerColAll = (int*)malloc(srp.size()*patches*noControls*sizeof(
                int));

        int maxElements = 0;

        for (int ii = 0; ii < srp.size(); ii++) {
            const Eigen::MatrixXd& transProbs = srp[ii]->getTransProbs();

            for (int jj = 0; jj < noControls; jj++) {
                const Eigen::MatrixXd& survProbs = srp[ii]->getSurvivalProbs()
                        [jj];
                Eigen::MatrixXf mmm = (transProbs.array()*survProbs.array()).
                        cast<float>();

                Eigen::MatrixXf sparseOut(mmm.rows(),mmm.cols());
                Eigen::VectorXi elemsPerCol(patches);
                Eigen::VectorXi rowIdx(mmm.rows()*mmm.cols());

                int totalElements;
                SimulateGPU::dense2Sparse(mmm.data(),patches,patches,
                        sparseOut.data(),elemsPerCol.data(),rowIdx.data(),
                        totalElements);

                if (totalElements > maxElements) {
                    maxElements = totalElements;
                }

                memcpy(sparseOutAll + (ii*noControls + jj)*patches*patches,
                       sparseOut.data(),totalElements*sizeof(float));
                memcpy(rowIdxAll + (ii*noControls + jj)*patches*patches,
                       rowIdx.data(),totalElements*sizeof(int));
                memcpy(elemsPerColAll + (ii*noControls + jj)*patches,
                       elemsPerCol.data(),patches*sizeof(int));
            }
        }

        // Allocate GPU memory for sparse matrix
        CUDA_CALL(cudaMalloc((void**)&d_sparseOut,maxElements*srp.size()*
                noControls*sizeof(float)));
        CUDA_CALL(cudaMalloc((void**)&d_rowIdx,maxElements*srp.size()*
                noControls*sizeof(int)));
        CUDA_CALL(cudaMalloc((void**)&d_elemsPerCol,patches*srp.size()*
                noControls*sizeof(int)));

        for (int ii = 0; ii < srp.size(); ii++) {
            for (int jj = 0; jj < noControls; jj++) {
                CUDA_CALL(cudaMemcpy(d_sparseOut + (ii*noControls + jj)*
                        maxElements,sparseOutAll + (ii*noControls + jj)*
                        patches*patches,maxElements*sizeof(float),
                        cudaMemcpyHostToDevice));
                CUDA_CALL(cudaMemcpy(d_rowIdx + (ii*noControls + jj)*
                        maxElements,rowIdxAll + (ii*noControls + jj)*patches*
                        patches,maxElements*sizeof(int),
                        cudaMemcpyHostToDevice));
                CUDA_CALL(cudaMemcpy(d_elemsPerCol + (ii*noControls + jj)*
                        patches,elemsPerColAll + (ii*noControls + jj)*patches,
                        patches*sizeof(int),cudaMemcpyHostToDevice));
            }
        }

        free(sparseOutAll);
        free(rowIdxAll);
        free(elemsPerColAll);

        // Exogenous parameters (fuels, commodities)
        // Ore composition is simply Gaussian for now
        float *d_randCont, *d_growthRate, *d_uBrownian, *d_uJumpSizes,
                *d_uJumps, *d_uResults, *d_uComposition, *d_flowRates,
                *d_uBrownianSpecies, *d_uJumpSizesSpecies, *d_uJumpsSpecies;
        int *d_controls;

        srand(time(NULL));
        int _seed = rand();
        curandGenerator_t gen;
        CURAND_CALL(curandCreateGenerator(&gen, CURAND_RNG_PSEUDO_DEFAULT));
        CURAND_CALL(curandSetPseudoRandomGeneratorSeed(gen, _seed));

        CUDA_CALL(cudaMalloc((void**)&d_randCont,nYears*noPaths*sizeof(
                float)));

        // 2. Random matrices for randomised control
        CURAND_CALL(curandGenerateUniform(gen, d_randCont, nYears*noPaths));

        CUDA_CALL(cudaMalloc((void**)&d_controls,nYears*noPaths*sizeof(int)));

        int noBlocks = (int)(noPaths*nYears % maxThreadsPerBlock) ?
                (int)(noPaths*nYears/maxThreadsPerBlock  + 1) :
                (int)(noPaths*nYears/maxThreadsPerBlock);
        int noThreadsPerBlock = min(noPaths*nYears,maxThreadsPerBlock);

        randControls<<<noBlocks,noThreadsPerBlock>>>(noPaths,nYears,noControls,
                d_randCont,d_controls);
        CUDA_CALL(cudaPeekAtLastError());
        CUDA_CALL(cudaDeviceSynchronize());

        // The flow rates corresponding to the random controls
        Eigen::MatrixXf flowRatesF = program->getFlowRates().cast<float>();
        CUDA_CALL(cudaMalloc((void**)&d_flowRates,noControls*sizeof(float)));
        CUDA_CALL(cudaMemcpy(d_flowRates,flowRatesF.data(),noControls*sizeof(
                float),cudaMemcpyHostToDevice));

        // We no longer need the floating point random controls vector
        CUDA_CALL(cudaFree(d_randCont));

        // Endogenous uncertainty
        CUDA_CALL(cudaMalloc((void**)&d_growthRate,nYears*noPaths*patches*srp
                .size()*sizeof(float)));
        CUDA_CALL(cudaMalloc((void**)&d_uBrownianSpecies,nYears*noPaths*srp.
                size()*sizeof(float)));
        CUDA_CALL(cudaMalloc((void**)&d_uJumpSizesSpecies,nYears*noPaths*srp
                .size()*sizeof(float)));
        CUDA_CALL(cudaMalloc((void**)&d_uJumpsSpecies,nYears*noPaths*srp.size()
                *sizeof(float)));

        // Exogenous uncertainty
        CUDA_CALL(cudaMalloc((void**)&d_uBrownian,nYears*noPaths*
                noUncertainties*sizeof(float)));
        CUDA_CALL(cudaMalloc((void**)&d_uJumpSizes,nYears*noPaths*
                noUncertainties*sizeof(float)));
        CUDA_CALL(cudaMalloc((void**)&d_uJumps,nYears*noPaths*noUncertainties*
                sizeof(float)));
        CUDA_CALL(cudaMalloc((void**)&d_uResults,noUncertainties*nYears*
                noPaths*sizeof(float)));
        CUDA_CALL(cudaMalloc((void**)&d_uComposition,nYears*noPaths*(
                commodities.size())*sizeof(float)));

        // 3. Random matrices for growth rate parameter for species
        CURAND_CALL(curandGenerateNormal(gen, d_growthRate, nYears*noPaths*
                patches*srp.size(),0.0f,1.0f));

        CURAND_CALL(curandGenerateNormal(gen, d_uBrownianSpecies, nYears*
                noPaths*srp.size(),0.0f,1.0f));

        CURAND_CALL(curandGenerateNormal(gen, d_uJumpSizesSpecies, nYears*
                noPaths*srp.size(),0.0f,1.0f));

        CURAND_CALL(curandGenerateUniform(gen, d_uJumpsSpecies, nYears*noPaths*
                srp.size()));

        // 4. Random matrices for other uncertainties
        CURAND_CALL(curandGenerateNormal(gen, d_uBrownian, nYears*noPaths*
                noUncertainties,0.0f,1.0f));

        CURAND_CALL(curandGenerateNormal(gen, d_uJumpSizes, nYears*noPaths*
                noUncertainties,0.0f,1.0f));

        CURAND_CALL(curandGenerateUniform(gen, d_uJumps, nYears*noPaths*
                noUncertainties));

        // 5. Ore composition paths
        for (int ii = 0; ii < commodities.size(); ii++) {
            CURAND_CALL(curandGenerateNormal(gen, d_uComposition + ii*nYears*
                    noPaths,nYears*noPaths,commodities[ii]->getOreContent(),
                    commodities[ii]->getOreContentSD()*varParams->
                    getCommodityPropSD()(scenario->getOreCompositionSD())));
        }

        // Destroy generator
        CURAND_CALL(curandDestroyGenerator(gen));

        // Finally, allocate space on the device for the path results. This is
        // what we use in our policy map.
        float *d_totalPops, *d_aars;
        CUDA_CALL(cudaMalloc(&d_totalPops,srp.size()*(nYears+1)*noPaths*sizeof(
                float)));
        CUDA_CALL(cudaMalloc(&d_aars,srp.size()*(nYears+1)*noPaths*noControls*
                sizeof(float)));

        // We will only use up to 48kB of shared memory at a time
        // We share patch information as well as overall AAR for each control
        int maxElems = *std::max_element(noPatches,noPatches + srp.size())*srp.
                size()*2 + noControls;
        int maxThreadsPerBlock1 = (int)(48000/(maxElems*sizeof(float)));

        // Compute forward paths (CUDA kernel)
        noBlocks = (int)(noPaths % maxThreadsPerBlock1) ?
                (int)(noPaths/maxThreadsPerBlock1 + 1) :
                (int)(noPaths/maxThreadsPerBlock1);

    //    printControls<<<1,1>>>(noPaths,0,nYears,d_controls);

    //    time_t begin = clock();
        free(noPatches);

        ///////////////////////////////////////////////////////////////////////////
        begin = clock();
        ///////////////////////////////////////////////////////////////////////////
        forwardPathKernel<<<noBlocks,maxThreadsPerBlock1>>>(noPaths,nYears,
                srp.size(),patches,noControls,noUncertainties,stepSize,
                d_initPops,d_tempPops,d_sparseOut,d_rowIdx,d_elemsPerCol,
                maxElements,d_speciesParams,d_capacities,d_aars,d_uncertParams,
                d_controls,d_uJumps,d_uBrownian,d_uJumpSizes,d_uJumpsSpecies,
                d_uBrownianSpecies,d_uJumpSizesSpecies,d_growthRate,d_uResults,
                d_totalPops);
        CUDA_CALL(cudaPeekAtLastError());
        CUDA_CALL(cudaDeviceSynchronize());
        ///////////////////////////////////////////////////////////////////////////
        end = clock();
        pathsTime += double(end - begin)/CLOCKS_PER_SEC;
        ///////////////////////////////////////////////////////////////////////////

    //    printAverages<<<1,50>>>(nYears,srp.size(),noControls,noPaths,d_totalPops,d_aars);
    //    CUDA_CALL(cudaPeekAtLastError());
    //    CUDA_CALL(cudaDeviceSynchronize());

    //    time_t end = clock();
    //    double elapsed_secs = double(end - begin) / CLOCKS_PER_SEC;
    //    std::cout << "Forward paths time: " << elapsed_secs << " s" << std::endl;

        // Free device memory that is no longer needed
        CUDA_CALL(cudaFree(d_uBrownian));
        CUDA_CALL(cudaFree(d_uJumpSizes));
        CUDA_CALL(cudaFree(d_uJumps));
        CUDA_CALL(cudaFree(d_uBrownianSpecies));
        CUDA_CALL(cudaFree(d_uJumpSizesSpecies));
        CUDA_CALL(cudaFree(d_uJumpsSpecies));

        // Determine the number of uncertainties. The uncertainties are the
        // unit payoff of the road (comprised of commodity and fuel prices,
        // which are pre-processed to determine covariances etc. and are
        // treated as a single uncertainty) and the adjusted population of each
        // species under each control.
        int noDims = srp.size() + 1;
        int dimRes = sim->getRoad()->getOptimiser()->getOtherInputs()->
                getDimRes();

        // Prepare the floating point versions of the output
        float *d_condExp;
        // Prepare the index of the fuel use by each vehicle class
        int *d_fuelIdx;

        CUDA_CALL(cudaMalloc(&d_fuelIdx,fuelIdx.size()*sizeof(int)));
        CUDA_CALL(cudaMemcpy(d_fuelIdx,fuelIdx.data(),fuelIdx.size()*sizeof(
                int),cudaMemcpyHostToDevice));

        // Where to copy the results back to the host in floating point ready
        // to copy to the double precision outputs.
        std::vector<Eigen::MatrixXf> adjPopsF(nYears+1);

        for (int ii = 0; ii <= nYears; ii++) {
            adjPopsF[ii].resize(noPaths,srp.size());
        }

        // 2. Unit profits map inputs at each time step
        Eigen::MatrixXf unitProfitsF(nYears+1,noPaths);

        float *d_unitProfits;
        CUDA_CALL(cudaMalloc((void**)&d_unitProfits,unitProfitsF.rows()*
                unitProfitsF.cols()*sizeof(float)));

        // 3. Optimal profit-to-go outputs matrix (along each path)
        Eigen::MatrixXf condExpF(noPaths,nYears+1);

        CUDA_CALL(cudaMalloc((void**)&d_condExp,condExp.rows()*condExp.cols()*
                sizeof(float)));

        int* d_optCont;
        CUDA_CALL(cudaMalloc((void**)&d_optCont,optCont.rows()*optCont.cols()*
                sizeof(int)));

        // 5. Regression Data
        Eigen::VectorXf regressionsF(nYears*noControls*(dimRes*noDims + pow(
                dimRes,noDims)*2));

        // Make the grid for regressions. One for each control for each
        // time step mapped against the N-dimensional grid. We interpolate
        // when we compute the forward paths.
        float *d_regression, *d_adjPops, *d_stats;

        CUDA_CALL(cudaMalloc((void**)&d_regression,nYears*noControls*(
                dimRes*noDims + pow(dimRes,noDims)*2)*sizeof(float)));
        CUDA_CALL(cudaMalloc((void**)&d_adjPops,adjPops[0].rows()*
                adjPops[0].cols()*sizeof(float)));

        Eigen::VectorXf stats(3);
        CUDA_CALL(cudaMalloc((void**)&d_stats,3*sizeof(float)));

        // Choose the appropriate method for backwards induction
        switch (optimiser->getROVMethod()) {

            case Optimiser::ALGO1:
            {
                // I will not code this method
            }
            break;

            case Optimiser::ALGO2:
            {
                // I will not code this method
            }
            break;

            case Optimiser::ALGO3:
            {
                // I will not code this method
            }
            break;

            case Optimiser::ALGO4:
            // Same as ALGO6 but without forward path recomputation
            {
                backwardInduction<<<noBlocks,maxThreadsPerBlock1>>>(nYears,
                        noPaths,nYears,srp.size(),noControls,noUncertainties,
                        stepSize,unitCost,unitRevenue,rrr,fuels.size(),
                        commodities.size(),dimRes,d_flowRates,d_fuelCosts,
                        d_totalPops,d_speciesParams,d_controls,d_aars,
                        d_regression,d_uComposition,d_uResults,d_fuelIdx,
                        d_condExp,d_optCont,d_adjPops,d_unitProfits);
                CUDA_CALL(cudaPeekAtLastError());
                CUDA_CALL(cudaDeviceSynchronize());

                // Copy the adjusted populations to the output variable. This
                // is only provided for completeness. The algorithm does not
                // use the results as they pertain to the very last time step.
                CUDA_CALL(cudaMemcpy(adjPopsF[nYears].data(),d_adjPops,srp.
                        size()*noPaths*sizeof(float),cudaMemcpyDeviceToHost));

                // Find the maximum and minimum x value along each dimension
                // for the dependant variables. This allocates space for the
                // input variables for the regressions.
                float *d_xmaxes, *d_xmins;
                CUDA_CALL(cudaMalloc((void**)&d_xmaxes,noControls*noDims*
                        sizeof(float)));
                CUDA_CALL(cudaMalloc((void**)&d_xmins,noControls*noDims*sizeof(
                        float)));

                // For each backward step not including the last period, we
                // need to determine the adjusted population for each species
                // and the unit payoffs.
                for (int ii = nYears-1; ii > 0; ii--) {
                    // Perform regression and save results
                    int noBlocks2 = (int)((int)pow(dimRes,noDims)*noControls %
                            maxThreadsPerBlock) ? (int)(pow(dimRes,noDims)*
                            noControls/maxThreadsPerBlock + 1) : (int)(
                            pow(dimRes,noDims)*noControls/maxThreadsPerBlock);
                    int maxThreadsPerBlock2 = min((int)pow(dimRes,noDims)*
                            noControls,maxThreadsPerBlock);

                    float *d_xin, *d_xvals, *d_yvals;
                    int* d_dataPoints, *d_controlsTemp;
                    // The data points are arranged so that the number of rows
                    // equals the number of dimensions and the number of
                    // columns equals the number of data points.
                    CUDA_CALL(cudaMalloc((void**)&d_xin,noDims*noPaths*sizeof(
                            float)));
                    CUDA_CALL(cudaMalloc((void**)&d_xvals,noControls*noDims*
                            noPaths*sizeof(float)));
                    CUDA_CALL(cudaMalloc((void**)&d_yvals,noControls*noPaths*
                            sizeof(float)));
                    CUDA_CALL(cudaMalloc((void**)&d_dataPoints,noControls*
                            sizeof(int)));
                    CUDA_CALL(cudaMalloc((void**)&d_controlsTemp,noPaths*
                            sizeof(int)));

                    // Compute the state values
                    computePathStates<<<noBlocks,noThreadsPerBlock>>>(noPaths,
                            noDims,nYears,noControls,ii,unitCost,unitRevenue,
                            d_controls,fuels.size(),d_fuelCosts,d_uResults,
                            d_uComposition,noUncertainties,d_fuelIdx,
                            commodities.size(),d_aars,d_totalPops,d_xin,
                            d_controlsTemp);
                    CUDA_CALL(cudaPeekAtLastError());
                    CUDA_CALL(cudaDeviceSynchronize());

                    // This kernel does not take advantage of massive
                    // parallelism. It is simply to allow us to call data that
                    // is already on the device for allocating data for use in
                    // the regressions. We pass the species data as we need to
                    // make sure that only in-the-money paths are considered
                    // for each control.
                    allocateXYRegressionData<<<1,1>>>(noPaths,noControls,
                            noDims,nYears,d_speciesParams,ii,d_controlsTemp,
                            d_xin,d_condExp,d_dataPoints,d_xvals,d_yvals);
                    CUDA_CALL(cudaPeekAtLastError());
                    CUDA_CALL(cudaDeviceSynchronize());

                    // Get the minimum and maximum X value for each dimension
                    // for each control.
                    computeStateMinMax<<<1,1>>>(noControls,noDims,noPaths,
                            d_dataPoints,d_xvals,d_xmins,d_xmaxes);
                    CUDA_CALL(cudaPeekAtLastError());
                    CUDA_CALL(cudaDeviceSynchronize());

                    // Allocate the k nearest neighbours for each design point
                    // in order to do the regression stage after this. This
                    // component is the slowest component.
                    int *dataPoints;
                    dataPoints = (int*)malloc(noControls*sizeof(int));
                    CUDA_CALL(cudaMemcpy(dataPoints,d_dataPoints,noControls*
                            sizeof(int),cudaMemcpyDeviceToHost));

                    for (int jj = 0; jj < noControls; jj++) {
                        // Let k be the natural logarithm of the number of data
                        // points
                        int k = 3*(log(dataPoints[jj]) + 1);

                        // We first need to perform a k nearest neighbour
                        // search
                        Eigen::MatrixXf ref(dataPoints[jj],noDims);
                        Eigen::MatrixXf query((int)pow(dimRes,noDims),noDims);
                        Eigen::MatrixXf dist((int)pow(dimRes,noDims),k);
                        Eigen::MatrixXi ind((int)pow(dimRes,noDims),k);

                        float *d_queryPts, *d_dist;
                        int *d_ind;
                        CUDA_CALL(cudaMalloc((void**)&d_queryPts,pow(dimRes,
                                noDims)*noDims*sizeof(float)));
                        CUDA_CALL(cudaMalloc((void**)&d_dist,pow(dimRes,noDims)
                                *k*sizeof(float)));
                        CUDA_CALL(cudaMalloc((void**)&d_ind,pow(dimRes,noDims)
                                *k*sizeof(int)));

                        createQueryPoints<<<noBlocks2,maxThreadsPerBlock2>>>(
                                (int)pow(dimRes,noDims),noDims,dimRes,jj,
                                noControls,ii,d_xmins,d_xmaxes,d_regression,
                                d_queryPts);
                        CUDA_CALL(cudaPeekAtLastError());
                        CUDA_CALL(cudaDeviceSynchronize());

                        CUDA_CALL(cudaMemcpy(query.data(),d_queryPts,pow(
                                dimRes,noDims)*noDims*sizeof(float),
                                cudaMemcpyDeviceToHost));
                        CUDA_CALL(cudaFree(d_queryPts));

                        // Transfer sample data points to the host
                        for (int kk = 0; kk < noDims; kk++) {
                            CUDA_CALL(cudaMemcpy(ref.data() + kk*dataPoints[
                                    jj],d_xvals + jj*noPaths*noDims + kk*
                                    noPaths,dataPoints[jj]*sizeof(float),
                                    cudaMemcpyDeviceToHost));
                        }

                        // Send the reference point data to a temporary array
                        // for use in the multiple local linear regression.
                        float* d_refX;
                        CUDA_CALL(cudaMalloc((void**)&d_refX,dataPoints[jj]*
                                noDims*sizeof(float)));
                        CUDA_CALL(cudaMemcpy(d_refX,ref.data(),dataPoints[jj]*
                                noDims*sizeof(float),cudaMemcpyHostToDevice));

                        // We need to normalise in each dimension before
                        // performing the regression. (We also convert the unit
                        // profit component to a log scale. NOT IMPLEMENTED
                        // YET). As the profit component is not as important as
                        // the population component for making immediate
                        // decisions, we give it a smaller scale. We
                        // arbitrarily choose a factor of 4 for now.
                        Eigen::VectorXf xmins(noDims);
                        Eigen::VectorXf xmaxes(noDims);
                        CUDA_CALL(cudaMemcpy(xmins.data(),d_xmins,noDims*
                                sizeof(float),cudaMemcpyDeviceToHost));
                        CUDA_CALL(cudaMemcpy(xmaxes.data(),d_xmaxes,noDims*
                                sizeof(float),cudaMemcpyDeviceToHost));
                        // Species components
                        for(int kk = 0; kk < (noDims-1); kk++) {
                            ref.block(0,kk,dataPoints[jj],1) = (ref.block(0,kk,
                                    dataPoints[jj],1).array() - xmins(kk))/(
                                    xmaxes(kk) - xmins(kk));
                            query.block(0,kk,pow(dimRes,noDims),1) = (query.
                                    block(0,kk,pow(dimRes,noDims),1).array() -
                                    xmins(kk))/(xmaxes(kk) - xmins(kk));
                        }
                        // Profit component
                        ref.block(0,(noDims-1),dataPoints[jj],1) = (ref.block(
                                0,(noDims-1),dataPoints[jj],1).array() - xmins(
                                (noDims-1)))/(scaling*(xmaxes((noDims-1)) -
                                xmins((noDims -1))));
                        query.block(0,(noDims-1),pow(dimRes,noDims),1) = (query
                                .block(0,(noDims-1),pow(dimRes,noDims),1).
                                array() - xmins((noDims-1)))/(scaling*(xmaxes((
                                noDims-1)) - xmins((noDims-1))));

                        // Compute the knn searches
                        knn_cuda_with_indexes::knn(ref.data(),dataPoints[jj],
                                query.data(),pow(dimRes,noDims),noDims,k,dist
                                .data(),ind.data());
                        CUDA_CALL(cudaPeekAtLastError());
                        CUDA_CALL(cudaDeviceSynchronize());

                        CUDA_CALL(cudaMemcpy(d_dist,dist.data(),pow(dimRes,
                                noDims)*k*sizeof(float),
                                cudaMemcpyHostToDevice));

                        CUDA_CALL(cudaMemcpy(d_ind,ind.data(),pow(dimRes,
                                noDims)*k*sizeof(float),
                                cudaMemcpyHostToDevice));

                        // Due to normalisation, we need to make the minimum
                        // and maximum values for each dimension 0 and 1,
                        // respectively.
                        float *d_xminsN, *d_xmaxesN;
                        Eigen::VectorXf xminsN = Eigen::VectorXf::Zero(noDims);
                        Eigen::VectorXf xmaxesN = Eigen::VectorXf::Constant(
                                noDims,1);
                        CUDA_CALL(cudaMalloc((void**)&d_xminsN,noDims*sizeof(
                                float)));
                        CUDA_CALL(cudaMalloc((void**)&d_xmaxesN,noDims*sizeof(
                                float)));
                        CUDA_CALL(cudaMemcpy(d_xminsN,xminsN.data(),noDims*
                                sizeof(float),cudaMemcpyHostToDevice));
                        CUDA_CALL(cudaMemcpy(d_xmaxesN,xmaxesN.data(),noDims*
                                sizeof(float),cudaMemcpyHostToDevice));

                        // Perform the regression for this control at this time
                        // at each of the query points.
                        multiLocLinReg<<<noBlocks2,maxThreadsPerBlock2>>>((int)
                                pow(dimRes,noDims),noDims,dimRes,nYears,
                                noControls,ii,jj,k,d_dataPoints,d_refX,d_yvals+
                                jj*noPaths,d_regression,d_xminsN,d_xmaxesN,
                                d_dist,d_ind);
                        CUDA_CALL(cudaPeekAtLastError());
                        CUDA_CALL(cudaDeviceSynchronize());

                        CUDA_CALL(cudaFree(d_dist));
                        CUDA_CALL(cudaFree(d_ind));
                        CUDA_CALL(cudaFree(d_refX));
                        CUDA_CALL(cudaFree(d_xminsN));
                        CUDA_CALL(cudaFree(d_xmaxesN));

//                        // Test plots /////////////////////////////////////////////
//                        // Prepare raw data
//                        for(int kk = 0; kk < (noDims-1); kk++) {
//                            ref.block(0,kk,dataPoints[jj],1) = (ref.block(0,kk,
//                                    dataPoints[jj],1)*(xmaxes(kk) - xmins(kk)))
//                                    .array() + xmins(kk);
//                        }

//                        ref.block(0,noDims-1,dataPoints[jj],1) = (ref.block(0,
//                                noDims-1,dataPoints[jj],1)*(scaling*(xmaxes(noDims
//                                -1) - xmins(noDims-1)))).array() + xmins(noDims-1);

//    //                    ref.block(0,1,dataPoints[jj],1) = (-1*((ref.block(0,1,
//    //                            dataPoints[jj],1)*(xmaxes(1) - xmins(1)))
//    //                            .array() + xmins(1) -1)).array().log();

//                        std::vector<std::vector<double>> raw;
//                        raw.resize(dataPoints[jj]);

//                        // Raw data points copy
//                        Eigen::VectorXf yVals(dataPoints[jj]);
//                        CUDA_CALL(cudaMemcpy(yVals.data(),d_yvals + jj*noPaths,
//                                dataPoints[jj]*sizeof(float),
//                                cudaMemcpyDeviceToHost));

//                        for (int kk = 0; kk < dataPoints[jj]; kk++) {
//                            raw[kk].resize(noDims+1);
//                            raw[kk][0] = ref.data()[kk];
//                            raw[kk][1] = ref.data()[dataPoints[jj] + kk];
//                            raw[kk][2] = yVals(kk);
//                        }

//                        // Regressed data points
//                        // XVALS
//                        Eigen::MatrixXf regX(dimRes,noDims);
//                        CUDA_CALL(cudaMemcpy(regX.data(),d_regression + ii*
//                                noControls*(dimRes*noDims + (int)pow(dimRes,noDims)
//                                *2) + jj*(dimRes*noDims + (int)pow(dimRes,noDims)
//                                *2),dimRes*noDims*sizeof(float),
//                                cudaMemcpyDeviceToHost));
//                        // YVALS
//                        Eigen::VectorXf surrogate((int)pow(dimRes,noDims));
//                        CUDA_CALL(cudaMemcpy(surrogate.data(),d_regression + ii*
//                                noControls*(dimRes*noDims + (int)pow(dimRes,noDims)
//                                *2) + jj*(dimRes*noDims + (int)pow(dimRes,noDims)
//                                *2) + dimRes*noDims,pow(dimRes,noDims)*sizeof(
//                                float),cudaMemcpyDeviceToHost));

//                        // Prepare regressed data
//                        std::vector<std::vector<std::vector<double>>> reg;
//                        reg.resize(dimRes);

//                        for (int kk = 0; kk < dimRes; kk++) {
//                            reg[kk].resize(dimRes);
//                            for (int ll = 0; ll < dimRes; ll++) {
//                                reg[kk][ll].resize(noDims+1);
//                            }
//                        }

//                        for (int kk = 0; kk < dimRes; kk++) {
//                            for (int ll = 0; ll < dimRes; ll++) {
//                                reg[kk][ll][0] = regX(ll,0);
//                                reg[kk][ll][1] = regX(kk,1);
//                                reg[kk][ll][2] = surrogate(kk + ll*dimRes);
//                            }
//                        }

//                        // Plot 2 (Multiple linear regression model)
//                        (*plotPtr) << "set title 'Multiple regression'\n";
//                        (*plotPtr) << "set grid\n";
//                    //    (*plotPtr) << "set hidden3d\n";
//                        (*plotPtr) << "unset key\n";
//                        (*plotPtr) << "unset view\n";
//                        (*plotPtr) << "unset pm3d\n";
//                        (*plotPtr) << "unset xlabel\n";
//                        (*plotPtr) << "unset ylabel\n";
//                        (*plotPtr) << "set xrange [*:*]\n";
//                        (*plotPtr) << "set yrange [*:*]\n";
//                        (*plotPtr) << "set view 45,45\n";
//    //                    (*plotPtr) << "splot '-' with points pointtype 7\n";
//    //                  (*plotPtr) << "splot '-' with lines,\n";
//                        (*plotPtr) << "splot '-' with lines, '-' with points pointtype 7\n";
//                        (*plotPtr).send2d(reg);
//                        (*plotPtr).send1d(raw);
//                        (*plotPtr).flush();

                        ///////////////////////////////////////////////////////
                    }

                    free(dataPoints);

//                    if (ii == 1) {
//                        std::cout << "final step" << std::endl;
//                    } else if (ii == 0) {
//                        std::cout << "final step" << std::endl;
//                    }

                    backwardInduction<<<noBlocks,maxThreadsPerBlock1>>>(ii,
                            noPaths,nYears,srp.size(),noControls,noUncertainties,
                            stepSize,unitCost,unitRevenue,rrr,fuels.size(),
                            commodities.size(),dimRes,d_flowRates,d_fuelCosts,
                            d_totalPops,d_speciesParams,d_controls,d_aars,
                            d_regression,d_uComposition,d_uResults,d_fuelIdx,
                            d_condExp,d_optCont,d_adjPops,d_unitProfits);
                    CUDA_CALL(cudaPeekAtLastError());
                    CUDA_CALL(cudaDeviceSynchronize());

                    // Copy the adjusted populations for this time step to the
                    // output variables. The conditional expectations, optimal
                    // controls and unit profits are copied as well if we are
                    // producing the policy map for the optimal road.
                    CUDA_CALL(cudaMemcpy(adjPopsF[ii].data(),d_adjPops,srp.
                            size()*noPaths*sizeof(float),
                            cudaMemcpyDeviceToHost));

                    CUDA_CALL(cudaFree(d_xvals));
                    CUDA_CALL(cudaFree(d_yvals));
                    CUDA_CALL(cudaFree(d_dataPoints));
                    CUDA_CALL(cudaFree(d_xin));
                    CUDA_CALL(cudaFree(d_controlsTemp));
                }

                // Compute the average expected payoff for each control by
                // taking the average across all paths for each control and
                // adding the first period's payoff for that control. As we do
                // not perform forward path recomputation, we simply take the
                // average across all paths for each control.
                firstPeriodInduction<<<1,1>>>(noPaths,nYears,srp.size(),
                        noControls,stepSize,unitCost,unitRevenue,rrr,fuels.
                        size(),commodities.size(),d_flowRates,d_fuelCosts,
                        d_totalPops,d_speciesParams,d_controls,d_aars,
                        d_uComposition,d_uResults,d_fuelIdx,d_condExp,
                        d_optCont,d_stats);
                CUDA_CALL(cudaPeekAtLastError());
                CUDA_CALL(cudaDeviceSynchronize());

                // Free memory
                CUDA_CALL(cudaPeekAtLastError());
                CUDA_CALL(cudaDeviceSynchronize());
                CUDA_CALL(cudaFree(d_xmaxes));
                CUDA_CALL(cudaFree(d_xmins));
            }
            break;

            case Optimiser::ALGO5:
            {
                // I will not code this method
            }
            break;

            case Optimiser::ALGO6:
            // Full model with local linear kernel and forward path
            // recomputation. This method is the most accurate but is very
            // slow.
            {
                // The last step is simply the valid control with the highest
                // single period payoff

    //            time_t begin = clock();

                ///////////////////////////////////////////////////////////////////////////
                begin = clock();
                ///////////////////////////////////////////////////////////////////////////
                optimalForwardPaths<<<noBlocks,maxThreadsPerBlock1,maxElems*
                        maxThreadsPerBlock1*sizeof(float)>>>(nYears,noPaths,
                        nYears,srp.size(),patches,noControls,noUncertainties,
                        stepSize,unitCost,unitRevenue,rrr,fuels.size(),
                        commodities.size(),dimRes,d_flowRates,d_fuelCosts,
                        d_tempPops,d_totalPops,d_sparseOut,d_rowIdx,
                        d_elemsPerCol,maxElements,d_speciesParams,d_growthRate,
                        d_capacities,d_controls,d_aars,d_regression,
                        d_uComposition,d_uResults,d_fuelIdx,d_condExp,
                        d_optCont,d_adjPops,d_unitProfits);
                CUDA_CALL(cudaPeekAtLastError());
                CUDA_CALL(cudaDeviceSynchronize());

                ///////////////////////////////////////////////////////////////////////////
                end = clock();
                pathsTime += double(end - begin)/CLOCKS_PER_SEC;
                ///////////////////////////////////////////////////////////////////////////


                // For testing
    //            Eigen::VectorXf tempCondExp(noPaths);
    //            CUDA_CALL(cudaMemcpy(tempCondExp.data(),d_condExp+(nYears-1)*
    //                    noPaths,noPaths*sizeof(float),cudaMemcpyDeviceToHost));

    //            time_t end = clock();
    //            double elapsed_secs = double(end - begin) / CLOCKS_PER_SEC;
    //            std::cout << "Optimal forward paths time: " << elapsed_secs << " s" << std::endl;

                // Copy the adjusted populations to the output variable. This
                // is only provided for completeness. The algorithm does not
                // use the results as they pertain to the very last time step.
                CUDA_CALL(cudaMemcpy(adjPopsF[nYears].data(),d_adjPops,srp.
                        size()*noPaths*sizeof(float),cudaMemcpyDeviceToHost));

                // Find the maximum and minimum x value along each dimension
                // for the dependant variables. This allocates space for the
                // input variables for the regressions.
                float *d_xmaxes, *d_xmins;
                CUDA_CALL(cudaMalloc((void**)&d_xmaxes,noControls*noDims*
                        sizeof(float)));
                CUDA_CALL(cudaMalloc((void**)&d_xmins,noControls*noDims*sizeof(
                        float)));

                // For each backward step not including the last period, we
                // need to determine the adjusted population for each species
                // and the unit payoffs.
                for (int ii = nYears-1; ii > 0; ii--) {
                    // Perform regression and save results
                    int noBlocks2 = (int)((int)pow(dimRes,noDims)*noControls %
                            maxThreadsPerBlock) ? (int)(pow(dimRes,noDims)*
                            noControls/maxThreadsPerBlock + 1) : (int)(
                            pow(dimRes,noDims)*noControls/maxThreadsPerBlock);
                    int maxThreadsPerBlock2 = min((int)pow(dimRes,noDims)*
                            noControls,maxThreadsPerBlock);

                    float *d_xin, *d_xvals, *d_yvals;
                    int* d_dataPoints, *d_controlsTemp;
                    // The data points are arranged so that the number of rows
                    // equals the number of dimensions and the number of
                    // columns equals the number of data points.
                    CUDA_CALL(cudaMalloc((void**)&d_xin,noDims*noPaths*sizeof(
                            float)));
                    CUDA_CALL(cudaMalloc((void**)&d_xvals,noControls*noDims*
                            noPaths*sizeof(float)));
                    CUDA_CALL(cudaMalloc((void**)&d_yvals,noControls*noPaths*
                            sizeof(float)));
                    CUDA_CALL(cudaMalloc((void**)&d_dataPoints,noControls*
                            sizeof(int)));
                    CUDA_CALL(cudaMalloc((void**)&d_controlsTemp,noPaths*
                            sizeof(int)));

                    // Compute the state values
                    computePathStates<<<noBlocks,noThreadsPerBlock>>>(noPaths,
                            noDims,nYears,noControls,ii,unitCost,unitRevenue,
                            d_controls,fuels.size(),d_fuelCosts,d_uResults,
                            d_uComposition,noUncertainties,d_fuelIdx,
                            commodities.size(),d_aars,d_totalPops,d_xin,
                            d_controlsTemp);
                    CUDA_CALL(cudaPeekAtLastError());
                    CUDA_CALL(cudaDeviceSynchronize());

                    // This kernel does not take advantage of massive
                    // parallelism. It is simply to allow us to call data that
                    // is already on the device for allocating data for use in
                    // the regressions. We pass the species data as we need to
                    // make sure that only in-the-money paths are considered
                    // for each control.
                    allocateXYRegressionData<<<1,1>>>(noPaths,noControls,
                            noDims,nYears,d_speciesParams,ii,d_controlsTemp,
                            d_xin,d_condExp,d_dataPoints,d_xvals,d_yvals);
                    CUDA_CALL(cudaPeekAtLastError());
                    CUDA_CALL(cudaDeviceSynchronize());

                    // Get the minimum and maximum X value for each dimension
                    // for each control.
                    computeStateMinMax<<<1,1>>>(noControls,noDims,noPaths,
                            d_dataPoints,d_xvals,d_xmins,d_xmaxes);
                    CUDA_CALL(cudaPeekAtLastError());
                    CUDA_CALL(cudaDeviceSynchronize());

                    // Allocate the k nearest neighbours for each design point
                    // in order to do the regression stage after this. This
                    // component is the slowest component.
                    int *dataPoints;
                    dataPoints = (int*)malloc(noControls*sizeof(int));
                    CUDA_CALL(cudaMemcpy(dataPoints,d_dataPoints,noControls*
                            sizeof(int),cudaMemcpyDeviceToHost));

                    for (int jj = 0; jj < noControls; jj++) {
                        // Let k be the natural logarithm of the number of data
                        // points
                        int k = 3*(log(dataPoints[jj]) + 1);

                        // We first need to perform a k nearest neighbour
                        // search
                        Eigen::MatrixXf ref(dataPoints[jj],noDims);
                        Eigen::MatrixXf query((int)pow(dimRes,noDims),noDims);
                        Eigen::MatrixXf dist((int)pow(dimRes,noDims),k);
                        Eigen::MatrixXi ind((int)pow(dimRes,noDims),k);

                        float *d_queryPts, *d_dist;
                        int *d_ind;
                        CUDA_CALL(cudaMalloc((void**)&d_queryPts,pow(dimRes,
                                noDims)*noDims*sizeof(float)));
                        CUDA_CALL(cudaMalloc((void**)&d_dist,pow(dimRes,noDims)
                                *k*sizeof(float)));
                        CUDA_CALL(cudaMalloc((void**)&d_ind,pow(dimRes,noDims)
                                *k*sizeof(int)));

                        createQueryPoints<<<noBlocks2,maxThreadsPerBlock2>>>(
                                (int)pow(dimRes,noDims),noDims,dimRes,jj,
                                noControls,ii,d_xmins,d_xmaxes,d_regression,
                                d_queryPts);
                        CUDA_CALL(cudaPeekAtLastError());
                        CUDA_CALL(cudaDeviceSynchronize());

                        CUDA_CALL(cudaMemcpy(query.data(),d_queryPts,pow(
                                dimRes,noDims)*noDims*sizeof(float),
                                cudaMemcpyDeviceToHost));
                        CUDA_CALL(cudaFree(d_queryPts));

                        // Transfer sample data points to the host
                        for (int kk = 0; kk < noDims; kk++) {
                            CUDA_CALL(cudaMemcpy(ref.data() + kk*dataPoints[
                                    jj],d_xvals + jj*noPaths*noDims + kk*
                                    noPaths,dataPoints[jj]*sizeof(float),
                                    cudaMemcpyDeviceToHost));
                        }

                        // Send the reference point data to a temporary array
                        // for use in the multiple local linear regression.
                        float* d_refX;
                        CUDA_CALL(cudaMalloc((void**)&d_refX,dataPoints[jj]*
                                noDims*sizeof(float)));
                        CUDA_CALL(cudaMemcpy(d_refX,ref.data(),dataPoints[jj]*
                                noDims*sizeof(float),cudaMemcpyHostToDevice));

                        // We need to normalise in each dimension before
                        // performing the regression. (We also convert the unit
                        // profit component to a log scale. NOT IMPLEMENTED
                        // YET). As the profit component is not as important as
                        // the population component for making immediate
                        // decisions, we give it a smaller scale. We
                        // arbitrarily choose a factor of 4 for now.
                        Eigen::VectorXf xmins(noDims);
                        Eigen::VectorXf xmaxes(noDims);
                        CUDA_CALL(cudaMemcpy(xmins.data(),d_xmins,noDims*
                                sizeof(float),cudaMemcpyDeviceToHost));
                        CUDA_CALL(cudaMemcpy(xmaxes.data(),d_xmaxes,noDims*
                                sizeof(float),cudaMemcpyDeviceToHost));
                        // Species components
                        for(int kk = 0; kk < (noDims-1); kk++) {
                            ref.block(0,kk,dataPoints[jj],1) = (ref.block(0,kk,
                                    dataPoints[jj],1).array() - xmins(kk))/(
                                    xmaxes(kk) - xmins(kk));
                            query.block(0,kk,pow(dimRes,noDims),1) = (query.
                                    block(0,kk,pow(dimRes,noDims),1).array() -
                                    xmins(kk))/(xmaxes(kk) - xmins(kk));
                        }
                        // Profit component
                        ref.block(0,(noDims-1),dataPoints[jj],1) = (ref.block(
                                0,(noDims-1),dataPoints[jj],1).array() - xmins(
                                (noDims-1)))/(scaling*(xmaxes((noDims-1)) -
                                xmins((noDims -1))));
                        query.block(0,(noDims-1),pow(dimRes,noDims),1) = (query
                                .block(0,(noDims-1),pow(dimRes,noDims),1).
                                array() - xmins((noDims-1)))/(scaling*(xmaxes((
                                noDims-1)) - xmins((noDims-1))));

                        // Compute the knn searches
                        knn_cuda_with_indexes::knn(ref.data(),dataPoints[jj],
                                query.data(),pow(dimRes,noDims),noDims,k,dist
                                .data(),ind.data());
                        CUDA_CALL(cudaPeekAtLastError());
                        CUDA_CALL(cudaDeviceSynchronize());

                        CUDA_CALL(cudaMemcpy(d_dist,dist.data(),pow(dimRes,
                                noDims)*k*sizeof(float),
                                cudaMemcpyHostToDevice));

                        CUDA_CALL(cudaMemcpy(d_ind,ind.data(),pow(dimRes,
                                noDims)*k*sizeof(float),
                                cudaMemcpyHostToDevice));

                        // Due to normalisation, we need to make the minimum
                        // and maximum values for each dimension 0 and 1,
                        // respectively.
                        float *d_xminsN, *d_xmaxesN;
                        Eigen::VectorXf xminsN = Eigen::VectorXf::Zero(noDims);
                        Eigen::VectorXf xmaxesN = Eigen::VectorXf::Constant(
                                noDims,1);
                        CUDA_CALL(cudaMalloc((void**)&d_xminsN,noDims*sizeof(
                                float)));
                        CUDA_CALL(cudaMalloc((void**)&d_xmaxesN,noDims*sizeof(
                                float)));
                        CUDA_CALL(cudaMemcpy(d_xminsN,xminsN.data(),noDims*
                                sizeof(float),cudaMemcpyHostToDevice));
                        CUDA_CALL(cudaMemcpy(d_xmaxesN,xmaxesN.data(),noDims*
                                sizeof(float),cudaMemcpyHostToDevice));

                        // Perform the regression for this control at this time
                        // at each of the query points.
                        ///////////////////////////////////////////////////////////////////////////
                        begin = clock();
                        ///////////////////////////////////////////////////////////////////////////
                        multiLocLinReg<<<noBlocks2,maxThreadsPerBlock2>>>((int)
                                pow(dimRes,noDims),noDims,dimRes,nYears,
                                noControls,ii,jj,k,d_dataPoints,d_refX,d_yvals+
                                jj*noPaths,d_regression,d_xminsN,d_xmaxesN,
                                d_dist,d_ind);
                        CUDA_CALL(cudaPeekAtLastError());
                        CUDA_CALL(cudaDeviceSynchronize());
                        ///////////////////////////////////////////////////////////////////////////
                        end = clock();
                        regressionsTime += double(end - begin)/CLOCKS_PER_SEC;
                        ///////////////////////////////////////////////////////////////////////////

                        CUDA_CALL(cudaFree(d_dist));
                        CUDA_CALL(cudaFree(d_ind));
                        CUDA_CALL(cudaFree(d_refX));
                        CUDA_CALL(cudaFree(d_xminsN));
                        CUDA_CALL(cudaFree(d_xmaxesN));

//                        // Test plots /////////////////////////////////////////////
//                        // Prepare raw data
//                        for(int kk = 0; kk < (noDims-1); kk++) {
//                            ref.block(0,kk,dataPoints[jj],1) = (ref.block(0,kk,
//                                    dataPoints[jj],1)*(xmaxes(kk) - xmins(kk)))
//                                    .array() + xmins(kk);
//                        }

//                        ref.block(0,noDims-1,dataPoints[jj],1) = (ref.block(0,
//                                noDims-1,dataPoints[jj],1)*(scaling*(xmaxes(noDims
//                                -1) - xmins(noDims-1)))).array() + xmins(noDims-1);

//    //                    ref.block(0,1,dataPoints[jj],1) = (-1*((ref.block(0,1,
//    //                            dataPoints[jj],1)*(xmaxes(1) - xmins(1)))
//    //                            .array() + xmins(1) -1)).array().log();

//                        std::vector<std::vector<double>> raw;
//                        raw.resize(dataPoints[jj]);

//                        // Raw data points copy
//                        Eigen::VectorXf yVals(dataPoints[jj]);
//                        CUDA_CALL(cudaMemcpy(yVals.data(),d_yvals + jj*noPaths,
//                                dataPoints[jj]*sizeof(float),
//                                cudaMemcpyDeviceToHost));

//                        for (int kk = 0; kk < dataPoints[jj]; kk++) {
//                            raw[kk].resize(noDims+1);
//                            raw[kk][0] = ref.data()[kk];
//                            raw[kk][1] = ref.data()[dataPoints[jj] + kk];
//                            raw[kk][2] = yVals(kk);
//                        }

//                        // Regressed data points
//                        // XVALS
//                        Eigen::MatrixXf regX(dimRes,noDims);
//                        CUDA_CALL(cudaMemcpy(regX.data(),d_regression + ii*
//                                noControls*(dimRes*noDims + (int)pow(dimRes,noDims)
//                                *2) + jj*(dimRes*noDims + (int)pow(dimRes,noDims)
//                                *2),dimRes*noDims*sizeof(float),
//                                cudaMemcpyDeviceToHost));
//                        // YVALS
//                        Eigen::VectorXf surrogate((int)pow(dimRes,noDims));
//                        CUDA_CALL(cudaMemcpy(surrogate.data(),d_regression + ii*
//                                noControls*(dimRes*noDims + (int)pow(dimRes,noDims)
//                                *2) + jj*(dimRes*noDims + (int)pow(dimRes,noDims)
//                                *2) + dimRes*noDims,pow(dimRes,noDims)*sizeof(
//                                float),cudaMemcpyDeviceToHost));

//                        // Prepare regressed data
//                        std::vector<std::vector<std::vector<double>>> reg;
//                        reg.resize(dimRes);

//                        for (int kk = 0; kk < dimRes; kk++) {
//                            reg[kk].resize(dimRes);
//                            for (int ll = 0; ll < dimRes; ll++) {
//                                reg[kk][ll].resize(noDims+1);
//                            }
//                        }

//                        for (int kk = 0; kk < dimRes; kk++) {
//                            for (int ll = 0; ll < dimRes; ll++) {
//                                reg[kk][ll][0] = regX(ll,0);
//                                reg[kk][ll][1] = regX(kk,1);
//                                reg[kk][ll][2] = surrogate(kk + ll*dimRes);
//                            }
//                        }

//                        // Plot 2 (Multiple linear regression model)
//                        (*plotPtr) << "set title 'Multiple regression'\n";
//                        (*plotPtr) << "set grid\n";
//                    //    (*plotPtr) << "set hidden3d\n";
//                        (*plotPtr) << "unset key\n";
//                        (*plotPtr) << "unset view\n";
//                        (*plotPtr) << "unset pm3d\n";
//                        (*plotPtr) << "unset xlabel\n";
//                        (*plotPtr) << "unset ylabel\n";
//                        (*plotPtr) << "set xrange [*:*]\n";
//                        (*plotPtr) << "set yrange [*:*]\n";
//                        (*plotPtr) << "set view 45,45\n";
//    //                    (*plotPtr) << "splot '-' with points pointtype 7\n";
//    //                  (*plotPtr) << "splot '-' with lines,\n";
//                        (*plotPtr) << "splot '-' with lines, '-' with points pointtype 7\n";
//                        (*plotPtr).send2d(reg);
//                        (*plotPtr).send1d(raw);
//                        (*plotPtr).flush();

                        ///////////////////////////////////////////////////////
                    }

                    free(dataPoints);

//                    if (ii == 0) {
//                        std::cout << "final step" << std::endl;
//                    }

                    // Recompute forward paths
                    ///////////////////////////////////////////////////////////////////////////
                    begin = clock();
                    ///////////////////////////////////////////////////////////////////////////
                    optimalForwardPaths<<<noBlocks,maxThreadsPerBlock1,maxElems
                            *maxThreadsPerBlock1*sizeof(float)>>>(ii,noPaths,
                            nYears,srp.size(),patches,noControls,
                            noUncertainties,stepSize,unitCost,unitRevenue,rrr,
                            fuels.size(),commodities.size(),dimRes,d_flowRates,
                            d_fuelCosts,d_tempPops,d_totalPops,d_sparseOut,
                            d_rowIdx,d_elemsPerCol,maxElements,d_speciesParams,
                            d_growthRate,d_capacities,d_controls,d_aars,
                            d_regression,d_uComposition,d_uResults,d_fuelIdx,
                            d_condExp,d_optCont,d_adjPops,d_unitProfits);
                    CUDA_CALL(cudaPeekAtLastError());
                    CUDA_CALL(cudaDeviceSynchronize());
                    ///////////////////////////////////////////////////////////////////////////
                    end = clock();
                    pathsTime += double(end - begin)/CLOCKS_PER_SEC;
                    ///////////////////////////////////////////////////////////////////////////

                    // Copy the adjusted populations for this time step to the
                    // output variables. The conditional expectations, optimal
                    // controls and unit profits are copied as well if we are
                    // producing the policy map for the optimal road.
                    CUDA_CALL(cudaMemcpy(adjPopsF[ii].data(),d_adjPops,srp.
                            size()*noPaths*sizeof(float),
                            cudaMemcpyDeviceToHost));

                    CUDA_CALL(cudaFree(d_xvals));
                    CUDA_CALL(cudaFree(d_yvals));
                    CUDA_CALL(cudaFree(d_dataPoints));
                    CUDA_CALL(cudaFree(d_xin));
                    CUDA_CALL(cudaFree(d_controlsTemp));
                }

                // Compute the optimal control, overall value and overall value
                // standard deviation at the first time period.
                firstPeriodInduction<<<1,1>>>(noPaths,nYears,srp.size(),
                        noControls,stepSize,unitCost,unitRevenue,rrr,fuels.
                        size(),commodities.size(),d_flowRates,d_fuelCosts,
                        d_totalPops,d_speciesParams,d_controls,d_aars,
                        d_uComposition,d_uResults,d_fuelIdx,d_condExp,
                        d_optCont,d_stats);
                CUDA_CALL(cudaPeekAtLastError());
                CUDA_CALL(cudaDeviceSynchronize());

                // Recompute the full forward paths
                optimalForwardPaths<<<noBlocks,maxThreadsPerBlock1,maxElems
                        *maxThreadsPerBlock1*sizeof(float)>>>(0,noPaths,nYears,
                        srp.size(),patches,noControls,noUncertainties,stepSize,
                        unitCost,unitRevenue,rrr,fuels.size(),
                        commodities.size(),dimRes,d_flowRates,d_fuelCosts,
                        d_tempPops,d_totalPops,d_sparseOut,d_rowIdx,
                        d_elemsPerCol,maxElements,d_speciesParams,d_growthRate,
                        d_capacities,d_controls,d_aars,d_regression,
                        d_uComposition,d_uResults,d_fuelIdx,d_condExp,
                        d_optCont,d_adjPops,d_unitProfits);
                CUDA_CALL(cudaPeekAtLastError());
                CUDA_CALL(cudaDeviceSynchronize());

                // Free memory
                CUDA_CALL(cudaFree(d_xmaxes));
                CUDA_CALL(cudaFree(d_xmins));
            }
            break;

            case Optimiser::ALGO7:
            // From Zhang et al. 2016
            {
                // Compute global linear regression

                // For each backward step
                for (int ii = nYears; ii > 0; ii--) {

                }
            }
            break;

            default:
            {
            }
            break;
        }

        // Copy the conditional expectations, optimal controls and unit
        // profits for to the output variables to host memory and then to
        // double precision to the output variables (where needed).
        CUDA_CALL(cudaMemcpy(optCont.data(),d_optCont,optCont.rows()*
                optCont.cols()*sizeof(int),cudaMemcpyDeviceToHost));

        CUDA_CALL(cudaMemcpy(condExpF.data(),d_condExp,condExp.rows()*
                condExp.cols()*sizeof(float),cudaMemcpyDeviceToHost));
        condExp = condExpF.cast<double>();

        CUDA_CALL(cudaMemcpy(unitProfitsF.data(),d_unitProfits,
                unitProfitsF.rows()*unitProfitsF.cols()*sizeof(float),
                cudaMemcpyDeviceToHost));
        unitProfits = unitProfitsF.cast<double>();

        CUDA_CALL(cudaMemcpy(regressionsF.data(),d_regression,regressionsF.
                size()*sizeof(float),cudaMemcpyDeviceToHost));
        regressions = regressionsF.cast<double>();

        for (int ii = 0; ii <= nYears; ii++) {
            adjPops[ii] = adjPopsF[ii].cast<double>();
        }

        CUDA_CALL(cudaMemcpy(stats.data(),d_stats,3*sizeof(float),
                cudaMemcpyDeviceToHost));

        // Save the attributes to the road
        sim->getRoad()->getAttributes()->setVarProfitIC(condExp(0));
        sim->getRoad()->getAttributes()->setTotalUtilisationROV(stats(0));
        sim->getRoad()->getAttributes()->setTotalUtilisationROVSD(stats(2));

        // Free remaining device memory
        CUDA_CALL(cudaFree(d_unitProfits));
        CUDA_CALL(cudaFree(d_initPops));
        CUDA_CALL(cudaFree(d_tempPops));
        CUDA_CALL(cudaFree(d_capacities));
        CUDA_CALL(cudaFree(d_noPatches));
        CUDA_CALL(cudaFree(d_speciesParams));
        CUDA_CALL(cudaFree(d_uncertParams));
        CUDA_CALL(cudaFree(d_fuelCosts));
        CUDA_CALL(cudaFree(d_growthRate));
        CUDA_CALL(cudaFree(d_uResults));
        CUDA_CALL(cudaFree(d_uComposition));
        CUDA_CALL(cudaFree(d_flowRates));
        CUDA_CALL(cudaFree(d_controls));
        CUDA_CALL(cudaFree(d_regression));
        CUDA_CALL(cudaFree(d_adjPops));

        CUDA_CALL(cudaFree(d_condExp));
        CUDA_CALL(cudaFree(d_optCont));
        CUDA_CALL(cudaFree(d_fuelIdx));

        // Remove these here?
        CUDA_CALL(cudaFree(d_totalPops));
        CUDA_CALL(cudaFree(d_aars));
        CUDA_CALL(cudaFree(d_stats));
    } catch (const char* err) {
        throw err;
    }
}

//void SimulateGPU::simulateSingleROVPath(SimulatorPtr sim, std::vector<
//        Eigen::MatrixXd>& visualisePops, Eigen::VectorXi& visualiseFlows,
//        Eigen::VectorXd& visualiseUnitProfits) {

//}

void SimulateGPU::buildSurrogateMTECUDA(RoadGAPtr op, int speciesID) {

    try {
        // Get device properties
        int device = 0;
        struct cudaDeviceProp properties;
        cudaGetDeviceProperties(&properties, device);
        int maxThreadsPerBlock = properties.maxThreadsPerBlock;

        // Pertinent parameters
        int dimRes = op->getSurrDimRes();
        int samples = op->getNoSamples();
        // Use the 10 nearest points for now
        int k;
        if (samples < 50) {
            k = 5;
        } else if (samples < 100) {
            k = 7;
        } else if (samples < 500) {
            k = (int)ceil(0.02*(samples-100)+7);
        } else {
            k = (int)ceil(log(samples)+8);
        }

        // MEAN VALUES ////////////////////////////////////////////////////////
        // Convert to floating point
        Eigen::VectorXf surrogateF(dimRes*2);
        Eigen::VectorXf predictors(samples);
        Eigen::VectorXf population = op->getPops().block(0,speciesID,samples,1)
                .cast<float>();
        Eigen::VectorXf populationSD = op->getPopsSD().block(0,speciesID,
                samples,1).cast<float>();

        predictors = op->getIARS().block(0,speciesID,samples,1).cast<float>();

        // Call regression kernel for computing the mean and standard deviation
        int noBlocks = (int)(dimRes % maxThreadsPerBlock) ? (int)(dimRes/
                maxThreadsPerBlock + 1) : (int)(dimRes/maxThreadsPerBlock);
        maxThreadsPerBlock = min(dimRes,maxThreadsPerBlock);

        // Arrange the incoming information
        float *d_xmin, *d_xmax, *d_xvals, *d_yvals, *d_surrogate;
        float xmin = predictors.minCoeff();
        float xmax = predictors.maxCoeff();
        int *d_samples;
        CUDA_CALL(cudaMalloc((void**)&d_samples,sizeof(int)));
        CUDA_CALL(cudaMalloc((void**)&d_xmin,sizeof(float)));
        CUDA_CALL(cudaMalloc((void**)&d_xmax,sizeof(float)));
        CUDA_CALL(cudaMemcpy(d_xmin,&xmin,sizeof(float),
                cudaMemcpyHostToDevice));
        CUDA_CALL(cudaMemcpy(d_xmax,&xmax,sizeof(float),
                cudaMemcpyHostToDevice));

        CUDA_CALL(cudaMalloc((void**)&d_xvals,samples*sizeof(float)));
        CUDA_CALL(cudaMalloc((void**)&d_yvals,samples*sizeof(float)));
        CUDA_CALL(cudaMemcpy(d_xvals,predictors.data(),samples*sizeof(float),
                cudaMemcpyHostToDevice));
        CUDA_CALL(cudaMemcpy(d_samples,&samples,sizeof(int),
                cudaMemcpyHostToDevice));

        CUDA_CALL(cudaMalloc((void**)&d_surrogate,dimRes*2*sizeof(float)));

        // Allocate the k nearest neighbours for each design point in order to
        // do the regression stage after this. This component is the slowest
        // routine. K nearest neighbours search
        float *query, *dist;
        int *ind;
        query = (float*)malloc(dimRes*sizeof(float));
        dist = (float*)malloc(dimRes*k*sizeof(float));
        ind = (int*)malloc(dimRes*k*sizeof(float));

        float *d_queryPts, *d_dist;
        int *d_ind;
        CUDA_CALL(cudaMalloc((void**)&d_queryPts,dimRes*sizeof(float)));
        CUDA_CALL(cudaMalloc((void**)&d_dist,dimRes*k*sizeof(float)));
        CUDA_CALL(cudaMalloc((void**)&d_ind,dimRes*k*sizeof(int)));

        createQueryPoints<<<noBlocks,maxThreadsPerBlock>>>(dimRes,1,dimRes,0,1,
                0,d_xmin,d_xmax,d_surrogate,d_queryPts);
        CUDA_CALL(cudaPeekAtLastError());
        CUDA_CALL(cudaDeviceSynchronize());

        CUDA_CALL(cudaMemcpy(query,d_queryPts,dimRes*sizeof(float),
                cudaMemcpyDeviceToHost));
        CUDA_CALL(cudaFree(d_queryPts));

        // Compute the knn searches
        if (samples < k) {
            // We cannot have more nearest neighbours than are data points
            k = samples;
        }

        knn_cuda_with_indexes::knn(predictors.data(),samples,query,dimRes,1,k,
                dist,ind);

        CUDA_CALL(cudaPeekAtLastError());
        CUDA_CALL(cudaDeviceSynchronize());

        CUDA_CALL(cudaMemcpy(d_dist,dist,dimRes*k*sizeof(float),
                cudaMemcpyHostToDevice));
        CUDA_CALL(cudaMemcpy(d_ind,ind,dimRes*k*sizeof(int),
                cudaMemcpyHostToDevice));

        free(query);
        free(dist);
        free(ind);

        // Mean ///////////////////////////////////////////////////////////////
        CUDA_CALL(cudaMemcpy(d_yvals,population.data(),samples*sizeof(float),
                cudaMemcpyHostToDevice));

        multiLocLinReg<<<noBlocks,maxThreadsPerBlock>>>(dimRes,1,dimRes,1,1,0,
                0,k,d_samples,d_xvals,d_yvals,d_surrogate,d_xmin,d_xmax,d_dist,
                d_ind);
        CUDA_CALL(cudaPeekAtLastError());
        CUDA_CALL(cudaDeviceSynchronize());

        CUDA_CALL(cudaMemcpy(surrogateF.data(),d_surrogate,2*dimRes*sizeof(
                float),cudaMemcpyDeviceToHost));

        // Compute global regression

        // Save the surrogate to the RoadGA object
        op->getSurrogateML()[2*op->getScenario()->getCurrentScenario()][
                op->getScenario()->getRun()][speciesID] = surrogateF
                .cast<double>();

        // Standard deviation /////////////////////////////////////////////////
        CUDA_CALL(cudaMemcpy(d_yvals,populationSD.data(),samples*sizeof(float),
                cudaMemcpyHostToDevice));

        multiLocLinReg<<<noBlocks,maxThreadsPerBlock>>>(dimRes,1,dimRes,1,1,0,
                0,k,d_samples,d_xvals,d_yvals,d_surrogate,d_xmin,d_xmax,d_dist,
                d_ind);
        CUDA_CALL(cudaPeekAtLastError());
        CUDA_CALL(cudaDeviceSynchronize());

        CUDA_CALL(cudaMemcpy(surrogateF.data(),d_surrogate,2*dimRes*sizeof(
                float),cudaMemcpyDeviceToHost));

        // Save the surrogate to the RoadGA object
        op->getSurrogateML()[2*op->getScenario()->getCurrentScenario()+1][
                op->getScenario()->getRun()][speciesID] = surrogateF
                .cast<double>();

        // Free remaining memory
        CUDA_CALL(cudaPeekAtLastError());
        CUDA_CALL(cudaDeviceSynchronize());

        CUDA_CALL(cudaFree(d_xmax));
        CUDA_CALL(cudaFree(d_dist));
        CUDA_CALL(cudaFree(d_ind));
        CUDA_CALL(cudaFree(d_xmin));
        CUDA_CALL(cudaFree(d_xvals));
        CUDA_CALL(cudaFree(d_yvals));
        CUDA_CALL(cudaFree(d_surrogate));
        CUDA_CALL(cudaFree(d_samples));
    } catch (const char* err) {
        throw err;
    }
}

void SimulateGPU::buildSurrogateROVCUDA(RoadGAPtr op) {

    try {
        // Get device properties
        int device = 0;
        struct cudaDeviceProp properties;
        cudaGetDeviceProperties(&properties, device);
        int maxThreadsPerBlock = properties.maxThreadsPerBlock;

        // --- CUBLAS initialization

        // Pertinent parameters
        int dimRes = op->getSurrDimRes();
        int noDims = op->getSpecies().size()+1;
        int samples = op->getNoSamples();
        // Use 5% of the nearest points for now
        // Use the 10 nearest points for now
        int k;
        if (samples < 50) {
            k = 5;
        } else if (samples < 100) {
            k = 7;
        } else if (samples < 500) {
            k = (int)ceil(0.02*(samples-100)+5);
        } else {
            k = (int)ceil(log(samples)+8);
        }

        k = min(samples,k);

        // Adjust for the number of dimensions
        k = k*pow(2,noDims-1);

        // Convert to floating point
        Eigen::VectorXf surrogateF(dimRes*noDims+pow(dimRes,noDims));
        Eigen::VectorXf predictors(noDims*samples);
        Eigen::VectorXf ref = predictors;
        Eigen::VectorXf values = op->getValues().segment(0,samples).
                cast<float>();
        Eigen::VectorXf valuesSD = op->getValuesSD().segment(0,samples)
                .cast<float>();

        // We also need to transpose the data so that the individual
        // observations are in columns
        for (int ii = 0; ii < noDims-1; ii++) {
            predictors.segment(ii*samples,samples) = op->getIARS().block(0,ii,
                    samples,1).cast<float>();
        }
        predictors.segment(samples*(noDims-1),samples) = op->getUse().segment(
                0,samples).transpose().cast<float>();

        // Call regression kernel for computing the mean and standard
        // deviation
        int noBlocks = (int)((int)pow(dimRes,noDims) % maxThreadsPerBlock) ?
                (int)(pow(dimRes,noDims)/maxThreadsPerBlock + 1) : (int)(
                pow(dimRes,noDims)/maxThreadsPerBlock);
        maxThreadsPerBlock = min((int)pow(dimRes,noDims),maxThreadsPerBlock);

        // Arrange the incoming information
        float *d_xmaxes, *d_xmins, *d_xvals, *d_yvals, *d_surrogate;
        int *d_samples;
        CUDA_CALL(cudaMalloc((void**)&d_xmaxes,noDims*sizeof(float)));
        CUDA_CALL(cudaMalloc((void**)&d_xmins,noDims*sizeof(float)));
        CUDA_CALL(cudaMalloc((void**)&d_samples,sizeof(int)));

        CUDA_CALL(cudaMalloc((void**)&d_xvals,noDims*samples*sizeof(float)));
        CUDA_CALL(cudaMalloc((void**)&d_yvals,samples*sizeof(float)));
        CUDA_CALL(cudaMemcpy(d_xvals,predictors.data(),noDims*samples*sizeof(
                float),cudaMemcpyHostToDevice));
        CUDA_CALL(cudaMemcpy(d_samples,&samples,sizeof(int),
                cudaMemcpyHostToDevice));

        CUDA_CALL(cudaMalloc((void**)&d_surrogate,(dimRes*noDims+pow(dimRes,
                noDims))*sizeof(float)));

        // Get the minimum and maximum X value for each dimension for
        // each control.
        computeStateMinMax<<<1,1>>>(1,noDims,samples,d_samples,d_xvals,d_xmins,
                d_xmaxes);
        CUDA_CALL(cudaPeekAtLastError());
        CUDA_CALL(cudaDeviceSynchronize());

        // Allocate the k nearest neighbours for each design point in
        // order to do the regression stage after this. This component is
        // the slowest component.

        // Use 5% of the nearest points for now
        // We first need to perform a k nearest neighbour search        
        Eigen::MatrixXf query((int)pow(dimRes,noDims),noDims);
        float *dist;
        int *ind;
        dist = (float*)malloc(pow(dimRes,noDims)*noDims*k*sizeof(float));
        ind = (int*)malloc(pow(dimRes,noDims)*noDims*k*sizeof(int));

        float *d_queryPts, *d_dist;
        int *d_ind;
        CUDA_CALL(cudaMalloc((void**)&d_queryPts,pow(dimRes,noDims)*noDims*
                sizeof(float)));
        CUDA_CALL(cudaMalloc((void**)&d_dist,pow(dimRes,noDims)*noDims*k*
                sizeof(float)));
        CUDA_CALL(cudaMalloc((void**)&d_ind,pow(dimRes,noDims)*noDims*k*sizeof(
                int)));

        createQueryPoints<<<noBlocks,maxThreadsPerBlock>>>((int)pow(dimRes,
                noDims),noDims,dimRes,0,1,0,d_xmins,d_xmaxes,d_surrogate,
                d_queryPts);
        CUDA_CALL(cudaPeekAtLastError());
        CUDA_CALL(cudaDeviceSynchronize());

        CUDA_CALL(cudaMemcpy(query.data(),d_queryPts,pow(dimRes,noDims)*noDims*
                sizeof(float),cudaMemcpyDeviceToHost));
        CUDA_CALL(cudaFree(d_queryPts));

        // We need to normalise in each dimension before performing the
        // regression. (We also convert the unit profit component to a log
        // scale. NOT IMPLEMENTED YET). As the profit component is not as
        // important as the population component for making immediate
        // decisions, we give it a smaller scale. We arbitrarily choose a
        // factor of 4 for now.
        Eigen::VectorXf xmins(noDims);
        Eigen::VectorXf xmaxes(noDims);

        for (int ii = 0; ii < noDims; ii++) {
            xmins(ii) = predictors.segment(ii*samples,samples).minCoeff();
            xmaxes(ii) = predictors.segment(ii*samples,samples).maxCoeff();

            predictors.segment(ii*samples,samples) = (predictors.segment(ii*
                    samples,samples).array() - xmins(ii))/(xmaxes(ii) -
                    xmins(ii));
            query.block(0,ii,pow(dimRes,noDims),1) = (query.block(0,ii,pow(
                    dimRes,noDims),1).array() - xmins(ii))/(xmaxes(ii) -
                    xmins(ii));
        }

        // Compute the knn searches
        knn_cuda_with_indexes::knn(predictors.data(),samples,query.data(),pow(
                dimRes,noDims),noDims,k,dist,ind);
        CUDA_CALL(cudaPeekAtLastError());
        CUDA_CALL(cudaDeviceSynchronize());

        CUDA_CALL(cudaMemcpy(d_dist,dist,pow(dimRes,noDims)*k*sizeof(float),
                cudaMemcpyHostToDevice));

        CUDA_CALL(cudaMemcpy(d_ind,ind,pow(dimRes,noDims)*k*sizeof(float),
                cudaMemcpyHostToDevice));

        // Due to normalisation, we need to make the minimum
        // and maximum values for each dimension 0 and 1,
        // respectively.
        float *d_xminsN, *d_xmaxesN;
        Eigen::VectorXf xminsN = Eigen::VectorXf::Zero(noDims);
        Eigen::VectorXf xmaxesN = Eigen::VectorXf::Constant(
                noDims,1);
        CUDA_CALL(cudaMalloc((void**)&d_xminsN,noDims*sizeof(
                float)));
        CUDA_CALL(cudaMalloc((void**)&d_xmaxesN,noDims*sizeof(
                float)));
        CUDA_CALL(cudaMemcpy(d_xminsN,xminsN.data(),noDims*
                sizeof(float),cudaMemcpyHostToDevice));
        CUDA_CALL(cudaMemcpy(d_xmaxesN,xmaxesN.data(),noDims*
                sizeof(float),cudaMemcpyHostToDevice));

        free(dist);
        free(ind);

        // Mean ///////////////////////////////////////////////////////////////
        CUDA_CALL(cudaMemcpy(d_yvals,values.data(),samples*sizeof(float),
                cudaMemcpyHostToDevice));

        multiLocLinReg<<<noBlocks,maxThreadsPerBlock>>>(pow(dimRes,noDims),
                noDims,dimRes,1,1,0,0,k,d_samples,d_xvals,d_yvals,d_surrogate,
                d_xminsN,d_xmaxesN,d_dist,d_ind);
        CUDA_CALL(cudaPeekAtLastError());
        CUDA_CALL(cudaDeviceSynchronize());

        // As we are using ROV, we know by definition that the operating cost/
        // benefit cannot be above zero. Therefore, we convert all positive
        // values to zero.
        rovCorrection<<<noBlocks,maxThreadsPerBlock>>>(pow(dimRes,noDims),
                noDims,dimRes,1,1,0,0,d_surrogate);

        CUDA_CALL(cudaMemcpy(surrogateF.data(),d_surrogate,(dimRes*noDims+pow(
                dimRes,noDims))*sizeof(float),cudaMemcpyDeviceToHost));

        // Save the surrogate to the RoadGA object
        op->getSurrogateML()[2*op->getScenario()->getCurrentScenario()][
                op->getScenario()->getRun()][0] = surrogateF
                .cast<double>();

        // Standard deviation /////////////////////////////////////////////////
        CUDA_CALL(cudaMemcpy(d_yvals,valuesSD.data(),samples*sizeof(float),
                cudaMemcpyHostToDevice));

        multiLocLinReg<<<noBlocks,maxThreadsPerBlock>>>(pow(dimRes,noDims),
                noDims,dimRes,1,1,0,0,k,d_samples,d_xvals,d_yvals,d_surrogate,
                d_xminsN,d_xmaxesN,d_dist,d_ind);
        CUDA_CALL(cudaPeekAtLastError());
        CUDA_CALL(cudaDeviceSynchronize());

        CUDA_CALL(cudaMemcpy(surrogateF.data(),d_surrogate,dimRes*noDims+pow(
                dimRes,noDims)*sizeof(float),cudaMemcpyDeviceToHost));

        // Save the surrogate to the RoadGA object
        op->getSurrogateML()[2*op->getScenario()->getCurrentScenario()+1][
                op->getScenario()->getRun()][0] = surrogateF.cast<double>();

        // Free remaining memory
        CUDA_CALL(cudaPeekAtLastError());
        CUDA_CALL(cudaDeviceSynchronize());

        CUDA_CALL(cudaFree(d_xmaxes));
        CUDA_CALL(cudaFree(d_xmins));
        CUDA_CALL(cudaFree(d_xmaxesN));
        CUDA_CALL(cudaFree(d_xminsN));
        CUDA_CALL(cudaFree(d_ind));
        CUDA_CALL(cudaFree(d_dist));
        CUDA_CALL(cudaFree(d_xvals));
        CUDA_CALL(cudaFree(d_yvals));
        CUDA_CALL(cudaFree(d_surrogate));
        CUDA_CALL(cudaFree(d_samples));
    } catch (const char* err) {
        throw err;
    }
}

void SimulateGPU::interpolateSurrogateMulti(Eigen::VectorXd& surrogate,
        Eigen::VectorXd &predictors, Eigen::VectorXd &results, int dimRes,
        int noDims) {

    try {
        // Get device properties
        int device = 0;
        struct cudaDeviceProp properties;
        cudaGetDeviceProperties(&properties, device);
        int maxThreadsPerBlock = properties.maxThreadsPerBlock;

        int noBlocks = (int)((int)results.size() % maxThreadsPerBlock) ?
                (int)((int)results.size()/maxThreadsPerBlock + 1) : (int)((int)
                results.size()/maxThreadsPerBlock);
        maxThreadsPerBlock = min((int)results.size(),maxThreadsPerBlock);

        // Assign memory to the GPU
        Eigen::VectorXf surrogateF = surrogate.cast<float>();
        Eigen::VectorXf predictorsF = predictors.cast<float>();
        Eigen::VectorXf resultsF = results.cast<float>();

        float *d_surrogate, *d_predictors, *d_results;

        CUDA_CALL(cudaMalloc((void**)&d_surrogate,surrogate.size()*sizeof(float)));
        CUDA_CALL(cudaMalloc((void**)&d_predictors,predictors.size()*sizeof(
                float)));
        CUDA_CALL(cudaMalloc((void**)&d_results,results.size()*sizeof(float)));

        CUDA_CALL(cudaMemcpy(d_surrogate,surrogateF.data(),surrogateF.size()*
                sizeof(float),cudaMemcpyHostToDevice));
        CUDA_CALL(cudaMemcpy(d_predictors,predictorsF.data(),predictorsF.size()*
                sizeof(float),cudaMemcpyHostToDevice));

        interpolateMulti<<<noBlocks,maxThreadsPerBlock>>>(results.size(),noDims,
                dimRes,d_surrogate,d_predictors,d_results);
        CUDA_CALL(cudaPeekAtLastError());
        CUDA_CALL(cudaDeviceSynchronize());

        CUDA_CALL(cudaMemcpy(resultsF.data(),d_results,results.size()*sizeof(
                float),cudaMemcpyDeviceToHost));

        CUDA_CALL(cudaFree(d_surrogate));
        CUDA_CALL(cudaFree(d_predictors));
        CUDA_CALL(cudaFree(d_results));

        results = resultsF.cast<double>();
    } catch (const char* err) {
        throw err;
    }
}

// HELPER ROUTINES ////////////////////////////////////////////////////////////

// Conversion of dense matrix to sparse for movement and mortality
void SimulateGPU::dense2Sparse(float* denseIn, int rows, int cols,
        float* sparseOut, int* elemsPerCol, int* rowIdx, int& totalElements) {

    // We work by column as this is the order in which we will later use the
    // sparse matrix.
    int it_1 =0;

    for (int ii = 0; ii < cols; ii++) {
        int it_2 =0;

        for (int jj = 0; jj < rows; jj++) {
            // To account for system rounding, we use a small tolerance
            if (denseIn[ii*rows + jj] > 0.00001) {
                sparseOut[it_1] = denseIn[ii*rows + jj];
                rowIdx[it_1] = jj;
                it_1++;
                it_2++;
            }
        }
        elemsPerCol[ii] = it_2;
    }

    totalElements = it_1;
}

// Multiple global linear regression
void SimulateGPU::multiLinReg(int noPoints, int noDims, float *xvals, float
        *yvals, float *X) {

    float *A, *B;
    A = (float*)malloc(pow(noDims+1,2)*sizeof(float));
    B = (float*)malloc((noDims+1)*sizeof(float));

    for (int ii = 0; ii <= noDims; ii++) {
        // Initialise values to zero
        B[ii] = 0.0;

        for (int jj = 0; jj < noPoints; jj++) {
            if (ii == 0) {
                B[ii] += yvals[jj];
            } else {
                B[ii] += yvals[jj]*xvals[jj*noDims+ii-1];
            }
        }

        for (int jj = 0; jj <= noDims; jj++) {
            A[jj*(noDims+1)+ii] = 0.0;

            for (int kk = 0; kk < noPoints; kk++) {

                if ((ii == 0) && (jj == 0)) {
                    A[jj*(noDims+1)+ii] += 1.0;
                } else if (ii == 0) {
                    A[jj*(noDims+1)+ii] += (xvals[kk*noDims+ii-1]);
                } else if (jj == 0) {
                    A[jj*(noDims+1)+ii] += (xvals[kk*noDims+jj-1]);
                } else {
                    A[jj*(noDims+1)+ii] += (xvals[kk*noDims+ii-1])*
                            (xvals[kk*noDims+jj-1]);
                }
            }
        }
    }

    SimulateGPU::solveLinearSystem(noDims+1,A,B,X);

    free(A);
    free(B);
}

// Solve a set of linear equations. (Assumes non-singular coefficient matrix)
void SimulateGPU::solveLinearSystem(int dims, float *A, float *B, float *C) {
    // First generate upper triangular matrix for the augmented matrix
    float *swapRow;
    swapRow = (float*)malloc((dims+1)*sizeof(float));

    for (int ii = 0; ii < dims; ii++) {
        C[ii] = B[ii];
    }

    for (int ii = 0; ii < dims; ii++) {
        // Search for maximum in this column
        float maxElem = fabsf(A[ii*dims+ii]);
        int maxRow = ii;

        for (int jj = (ii+1); jj < dims; jj++) {
            if (fabsf(A[ii*dims+jj] > maxElem)) {
                maxElem = fabsf(A[ii*dims+jj]);
                maxRow = jj;
            }
        }

        // Swap maximum row with current row if needed
        if (maxRow != ii) {
            for (int jj = ii; jj < dims; jj++) {
                swapRow[jj] = A[jj*dims+ii];
                A[jj*dims+ii] = A[jj*dims+maxRow];
                A[jj*dims+maxRow] = swapRow[jj];
            }

            swapRow[dims] = C[ii];
            C[ii] = C[maxRow];
            C[maxRow] = swapRow[dims];
        }

        // Make all rows below this one 0 in current column
        for (int jj = (ii+1); jj < dims; jj++) {
            float factor = -A[ii*dims+jj]/A[ii*dims+ii];

            // Work across columns
            for (int kk = ii; kk < dims; kk++) {
                if (kk == ii) {
                    A[kk*dims+jj] = 0.0;
                } else {
                    A[kk*dims+jj] += factor*A[kk*dims+ii];
                }
            }

            // Results vector
            C[jj] += factor*C[ii];
        }
    }
    free(swapRow);

    // Solve equation for an upper triangular matrix
    for (int ii = dims-1; ii >= 0; ii--) {
        C[ii] = C[ii]/A[ii*dims+ii];

        for (int jj = ii-1; jj >= 0; jj--) {
            C[jj] -= C[ii]*A[ii*dims+jj];
        }
    }
}
