#include <stdio.h>
#include <cuda.h>
#include <curand.h>
#include <curand_kernel.h>
//#include <cusolverDn.h>
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include "../transportbase.h"
#include "knn_cublas_with_indexes.h"

// ERROR CHECKING MACROS //////////////////////////////////////////////////////

#define CUDA_CALL(x) { gpuAssert((x), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort
        =true)
{
    if (code != cudaSuccess) {
        fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file,
                line);
        cudaDeviceReset();

//        if (abort) throw std::exception();
    }
}

static const int max_shared_floats = 8000;

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
        // Get large grid cell subscripts of thread
        int blockIdxY = (int)(((int)(idx/uniqueRegions))/xres);
        int blockIdxX = (int)(idx/uniqueRegions) - blockIdxY*xres;
        // Valid region numbering starts at 1, not 0
        int regionNo = idx - blockIdxY*xres*uniqueRegions - blockIdxX*
                uniqueRegions + 1;

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
                int subIdx = blockIdxY*xres*skpx*skpy + blockIdxX*skpx
                        + jj*H + ii;
                area += (float)(labelledImage[subIdx] == regionNo);
            }
        }

        if (area > 0) {
            for (int ii = 0; ii < blockSizeX; ii++) {
                for (int jj = 0; jj < blockSizeY; jj++) {
                    int subIdx = blockIdxY*xres*skpx*skpy + blockIdxX*skpx
                            + jj*H + ii;
                    pop += pops[subIdx];
                    cx += ii*(float)(labelledImage[subIdx] == regionNo);
                    cy += jj*(float)(labelledImage[subIdx] == regionNo);
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

//        if (results[5*idx] > 0) {
//            printf("%4d, %5d, %8.0f, %5.0f, %5.0f, %5.0f, %5.0f\n",idx,blockSizeX,
//                    results[5*idx],results[5*idx+1],results[5*idx+2],
//                    results[5*idx+3],results[5*idx+4]);
//        }
    }
}

//// Computes the movement and mortality of a species from the forward path
//// kernels
//__global__ void mmKernel(float* popsIn, float* popsOut, float* mmm, int patches) {
//    int ii = threadIdx.x;

//    if (ii < patches) {
//        extern __shared__ float s[];

//        s[ii] = 0.0;

//        for (int jj = 0; jj < patches; jj++) {
//            s[ii] += popsIn[ii]*mmm[ii*patches + jj];
//        }
//        __syncthreads();

//        popsOut[ii] = s[ii];
//    }
//}

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
__global__ void randControls(int noPaths, int noControls, float* randCont,
        int* control) {
    int idx = blockIdx.x*blockDim.x + threadIdx.x;

    if (idx < noPaths) {
        control[idx] = (int)((randCont[idx])/(float)noControls);
    }
}

// The kernel for computing forward paths in ROV. This routine considers
// each patch as containing a certain number of each species.
__global__ void forwardPathKernel(int start, int noPaths, int nYears, int
        noSpecies, int noPatches, int noControls, int noFuels, int
        noUncertainties, float timeStep, float* pops, float* transitions,
        float* survival, float* speciesParams, float* caps, float* aars,
        float* uncertParams, int* controls, float* uJumps, float* uBrownian,
        float* uJumpSizes, float* uJumpsSpecies, float* uBrownianSpecies,
        float* uJumpSizesSpecies, float* rgr, float* uResults, float*
        totalPops) {

    // Global thread index
    int idx = blockIdx.x*blockDim.x + threadIdx.x;

    // Carry over the initial value for all uncertainties
    for (int ii = 0; ii < noUncertainties; ii++) {
        uResults[idx*noUncertainties*nYears + ii] = uncertParams[ii*6];
    }

    // Only perform matrix multiplication sequentially for now. Later, if so
    // desired, we can use dynamic parallelism because the card in the
    // machine has CUDA compute compatability 3.5

    if (idx < noPaths) {
        float* grMean;
        grMean = (float*)malloc(noSpecies*sizeof(float));

        for (int jj = 0; jj < noSpecies; jj++) {
            grMean[jj] = speciesParams[jj*8];
        }

        for (int ii = start; ii <= nYears; ii++) {
            // Control to pick
            int control = controls[idx*nYears + ii];

            for (int jj = 0; jj < noSpecies; jj++) {
// I think this code is unnecessary
//                // Determine the aar under each control
//                float* tempPops;
//                tempPops = malloc(noControls*sizeof(float));

//                for (int kk = 0; kk < noControls; kk++) {
//                    tempPop[kk] = 0;

//                    for (int ll = 0; ll < noPatches; ll++) {
//                        for (int mm = 0; mm < noPatches; mm++) {
//                            tempPop[kk] += pops[idx*nYears*noSpecies*
//                                    noPatches + (ii-1)*noSpecies*noPatches +
//                                    jj*noPatches + mm]*transitions[jj*noPatches
//                                    *noPatches + ll*noPatches + mm]*survival[
//                                    jj*noPatches*noPatches*noControls +
//                                    kk*noPatches*noPatches + ll*noPatches +mm];
//                        }
//                    }
//                }

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

                for (int kk = 0; kk < noControls; kk++) {
                    aars[jj*nYears*noPaths*noControls + ii*noPaths*
                            noControls + idx*noControls + kk] = 0;
                }

                float population = 0;

                for (int kk = 0; kk < noPatches; kk++) {
                    for (int ll = 0; ll < noPatches; ll++) {

                        float value = pops[idx*nYears*noSpecies*
                                noPatches + (ii-1)*noSpecies*noPatches
                                + jj*noPatches + ll]*transitions[jj*
                                noPatches*noPatches + kk*noPatches +
                                ll];

                        population += value;

                        for (int mm = 0; mm < noControls; mm++) {
                            float valCont = value*survival[jj*noPatches*
                                    noPatches*noControls + kk*noPatches*
                                    noPatches*control + kk*noPatches + ll];

                            if (mm == control) {
                                // Movement and mortality
                                pops[idx*nYears*noSpecies*noPatches
                                        + ii*noSpecies*noPatches + jj*noPatches
                                        + kk] += valCont;
                            } else {
                                aars[jj*nYears*noPaths*noControls + ii*noPaths*
                                        noControls + idx*noControls + mm] +=
                                        valCont;
                            }
                        }
                    }
                    // Population growth based on a mean-reverting process
                    rgr[idx*noSpecies*noPatches*nYears + ii*noSpecies*noPatches
                            + jj*noPatches + kk] = grMean[jj] + rgr[idx*
                            noSpecies*noPatches*nYears + ii*noSpecies*noPatches
                            + jj*noPatches + kk]*speciesParams[jj*8 + 7];

                    float gr = rgr[idx*noSpecies*noPatches*nYears + ii*
                            noSpecies*noPatches + jj*noPatches + kk];

                    pops[idx*nYears*noSpecies*noPatches +
                            ii*noSpecies*noPatches + jj*noPatches + kk] =
                            pops[idx*nYears*noSpecies*noPatches
                            + ii*noSpecies*noPatches + jj*noPatches + kk]*(1.0f
                            + gr*(caps[jj*noPatches + kk] - pops[idx*nYears*
                            noSpecies*noPatches + ii*noSpecies*noPatches +
                            jj*noPatches + kk])/caps[jj*noPatches + kk]/100.0);

                    totalPops[ii*noSpecies*noPaths + idx*noSpecies + jj] +=
                            pops[idx*nYears*noSpecies*noPatches + ii*noSpecies*
                            noPatches + jj*noPatches + kk];
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

__global__ void computePathStates(int noPaths, int noDims, int nYears, int
        noControls, int year, float unitCost, float unitRevenue, int* controls,
        int noFuels, float *fuelCosts, float *uResults, float *uComposition,
        int noUncertainties, int *fuelIdx, int noCommodities, float* aars,
        float* totalPops, float* xin) {

    // Global thread index
    int idx = blockIdx.x*blockDim.x + threadIdx.x;

    if (idx < noPaths) {
        // 1. Adjusted population for each species
        for (int ii = 0; ii < noDims-1; ii++) {
            xin[idx*noPaths + ii] = totalPops[year*(noDims-1)*noPaths + idx*
                    (noDims-1) + ii]*aars[ii*nYears*noPaths*noControls + idx*
                    noControls + controls[year*noPaths + idx]];
        }

        // 2. Unit profit
        float unitFuel = 0.0;
        float orePrice = 0.0;

        // Compute the unit fuel cost component
        for (int ii = 0; ii < noFuels; ii++) {
            unitFuel += fuelCosts[ii]*uResults[idx*nYears*
                    noUncertainties + (year+1)*noUncertainties +
                    fuelIdx[ii]];
        }
        // Compute the unit revenue from ore
        for (int ii = 0; ii < noCommodities; ii++) {
            orePrice += uComposition[idx*nYears*noCommodities + (year+1)*
                    noCommodities + ii]*uResults[idx*nYears*noUncertainties +
                    (year+1)*noUncertainties + noFuels + ii];
        }

        xin[idx*noPaths + noDims] = unitCost + unitFuel - unitRevenue*orePrice;
    }
}

__global__ void allocateXYRegressionData(int noPaths, int noControls, int
        noDims, int year, int* controls, float* xin, float *condExp, int
        *dataPoints, float *xvals, float *yvals) {

    for (int ii = 0; ii < noControls; ii++) {
        dataPoints[ii] = 0;
    }

    // For each path
    for (int ii = 0; ii < noPaths; ii++) {
        yvals[noPaths + controls[ii]*dataPoints[controls[ii]]] = condExp[
                year*noPaths* + ii];

        // Save the input dimension values to the corresponding data group
        for (int jj = 0; jj < noDims; jj++) {
            xvals[dataPoints[controls[ii]]*noDims + controls[ii]*noPaths*
                    noDims + jj] = xin[ii*noDims + jj];
        }

        // Increment the number of data points for this control
        dataPoints[controls[ii]]++;
    }
}

__global__ void computeStateMinMax(int noControls, int noDims, int noPaths,
        int* dataPoints, float* xvals, float* xmins, float* xmaxes) {

    for (int ii = 0; ii < noControls; ii++) {
        float *xmin, *xmax;
        xmin = (float*)malloc(noDims*sizeof(float));
        xmax = (float*)malloc(noDims*sizeof(float));

        for (int jj = 0; jj < noDims; jj++) {
            xmin[jj] = xvals[ii*noDims*noPaths + jj];
            xmax[jj] = xmin[jj];
        }

        for (int jj = 0; jj < dataPoints[ii]; jj++) {
            for (int kk = 0; kk < noDims; kk ++) {
                float xtemp = xvals[ii*noDims*noPaths + jj*noDims + kk];
                if (xmin[jj] > xtemp) {
                    xmin[jj] = xtemp;
                } else if (xmax[jj] < xtemp) {
                    xmax[jj] = xtemp;
                }
            }
        }

        for (int jj = 0; jj < noDims; jj++) {
            xmins[ii*noDims + jj] = xmin[jj];
            xmaxes[ii*noDims + jj] = xmax[jj];
        }

        free(xmin);
        free(xmax);
    }
}

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
            dimIdx[noDims - ii - 1] = div;
            rem = rem - div*pow(dimRes,noDims-ii-1);
        }

        // Get the query point coordinates
        for (int ii = 0; ii < noDims; ii++) {
            queryPts[idx*noDims + ii] = ((float)dimIdx[ii]+0.5)*(xmaxes[control
                    *noDims + ii] - xmins[control*noDims + ii])/(float)dimRes +
                    xmins[control*noDims + ii];

            // Save the X value for the query point
            regression[year*noControls*(dimRes*noDims + (int)pow(dimRes,
                    noDims)*2) + control*(dimRes*noDims + (int)pow(dimRes,
                    noDims)*2) + ii*dimRes + dimIdx[ii]] = queryPts[idx*noDims
                    + ii];
        }

        free(dimIdx);
    }
}

__global__ void optimalForwardPaths(int start, int noPaths, int nYears, int
        noSpecies, int noPatches, int noControls, int noUncertainties, float
        timeStep, float unitCost, float unitRevenue, float rrr, int noFuels,
        int noCommodities, int dimRes, float* Q, float* fuelCosts, float* pops,
        float* totalPops, float* transitions, float* survival, float*
        speciesParams, float* rgr, float* caps, float* aars, float* regression,
        float* uComposition, float* uResults, int* fuelIdx, float* condExp,
        int* optCont, float* adjPops, float*unitProfits) {

    // Global thread index
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
            unitFuel += fuelCosts[ii]*uResults[idx*nYears*
                    noUncertainties + (start+1)*noUncertainties +
                    fuelIdx[ii]];
        }
        // Compute the unit revenue from ore
        for (int ii = 0; ii < noCommodities; ii++) {
            orePrice += uComposition[idx*nYears*noCommodities + (start+1)*
                    noCommodities + ii]*uResults[idx*nYears*noUncertainties +
                    (start+1)*noUncertainties + noFuels + ii];
        }

        if (start == nYears) {
            // At the last period we run the road if the adjusted population
            // for a particular control (pop-pop*aar_jj) is greater than the
            // minimum permissible population. This becomes the optimal
            // payoff, which we then regress onto the adjusted population to
            // determine the expected payoff at time T given a prevailing end
            // adjusted population. Everything is considered deterministic at
            // this stage. Therefore, we do not need to compute this section.
            for (int ii = 0; ii < noControls; ii++) {
                // Compute the single period financial payoff for each control
                // for this period and the adjusted profit. If any adjusted
                // population is below the threshold, then the payoff is
                // invalid.
                valid[ii] = true;
                for (int jj = 0; jj < noSpecies; jj++) {
                    float adjPop = totalPops[start*noSpecies*noPaths + idx*
                            noSpecies + jj]*aars[jj*nYears*noPaths*noControls
                            + idx*noControls + ii];

                    if (adjPop < speciesParams[noSpecies*jj + 2]) {
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
            condExp[idx+nYears*noPaths] = payoffs[0];
            optCont[idx+nYears*noPaths] = 0;

            for (int ii = 1; ii < noControls; ii++) {
                if (isfinite(payoffs[ii])) {
                    if (payoffs[ii] < condExp[idx+(nYears+1*noPaths)]) {
                        condExp[idx+nYears*noPaths] = payoffs[ii];
                        optCont[idx+nYears*noPaths] = ii;
                    }
                }
            }

            // The states are the adjusted populations per unit traffic for
            // each species and the current period unit profit. We use the
            // aar of the very last control to compute this.
            for (int ii = 0; ii < noSpecies; ii++) {
                adjPops[ii*noPaths+idx] = totalPops[start*noSpecies*noPaths
                        + idx*noSpecies + ii]*aars[ii*nYears*noPaths*noControls
                        + (idx+1)*noControls - 1]/Q[noControls - 1];
            }

            // The previailing unit profit
            unitProfits[start*noPaths + idx] = unitCost + unitFuel -
                    unitRevenue*orePrice;

        } else {
            // For all other time periods, we need to recompute the forward
            // paths and add the present values of the expected payoffs to the
            // current period payoff using the regression functions that were
            // computed outside of this kernel.

            // As the original points were developed with linear regression,
            // we use linear interpolation as a reasonable approximation.
            // Furthermore, speed is an issue, so we need a faster approach
            // than a more accurate one such as cubic spline interpolation.

            // Find the current state through multilinear interpolation
            float *state;
            state = (float*)malloc((noSpecies+1)*sizeof(float));

            // We first determine the current state. This consists of the
            // ajdusted population of each species and the current unit
            // profit.
            for (int ii = 0; ii <noSpecies; ii++) {
                state[ii] = totalPops[start*noPaths + idx]*aars[start*noPaths
                        + idx];
            }
            // 2. Unit profit
            unitFuel = 0.0;
            orePrice = 0.0;

            // Compute the unit fuel cost component
            for (int ii = 0; ii < noFuels; ii++) {
                unitFuel += fuelCosts[ii]*uResults[idx*nYears*
                        noUncertainties + (start+1)*noUncertainties +
                        fuelIdx[ii]];
            }
            // Compute the unit revenue from ore
            for (int ii = 0; ii < noCommodities; ii++) {
                orePrice += uComposition[idx*nYears*noCommodities + (start+1)*
                        noCommodities + ii]*uResults[idx*nYears*noUncertainties +
                        (start+1)*noUncertainties + noFuels + ii];
            }
            state[noSpecies] = unitCost + unitFuel - unitRevenue*orePrice;

            // Determine the current period payoffs to select the optimal
            // control for this period.
            for (int ii = 0; ii < noControls; ii++) {
                // Compute the single period financial payoff for each control
                // for this period and the adjusted profit. If any adjusted
                // population is below the threshold, then the payoff is
                // invalid.
                valid[ii] = true;
                for (int jj = 0; jj < noSpecies; jj++) {
                    float adjPop = totalPops[start*noSpecies*noPaths + idx*
                            noSpecies + jj]*aars[jj*nYears*noPaths*noControls
                            + idx*noControls + ii];

                    if (adjPop < speciesParams[noSpecies*jj + 2]) {
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
                    currPayoffs[ii] = Q[ii]*(unitCost + unitFuel - unitRevenue*
                            orePrice);

                    // First find global the upper and lower bounds in each
                    // dimension as well as the index of the lower bound of the
                    // regressed value in each dimension.
                    float *lower, *upper, *coeffs;
                    int *lowerInd;
                    lower = (float*)malloc((noSpecies+1)*sizeof(float));
                    upper = (float*)malloc((noSpecies+1)*sizeof(float));
                    coeffs = (float*)malloc(((int)pow(2,noSpecies))*
                            sizeof(float));
                    lowerInd = (int*)malloc((noSpecies+1)*sizeof(float));

                    for (int jj = 0; jj <= noSpecies; jj++) {
                        lower[jj] = regression[start*noControls*(dimRes*(
                                noSpecies+1) + (int)pow(dimRes,noSpecies+1)*2)
                                + ii*(dimRes*(noSpecies+1) + (int)pow(dimRes,
                                (noSpecies+1))*2) + jj*dimRes];
                        upper[jj] = regression[start*noControls*(dimRes*(
                                noSpecies+1) + (int)pow(dimRes,noSpecies+1)*2)
                                + ii*(dimRes*(noSpecies+1) + (int)pow(dimRes,
                                (noSpecies+1))*2) + (jj+1)*dimRes - 1];

                        lowerInd[jj] = (int)dimRes*(state[jj] - lower[jj])/(
                                upper[jj] - lower[jj]);
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
                    float xd = (state[0] - x0)/(x1-x0);

                    // First, assign the yvalues to the coefficients matrix
                    for (int jj = 0; jj < (int)pow(2,noSpecies); jj++) {
                        // Get the indices of the yvalues of the lower and upper
                        // bounding values on this dimension.
                        int idxL = start*noControls*(dimRes*(noSpecies + 1) +
                                (int)pow(dimRes,(noSpecies+1))*2) + ii*(dimRes*
                                (noSpecies + 1) + (int)pow(dimRes,(noSpecies+1))
                                *2) + dimRes*(noSpecies + 1);

                        for (int kk = 1; kk <= noSpecies; kk++) {
                            int rem = ((int)(jj/((int)pow(2,noSpecies - kk))) +
                                    1) - 2*(int)(((int)(jj/((int)pow(2,
                                    noSpecies - kk))) + 1)/2);
                            if (rem > 0) {
                                idxL += lowerInd[kk]*(int)pow(dimRes,noSpecies
                                        - kk);
                            } else {
                                idxL += (lowerInd[kk]+1)*(int)pow(dimRes,noSpecies
                                        - kk);
                            }
                        }

                        int idxU = idxL + (lowerInd[0]+1)*(int)pow(dimRes,
                                noSpecies);

                        idxL += idxL + lowerInd[0]*(int)pow(dimRes,noSpecies);

                        coeffs[jj] = regression[idxL]*(1 - xd) +
                                regression[idxU]*xd;
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
                        xd = (state[jj] - x0)/(x1-x0);

                        for (int kk = 0; kk < (int)pow(2,jj); kk++) {
                            int jump = (int)pow(2,noSpecies-jj-1);
                            coeffs[kk] = coeffs[kk]*(1 - xd) + coeffs[kk+jump]
                                    *xd;
                        }
                    }

                    payoffs[ii] = currPayoffs[ii] + coeffs[0];

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
            condExp[idx+nYears*noPaths] = currPayoffs[0];
            optCont[idx+nYears*noPaths] = 0;

            for (int ii = 1; ii < noControls; ii++) {
                if (isfinite(payoffs[ii])) {
                    if (payoffs[ii] < condExp[idx+(nYears+1*noPaths)]) {
                        condExp[idx+nYears*noPaths] = currPayoffs[ii];
                        optCont[idx+nYears*noPaths] = ii;
                    }
                }
            }

            // Now recompute the optimal forward path and add the discounted
            // optimal payoff at each period to this path's conditional
            // expectation.

            for (int ii = start+1; ii <= nYears; ii++) {
                // We must keep track of the population(s) over time as well as
                // the optimal choice taken. This means computing the current
                // state. Update the population given the optimal control at the
                // previous stage.
                int control = optCont[idx+(ii-1)*noPaths];

                for (int jj = 0; jj < noSpecies; jj++) {
                    for (int kk = 0; kk < noPatches; kk++) {
                        for (int ll = 0; ll < noPatches; ll++) {

                            float value = pops[idx*nYears*noSpecies*
                                    noPatches + (ii-1)*noSpecies*noPatches
                                    + jj*noPatches + ll]*transitions[jj*
                                    noPatches*noPatches + kk*noPatches +
                                    ll];

                            for (int mm = 0; mm < noControls; mm++) {
                                float valCont = value*survival[jj*noPatches*
                                        noPatches*noControls + kk*noPatches*
                                        noPatches*control + kk*noPatches + ll];

                                if (mm == control) {
                                    // Movement and mortality
                                    pops[idx*nYears*noSpecies*noPatches
                                            + ii*noSpecies*noPatches + jj*
                                            noPatches + kk] += valCont;
                                } else {
                                    aars[jj*nYears*noPaths*noControls + ii*
                                            noPaths*noControls + idx*noControls
                                            + mm] += valCont;
                                }
                            }
                        }

                        // Population growth
                        float gr = speciesParams[jj*3]*rgr[idx*noSpecies*
                                noPatches*nYears + ii*noSpecies*noPatches + jj*
                                noPatches + kk] + speciesParams[jj*3+1];
                        pops[idx*nYears*noSpecies*noPatches +
                                ii*noSpecies*noPatches + jj*noPatches + kk] =
                                pops[idx*nYears*noSpecies*noPatches
                                + ii*noSpecies*noPatches + jj*noPatches + kk]*(
                                1.0f + gr*(caps[jj*noPatches + kk] - pops[idx*
                                nYears*noSpecies*noPatches + ii*noSpecies*
                                noPatches + jj*noPatches + kk])/caps[jj*
                                noPatches + kk]/100.0);

                        totalPops[ii*noSpecies*noPaths + idx*noSpecies + jj] +=
                                pops[idx*nYears*noSpecies*noPatches + ii*
                                noSpecies*noPatches + jj*noPatches + kk];
                    }
                }

                ///////////////////////////////////////////////////////////////
                // Now, as before, compute the current state and the optimal
                // control to pick using the regressions. /////////////////////
                for (int jj = 0; jj <noSpecies; jj++) {
                    state[jj] = totalPops[ii*noPaths + idx]*aars[ii*noPaths
                            + idx];
                }

                unitFuel = 0.0;
                orePrice = 0.0;

                // Compute the unit fuel cost component
                for (int jj = 0; jj < noFuels; jj++) {
                    unitFuel += fuelCosts[jj]*uResults[idx*nYears*
                            noUncertainties + (ii+1)*noUncertainties +
                            fuelIdx[jj]];
                }
                // Compute the unit revenue from ore
                for (int jj = 0; jj < noCommodities; jj++) {
                    orePrice += uComposition[idx*nYears*noCommodities +
                            (ii+1)*noCommodities + jj]*uResults[idx*nYears*
                            noUncertainties + (ii+1)*noUncertainties +
                            noFuels + jj];
                }
                state[noSpecies] = unitCost + unitFuel - unitRevenue*orePrice;

                // Determine the current period payoffs to select the optimal
                // control for this period.
                for (int jj = 0; jj < noControls; jj++) {
                    // Compute the single period financial payoff for each control
                    // for this period and the adjusted profit. If any adjusted
                    // population is below the threshold, then the payoff is
                    // invalid.
                    valid[jj] = true;
                    for (int kk = 0; kk < noSpecies; kk++) {
                        float adjPop = totalPops[ii*noSpecies*noPaths + idx*
                                noSpecies + kk]*aars[kk*nYears*noPaths*noControls
                                + idx*noControls + jj];

                        if (adjPop < speciesParams[noSpecies*kk + 2]) {
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
                    if (valid[jj]) {
                        // Now compute the overall period profit for this
                        // control given the prevailing stochastic factors
                        // (undiscounted).
                        currPayoffs[jj] = Q[jj]*(unitCost + unitFuel -
                                unitRevenue*orePrice);

                        // First find global the upper and lower bounds in each
                        // dimension as well as the indices of the
                        float *lower, *upper, *coeffs;
                        int *lowerInd;
                        lower = (float*)malloc((noSpecies+1)*sizeof(float));
                        upper = (float*)malloc((noSpecies+1)*sizeof(float));
                        coeffs = (float*)malloc(((int)pow(2,noSpecies))*
                                sizeof(float));
                        lowerInd = (int*)malloc((noSpecies+1)*sizeof(float));

                        for (int kk = 0; kk <= noSpecies; kk++) {
                            lower[kk] = regression[ii*noControls*(dimRes*(
                                    noSpecies+1) + (int)pow(dimRes,noSpecies+1)
                                    *2) + jj*(dimRes*(noSpecies+1) + (int)pow(
                                    dimRes,(noSpecies+1))*2) + kk*dimRes];
                            upper[kk] = regression[ii*noControls*(dimRes*(
                                    noSpecies+1) + (int)pow(dimRes,noSpecies+1)
                                    *2) + jj*(dimRes*(noSpecies+1) + (int)pow(
                                    dimRes,(noSpecies+1))*2) + (kk+1)*dimRes - 1];

                            lowerInd[kk] = (int)dimRes*(state[kk] - lower[kk])/(
                                    upper[kk] - lower[kk]);
                        }

                        // Now that we have all the index requirements, let's
                        // interpolate.
                        // Get the uppermost dimension x value
                        float x0 = regression[ii*noControls*(dimRes*(noSpecies
                                + 1) + (int)pow(dimRes,noSpecies+1)*2) + jj*(
                                dimRes*(noSpecies+1) + (int)pow(dimRes,
                                (noSpecies+1))*2) + lowerInd[0]];
                        float x1 = regression[ii*noControls*(dimRes*(noSpecies
                                + 1) + (int)pow(dimRes,noSpecies+1)*2) + jj*(
                                dimRes*(noSpecies+1) + (int)pow(dimRes,
                                (noSpecies+1))*2) + lowerInd[0] + 1];
                        float xd = (state[0] - x0)/(x1-x0);

                        // First, assign the yvalues to the coefficients matrix
                        for (int kk = 0; kk < (int)pow(2,noSpecies); kk++) {
                            // Get the indices of the yvales of the lower and
                            // upper bounding values on this dimension.
                            int idxL = ii*noControls*(dimRes*(noSpecies + 1) +
                                    (int)pow(dimRes,(noSpecies+1))*2) + jj*(
                                    dimRes*(noSpecies + 1) + (int)pow(dimRes,
                                    (noSpecies+1))*2) + dimRes*(noSpecies + 1);

                            for (int ll = 1; ll <= noSpecies; ll++) {
                                int rem = ((int)(kk/((int)pow(2,noSpecies -
                                        ll))) + 1) - 2*(int)(((int)(kk/((int)
                                        pow(2,noSpecies - ll))) + 1)/2);
                                if (rem > 0) {
                                    idxL += lowerInd[ll]*(int)pow(dimRes,
                                            noSpecies - ll);
                                } else {
                                    idxL += (lowerInd[ll] + 1)*(int)pow(dimRes,
                                            noSpecies - ll)*2;
                                }
                            }

                            int idxU = idxL + (lowerInd[0] + 1)*(int)pow(
                                    dimRes,noSpecies);

                            idxL += idxL + lowerInd[0]*(int)pow(dimRes,
                                    noSpecies);

                            coeffs[kk] = regression[idxL]*(1 - xd) +
                                    regression[idxU]*xd;
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
                            xd = (state[kk] - x0)/(x1-x0);

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
                optCont[idx+ii*noPaths] = 0;

                for (int jj = 1; jj < noControls; jj++) {
                    if (isfinite(payoffs[jj])) {
                        if (payoffs[jj] < currMax) {
                            currMax = currPayoffs[jj];
                            optCont[idx+ii*noPaths] = jj;
                        }
                    }
                }

                // Now add the discounted cash flow for the current period for
                // the control with the optimal payoff to the retained values
                // for the optimal path value at this time step.
                condExp[idx+ii*noPaths] += currMax/(pow(1+rrr,ii-start));
            }
        }
        // We don't need to keep the optimal control at this stage but can
        // easily store it later if we wish.

        // Free memory
        free(payoffs);
        free(valid);
    }
}

// Multiple global linear regression
__global__ void multiLinReg() {

}

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

// Multiple local linear regression
__global__ void multiLocLinReg(int noPoints, int noDims, int dimRes, int nYears,
        int noControls, int year, int control, int k, int* dataPoints, float
        *xvals, float *yvals, float *d_regression, float* xmins, float* xmaxes,
        float *dist, int *ind, cublasHandle_t* handles) {

    // Global thread index
    int idx = blockIdx.x*blockDim.x + threadIdx.x;

    // First, deconstruct the index into the index along each dimension
    int *dimIdx;
    dimIdx = (int*)malloc(noDims*sizeof(int));

    int rem = idx;

    for (int ii = 0; ii < noDims; ii++) {
        int div = (int)(rem/pow(dimRes,noDims-ii-1));
        dimIdx[noDims - ii - 1] = div;
        rem = rem - div*pow(dimRes,noDims-ii-1);
    }

    if (idx < noPoints) {
        // Get the query point coordinates
        float *xQ;
        xQ = (float*)malloc(noDims*sizeof(float));

        for (int ii = 0; ii < noDims; ii++) {
            xQ[ii] = ((float)dimIdx[ii]+0.5)*(xmaxes[control*noDims + ii] -
                    xmins[control*noDims + ii])/(float)dimRes +
                    xmins[control*noDims + ii];
        }

        // 1. First find the k nearest neighbours to the query point (already)
        // computed prior).

        // 2. Build the matrices used in the calculation
        // A - Input design matrix
        // B - Input known matrix
        // C - Output matrix of coefficients
        float *A, *B, *C, *X;

        A = (float*)malloc(pow(noDims+1,2)*sizeof(float));
        B = (float*)malloc((noDims+1)*sizeof(float));
        C = (float*)malloc(pow(noDims+1,2)*sizeof(float));
        X = (float*)malloc((noDims+1)*sizeof(float));

        for (int ii = 0; ii <= noDims; ii++) {
            // We will use a Gaussian kernel and normalise by the distance of
            // the furthest point of the nearest k neighbours.
            float h = dist[(idx+1)*k - 1];

            // Initialise values to zero
            B[ii] = 0.0;

            for (int kk = 0; kk < k; kk++) {
                float d = dist[idx*k + kk];
                float z = exp(-(d/h)*(d/h)/2)/sqrt(2*M_PI);

                B[ii] += yvals[ii]*(xvals[ind[kk]*noDims+ii] - xQ[ii])*z*z;
            }

            for (int jj = 0; jj <= noDims; jj++) {
                A[ii*(noDims+1)+jj] = 0;

                for (int kk = 0; kk <= dataPoints[control]; kk++) {
                    float d = dist[idx*k + kk];
                    float z = exp(-(d/h)*(d/h)/2)/sqrt(2*M_PI);

                    A[ii*(noDims+1)+jj] += (xvals[ind[kk]*noDims+ii]-xQ[ii])*(
                            xvals[ind[kk]*noDims+jj] - xQ[jj])*z*z;
                }
            }
        }

        // 3. Use cuBlas to solve the set of linear equations to determine the
        // coefficients matrix.

        // a. LU Decomposition
        // --- Creating the array of pointers needed as input/output to the
        // batched getrf
        float **inout_pointers = (float **)malloc(sizeof(float*));
        inout_pointers[0] = A;

        int *pivotArray;
        int *infoArray;
        pivotArray = (int*)malloc((noDims+1)*sizeof(int));
        infoArray = (int*)malloc(sizeof(int));
        // The payoffs are always finite and should not be different for the
        // same inputs. Therefore, we ought not to have any singular matrices.
        cublasSgetrfBatched(handles[threadIdx.x],noDims+1,inout_pointers,
                noDims,pivotArray,infoArray,1);

//        if (infoArray[0] != 0) {
//            std::cout <<
//                    "Factorization of matrix %d Failed: Matrix may be singular"
//                    << std::endl;
//            cudaDeviceReset();
//            exit(EXIT_FAILURE);
//        }

        // b. Compute the inverse of A
        // Allocate space for the inverted matrix
        float **out_pointers = (float**)malloc(sizeof(float*));
        out_pointers[0] = C;

        cublasSgetriBatched(handles[threadIdx.x],noDims+1,(const float**)
                inout_pointers,
                noDims+1,pivotArray,out_pointers,noDims+1,infoArray,noDims+1);

//        if (infoArray[0] != 0) {
//            std::cout <<
//                    "Factorization of matrix %d Failed: Matrix may be singular"
//                    << std::endl;
//            cudaDeviceReset();
//            exit(EXIT_FAILURE);
//        }

        // Now multiply to get the coefficients
        float alpha1 = 1.0f;
        float beta1 = 0.0f;

        cublasSgemv(handles[threadIdx.x],CUBLAS_OP_N,noDims+1,noDims+1,&alpha1,
                C,noDims+1,B,noDims+1,&beta1,X,1);

        // 4. Compute the y value at the x point of interest using the just-
        //    found regression coefficients. This is simply the y intercept we
        //    just computed and save to the regression matrix.
        d_regression[year*noControls*(dimRes*noDims + (int)pow(dimRes,
                noDims)*2) + control*(dimRes*noDims + (int)pow(dimRes,
                noDims)*2) + noDims*dimRes + idx] = X[0];

        // Free memory
        free(A);
        free(B);
        free(C);
        free(xQ);
        free(X);
        free(pivotArray);
        free(infoArray);
    }
    free(dimIdx);
}

// WRAPPERS ///////////////////////////////////////////////////////////////////

void SimulateGPU::expPV(UncertaintyPtr uncertainty) {
    // Get device properties
    int device = 0;
    struct cudaDeviceProp properties;
    CUDA_CALL(cudaGetDeviceProperties(&properties, device));
    maxMultiProcessors = properties.multiProcessorCount;
    maxThreadsPerBlock = properties.maxThreadsPerBlock;

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
    CUDA_CALL(cudaMalloc((void **)&d_brownian, sizeof(float)*nYears*noPaths));
    CUDA_CALL(cudaMalloc((void **)&d_jumpSizes, sizeof(float)*nYears*noPaths));
    CUDA_CALL(cudaMalloc((void **)&d_jumps, sizeof(float)*nYears*noPaths));
    CUDA_CALL(cudaMalloc((void **)&d_results, sizeof(float)*nYears*noPaths));

    CURAND_CALL(curandCreateGenerator(&gen, CURAND_RNG_PSEUDO_DEFAULT));
    CURAND_CALL(curandSetPseudoRandomGeneratorSeed(gen, _seed));

    CURAND_CALL(curandGenerateNormal(gen, d_brownian, nYears*noPaths, 0.0f,
            uncertainty->getNoiseSD()*timeStep*vp->getCommoditySDMultipliers()(
            sc->getCommoditySD())));
    CURAND_CALL(curandGenerateNormal(gen, d_jumpSizes, nYears*noPaths,
            -pow(uncertainty->getPoissonJump()*vp->getCommoditySDMultipliers()(
            sc->getCommoditySD()),2)/2,pow(uncertainty->
            getPoissonJump()*vp->getCommoditySDMultipliers()(sc->getCommoditySD()
            ),2)));
    CURAND_CALL(curandGenerateUniform(gen, d_jumps, nYears*noPaths));

    CURAND_CALL(curandDestroyGenerator(gen));

    // Compute path values
    int noBlocks = (noPaths % maxThreadsPerBlock) ? (int)(
            noPaths/maxThreadsPerBlock + 1) : (int)
            (noPaths/maxThreadsPerBlock);
    int noThreadsPerBlock = min(maxThreadsPerBlock,nYears*noPaths);

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
    free(results);
}

void SimulateGPU::eMMN(const Eigen::MatrixXd& A, const Eigen::MatrixXd& B,
        Eigen::MatrixXd& C) {

    int device = 0;
    struct cudaDeviceProp properties;
    cudaGetDeviceProperties(&properties, device);
    maxMultiProcessors = properties.multiProcessorCount;
    maxThreadsPerBlock = properties.maxThreadsPerBlock;

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

    // declare the number of blocks per grid and the number of threads per block
    dim3 threadsPerBlock(a, d);
    dim3 blocksPerGrid(1, 1);
        if (a*d > maxThreadsPerBlock){
            threadsPerBlock.x = maxThreadsPerBlock;
            threadsPerBlock.y = maxThreadsPerBlock;
            blocksPerGrid.x = ceil(double(a)/double(threadsPerBlock.x));
            blocksPerGrid.y = ceil(double(d)/double(threadsPerBlock.y));
        }

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
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
}

void SimulateGPU::eMM(const Eigen::MatrixXd& A, const Eigen::MatrixXd& B,
        Eigen::MatrixXd& C) {

    int device = 0;
    struct cudaDeviceProp properties;
    cudaGetDeviceProperties(&properties, device);
    maxMultiProcessors = properties.multiProcessorCount;
    maxThreadsPerBlock = properties.maxThreadsPerBlock;

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

    cudaMalloc(&d_C,a*d*sizeof(float));

    // declare the number of blocks per grid and the number of threads per block
    dim3 threads(BLOCK_SIZE,VECTOR_SIZE);
    dim3 grid(d/(BLOCK_SIZE*VECTOR_SIZE), a/BLOCK_SIZE);

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

void SimulateGPU::ewMM(const Eigen::MatrixXd& A, const Eigen::MatrixXd &B,
        Eigen::MatrixXd &C) {

    int device = 0;
    struct cudaDeviceProp properties;
    cudaGetDeviceProperties(&properties, device);
    maxMultiProcessors = properties.multiProcessorCount;
    maxThreadsPerBlock = properties.maxThreadsPerBlock;

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

void SimulateGPU::ewMD(const Eigen::MatrixXd& A, const Eigen::MatrixXd& B,
        Eigen::MatrixXd& C) {

    int device = 0;
    struct cudaDeviceProp properties;
    cudaGetDeviceProperties(&properties, device);
    maxMultiProcessors = properties.multiProcessorCount;
    maxThreadsPerBlock = properties.maxThreadsPerBlock;

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

void SimulateGPU::lineSegmentIntersect(const Eigen::MatrixXd& XY1, const
        Eigen::MatrixXd& XY2, Eigen::VectorXi& crossings) {

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
    CUDA_CALL(cudaMemcpy(d_XY1,XY1f.data(),XY1.rows()*XY1.cols()*sizeof(float),
            cudaMemcpyHostToDevice));
    CUDA_CALL(cudaMalloc(&d_XY2,XY2.rows()*XY2.cols()*sizeof(float)));
    CUDA_CALL(cudaMemcpy(d_XY2,XY2f.data(),XY2.rows()*XY2.cols()*sizeof(float),
            cudaMemcpyHostToDevice));

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
            (noCombos/maxThreadsPerBlock + 1) : (noCombos/maxThreadsPerBlock);
    double number = (double)(noBlocks)/(((double)maxBlocksPerGrid)*
            ((double)maxBlocksPerGrid));
    int blockYDim = ((number - floor(number)) > 0 ) ? (int)number + 1 :
            (int)number;
    int blockXDim = (int)min(maxBlocksPerGrid,noBlocks);

    dim3 dimGrid(blockXDim,blockYDim);
    pathAdjacencyKernel<<<dimGrid,maxThreadsPerBlock>>>(XY1.rows(),XY2.rows(),
            d_XY1,d_XY2,d_X4_X3,d_Y4_Y3,d_X2_X1,d_Y2_Y1,d_adjacency);
    CUDA_CALL(cudaPeekAtLastError());
    CUDA_CALL(cudaDeviceSynchronize());

//    cudaError_t error = cudaGetLastError();
//    if (error != cudaSuccess) {
//      fprintf(stderr, "ERROR: %s \n", cudaGetErrorString(error));
//    }

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
    CUDA_CALL(cudaFree(d_X4_X3));
    CUDA_CALL(cudaFree(d_Y4_Y3));
    CUDA_CALL(cudaFree(d_X2_X1));
    CUDA_CALL(cudaFree(d_Y2_Y1));
    CUDA_CALL(cudaFree(d_cross));
}

void SimulateGPU::buildPatches(int W, int H, int skpx, int skpy, int xres,
        int yres, int noRegions, double xspacing, double yspacing, double
        subPatchArea, HabitatTypePtr habTyp, const Eigen::MatrixXi&
        labelledImage, const Eigen::MatrixXd& populations,
        std::vector<HabitatPatchPtr>& patches, double& initPop,
        Eigen::VectorXd& initPops, int& noPatches) {

    // Get device properties
    int device = 0;
    struct cudaDeviceProp properties;
    cudaGetDeviceProperties(&properties, device);
    maxMultiProcessors = properties.multiProcessorCount;
    maxThreadsPerBlock = properties.maxThreadsPerBlock;

    Eigen::MatrixXf popsFloat = populations.cast<float>();

    float *results, *d_results, *d_populations;
    int *d_labelledImage;

    results = (float*)malloc(xres*yres*noRegions*5*sizeof(float));
    CUDA_CALL(cudaMalloc((void **)&d_results,xres*yres*noRegions*5*
            sizeof(float)));

    CUDA_CALL(cudaMalloc((void **)&d_labelledImage,H*W*sizeof(int)));
    CUDA_CALL(cudaMemcpy(d_labelledImage,labelledImage.data(),H*W*sizeof(int),
            cudaMemcpyHostToDevice));

    CUDA_CALL(cudaMalloc((void **)&d_populations,H*W*sizeof(float)));
    CUDA_CALL(cudaMemcpy(d_populations,popsFloat.data(),H*W*sizeof(float),
            cudaMemcpyHostToDevice));

    int noBlocks = ((xres*yres*noRegions) % maxThreadsPerBlock)? (int)(xres*
            yres*noRegions/maxThreadsPerBlock + 1) : (int)(xres*yres*noRegions/
            maxThreadsPerBlock);
    int noThreadsPerBlock = min(maxThreadsPerBlock,xres*yres*noRegions);

    patchComputation<<<noBlocks,noThreadsPerBlock>>>(xres*yres*noRegions,
            W, H, skpx, skpy, xres,yres,(float)subPatchArea,(float)xspacing,
            (float)yspacing,(float)habTyp->getMaxPop(),noRegions,
            d_labelledImage,d_populations,d_results);
    CUDA_CALL(cudaPeekAtLastError());
    CUDA_CALL(cudaDeviceSynchronize());

//    cudaError_t error = cudaGetLastError();
//    if (error != cudaSuccess) {
//      fprintf(stderr, "ERROR: %s \n", cudaGetErrorString(error));
//    }

    CUDA_CALL(cudaMemcpy(results,d_results,xres*yres*noRegions*5*sizeof(float),
               cudaMemcpyDeviceToHost));

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
            patches[noPatches++] = hab;
        }
    }

    CUDA_CALL(cudaFree(d_populations));
    CUDA_CALL(cudaFree(d_labelledImage));
    CUDA_CALL(cudaFree(d_results));
    free(results);
}

void SimulateGPU::simulateMTECUDA(SimulatorPtr sim,
        std::vector<SpeciesRoadPatchesPtr>& srp,
        std::vector<Eigen::VectorXd>& initPops,
        std::vector<Eigen::VectorXd>& capacities,
        Eigen::MatrixXd& endPops) {

    // Get device properties
    int device = 0;
    struct cudaDeviceProp properties;
    cudaGetDeviceProperties(&properties, device);
    maxMultiProcessors = properties.multiProcessorCount;
    maxThreadsPerBlock = properties.maxThreadsPerBlock;

    // We convert all inputs to floats from double as CUDA is much faster in
    // single precision than double precision

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
        float stepSize = (float)sim->getRoad()->getOptimiser()->getEconomic()->
                getTimeStep();
        int nPatches = capacities[ii].size();

        float *speciesParams, *d_speciesParams, *eps, *d_initPops, *d_eps,
                *d_caps;

        speciesParams = (float*)malloc(8*sizeof(float));
        CUDA_CALL(cudaMalloc((void**)&d_speciesParams,8*sizeof(float)));

        //int counter2 = 0;

        // Read in the information into the correct format
        speciesParams[0] = srp[ii]->getSpecies()->getGrowthRate()->getCurrent()
                *varParams->getGrowthRatesMultipliers()(scenario->getPopGR());
        speciesParams[1] = srp[ii]->getSpecies()->getGrowthRate()->getMean()
                *varParams->getGrowthRatesMultipliers()(scenario->getPopGR());
        speciesParams[2] = srp[ii]->getSpecies()->getGrowthRate()->getNoiseSD()
                *varParams->getGrowthRateSDMultipliers()(scenario->
                getPopGRSD());
        speciesParams[3] = srp[ii]->getSpecies()->getThreshold()*varParams->
                getPopulationLevels()(scenario->getPopLevel());
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

        //counter2 += pow(srp[ii]->getHabPatches().size(),2);

        CUDA_CALL(cudaMemcpy(d_speciesParams,speciesParams,8*sizeof(float),
                cudaMemcpyHostToDevice));

        // RANDOM MATRICES
        float *d_growthRates, *d_uBrownianSpecies, *d_uJumpSizesSpecies,
                *d_uJumpsSpecies;
        //allocate space for 100 floats on the GPU
        //could also do this with thrust vectors and pass a raw pointer
        CUDA_CALL(cudaMalloc((void**)&d_growthRates,nYears*noPaths*nPatches*
                sizeof(float)));
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
        CURAND_CALL(curandGenerateNormal(gen, d_growthRates, nYears*noPaths*
                nPatches,0.0f,1.0f));

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
        CUDA_CALL(cudaMemcpy(d_initPops,initPopsF.data(),initPops[ii].size()*
                sizeof(float),cudaMemcpyHostToDevice));

        // END POPULATIONS
        eps = (float*)malloc(noPaths*sizeof(float));
        CUDA_CALL(cudaMalloc((void**)&d_eps, noPaths*sizeof(float)));

        // TEMPORARY KERNEL POPULATIONS
        float *d_pathPops;
        CUDA_CALL(cudaMalloc((void**)&d_pathPops, noPaths*2*initPops[ii].size()
                *sizeof(float)));

//        for (int jj = 0; jj < noPaths; jj++) {
//            eps[jj] = 0.0f;
//        }

//        cudaMemcpy(d_eps,eps,noPaths*sizeof(float),cudaMemcpyHostToDevice);

        // CAPACITIES
        Eigen::VectorXf capsF = capacities[ii].cast<float>();
        CUDA_CALL(cudaMalloc((void**)&d_caps,capacities[ii].size()*
                sizeof(float)));
        CUDA_CALL(cudaMemcpy(d_caps,capsF.data(),capacities[ii].size()*
                sizeof(float),cudaMemcpyHostToDevice));

        // MOVEMENT AND MORTALITY MATRIX
        // Convert the movement and mortality matrix to a sparse matrix for use
        // in the kernel efficiently.
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
                    capacities[ii].size(),sparseOut.data(),elemsPerCol.data(),
                    rowIdx.data(),totalElements);

            // Allocate GPU memory for sparse matrix
            CUDA_CALL(cudaMalloc((void**)&d_sparseOut,totalElements*
                    sizeof(float)));
            CUDA_CALL(cudaMalloc((void**)&d_rowIdx,totalElements*sizeof(int)));
            CUDA_CALL(cudaMalloc((void**)&d_elemsPerCol,capacities[ii].size()*
                    sizeof(int)));

            CUDA_CALL(cudaMemcpy(d_sparseOut,sparseOut.data(),totalElements*
                    sizeof(float),cudaMemcpyHostToDevice));
            CUDA_CALL(cudaMemcpy(d_rowIdx,rowIdx.data(),totalElements*sizeof(
                    int),cudaMemcpyHostToDevice));
            CUDA_CALL(cudaMemcpy(d_elemsPerCol,elemsPerCol.data(),
                    capacities[ii].size()*sizeof(int),cudaMemcpyHostToDevice));
        }

        ///////////////////////////////////////////////////////////////////////
        // Perform N simulation paths. Currently, there is no species
        // interaction, so we run each kernel separately and do not need to use
        // the Thrust library.

        // Modify the below code the run the kernel multiple times depending on
        // how many paths are required.

        // Blocks and threads for each path
        int noBlocks = (int)(noPaths % maxThreadsPerBlock)?
                (int)(noPaths/maxThreadsPerBlock + 1) :
                (int)(noPaths/maxThreadsPerBlock);
        int noThreadsPerBlock = min(noPaths,maxThreadsPerBlock);
        // Maximum number of floating points to store in shared memory will be
        // 8000 for a 64KB shared memory block.
//        int noTiles = (int)((int)pow((double)capacities[ii].size(),2) %
//                max_shared_floats) ? (int)((int)pow((double)capacities[ii]
//                .size(),2) / max_shared_floats + 1) : (int)((int)pow(
//                (double)capacities[ii].size(),2) / max_shared_floats);
//        int tileDim = capacities[ii].size()/noTiles;
//        int sharedMemElements = tileDim*capacities[ii].size()*sizeof(float);

//        clock_t begin = clock();

        mteKernel<<<noBlocks,noThreadsPerBlock>>>(noPaths,nYears,
                capacities[ii].size(),stepSize,d_growthRates,
                d_uBrownianSpecies,d_uJumpSizesSpecies,d_uJumpsSpecies,
                d_speciesParams,d_initPops,d_caps,d_sparseOut,d_rowIdx,
                d_elemsPerCol,d_pathPops,d_eps);
        CUDA_CALL(cudaPeekAtLastError());
        CUDA_CALL(cudaDeviceSynchronize());

//        clock_t end = clock();
//        double elapsed_secs = double(end - begin) / CLOCKS_PER_SEC;
//        std::cout << "MTE Time: " << elapsed_secs << " s" << std::endl;

//        cudaError_t error = cudaGetLastError();
//        if (error != cudaSuccess) {
//          fprintf(stderr, "ERROR: %s \n", cudaGetErrorString(error));
//        }

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
}

void SimulateGPU::simulateROVCUDA(SimulatorPtr sim,
        std::vector<SpeciesRoadPatchesPtr>& srp,
        std::vector<Eigen::MatrixXd> &adjPops, Eigen::MatrixXd& unitProfits,
        Eigen::MatrixXd& condExp, Eigen::MatrixXi& optCont) {
    // Currently there is no species interaction. This can be a future question
    // and would be an interesting extension on how it can be implemented,
    // what the surrogate looks like and how the patches are formed.

    // The predictor variables are the adjusted population and current unit
    // profit. To determine the optimal control, we find the adjusted
    // populations for each species for each control and check against the
    // policy map.

    // Get device properties
    int device = 0;
    struct cudaDeviceProp properties;
    cudaGetDeviceProperties(&properties, device);
    maxMultiProcessors = properties.multiProcessorCount;
    maxThreadsPerBlock = properties.maxThreadsPerBlock;

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

    // Get important values for computation
    int nYears = sim->getRoad()->getOptimiser()->getEconomic()->getYears();
    int noPaths = sim->getRoad()->getOptimiser()->getOtherInputs()->
            getNoPaths();
    int noControls = program->getFlowRates().size();
    int noUncertainties = commodities.size() + fuels.size();

    // Fixed cost per unit traffic
    double unitCost = sim->getRoad()->getAttributes()->getUnitVarCosts();
    // Fuel consumption per vehicle class per unit traffic (L)
    Eigen::VectorXf fuelCosts = sim->getRoad()->getCosts()->getUnitFuelCost()
            .cast<float>();
    // Load per unit traffic
    float unitRevenue = (float)sim->getRoad()->getCosts()->getUnitRevenue();
    float stepSize = (float)optimiser->getEconomic()->getTimeStep();
    float rrr = (float)optimiser->getEconomic()->getRRR();

    // Get the important values for the road first and convert them to formats
    // that the kernel can use

    // Initialise CUDA memory /////////////////////////////////////////////////

    // 1. Transition and survival matrices for each species and each control
    float *transitions, *survival, *initPops, *capacities, *speciesParams,
            *uncertParams, *d_transitions, *d_survival, *d_initPops,
            *d_tempPops, *d_capacities, *d_speciesParams, *d_uncertParams,
            *d_fuelCosts;

    int *noPatches, *d_noPatches;

    noPatches = (int*)malloc(srp.size()*sizeof(int));

    int patches = 0;
    int transition = 0;

    for (int ii = 0; ii < srp.size(); ii++) {
        noPatches[ii] = srp[ii]->getHabPatches().size();
        patches += noPatches[ii];
        transition += pow(patches,2);
    }

    initPops = (float*)malloc(patches*sizeof(float));
    capacities = (float*)malloc(patches*sizeof(float));
    transitions = (float*)malloc(transition*sizeof(float));
    survival = (float*)malloc(transition*noControls*sizeof(float));
    speciesParams = (float*)malloc(srp.size()*8*sizeof(float));
    uncertParams = (float*)malloc(noUncertainties*6*sizeof(float));

    cudaMalloc((void**)&d_noPatches,srp.size()*sizeof(int));
    cudaMalloc((void**)&d_initPops,patches*sizeof(float));
    cudaMalloc((void**)&d_capacities,patches*sizeof(float));
    cudaMalloc((void**)&d_transitions,transition*sizeof(float));
    cudaMalloc((void**)&d_survival,transition*noControls*sizeof(float));
    cudaMalloc((void**)&d_speciesParams,srp.size()*8*sizeof(float));
    cudaMalloc((void**)&d_uncertParams,noUncertainties*6*sizeof(float));
    cudaMalloc((void**)&d_tempPops,noPaths*nYears*patches*srp.size()*
            sizeof(float));
    cudaMalloc((void**)&d_fuelCosts,fuelCosts.size()*sizeof(float));

    int counter1 = 0;
    int counter2 = 0;
    int counter3 = 0;

    // Read in the information into the correct format
    for (int ii = 0; ii < srp.size(); ii++) {
        memcpy(initPops+counter1,srp[ii]->getInitPops().data(),
                srp[ii]->getHabPatches().size()*sizeof(float));
        memcpy(capacities+counter1,srp[ii]->getCapacities().data(),
                srp[ii]->getHabPatches().size()*sizeof(float));

        speciesParams[counter1] = srp[ii]->getSpecies()->getGrowthRate()->
                getCurrent()*varParams->getGrowthRatesMultipliers()(scenario->
                getPopGR());
        speciesParams[counter1+1] = srp[ii]->getSpecies()->getGrowthRate()->
                getMean()*varParams->getGrowthRatesMultipliers()(scenario->
                getPopGR());
        speciesParams[counter1+2] = srp[ii]->getSpecies()->getGrowthRate()->
                getNoiseSD()*varParams->getGrowthRateSDMultipliers()(scenario->
                getPopGRSD());
        speciesParams[counter1+3] = srp[ii]->getSpecies()->getThreshold()*
                varParams->getPopulationLevels()(scenario->getPopLevel());
        speciesParams[counter1+4] = srp[ii]->getSpecies()->getGrowthRate()->
                getMRStrength()*varParams->getGrowthRateSDMultipliers()(
                scenario->getPopGRSD());
        speciesParams[counter1+5] = srp[ii]->getSpecies()->getGrowthRate()->
                getPoissonJump()*varParams->getGrowthRateSDMultipliers()(
                scenario->getPopGRSD());
        speciesParams[counter1+6] = srp[ii]->getSpecies()->getGrowthRate()->
                getJumpProb()*varParams->getGrowthRateSDMultipliers()(
                scenario->getPopGRSD());
        speciesParams[counter1+7] = srp[ii]->getSpecies()->
                getLocalVariability();

        counter1 += 8;

        memcpy(transitions+counter2,srp[ii]->getTransProbs().data(),
                pow(srp[ii]->getHabPatches().size(),2));
        counter2 += pow(srp[ii]->getHabPatches().size(),2);

        for (int jj = 0; jj < noControls; jj++) {
            memcpy(survival+counter3,srp[ii]->getSurvivalProbs()[jj].data(),
                pow(srp[ii]->getHabPatches().size(),2));
            counter3 += pow(srp[ii]->getHabPatches().size(),2);
        }
    }

    for (int ii = 0; ii < fuels.size(); ii++) {
        uncertParams[6*ii] = fuels[ii]->getCurrent();
        uncertParams[6*ii+1] = fuels[ii]->getMean()*varParams->
                getCommodityMultipliers()(scenario->getCommodity());
        uncertParams[6*ii+2] = fuels[ii]->getNoiseSD()*varParams->
                getCommoditySDMultipliers()(scenario->getCommoditySD());
        uncertParams[6*ii+3] = fuels[ii]->getMRStrength()*varParams->
                getCommoditySDMultipliers()(scenario->getCommoditySD());
        uncertParams[6*ii+4] = fuels[ii]->getPoissonJump()*varParams->
                getCommoditySDMultipliers()(scenario->getCommoditySD());
        uncertParams[6*ii+5] = fuels[ii]->getJumpProb()*varParams->
                getCommoditySDMultipliers()(scenario->getCommoditySD());
    }

    // Set the fuel indices for the vehicle classes corresponding to the fuels
    // order above
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
        uncertParams[fuels.size()*6 + 6*ii] = commodities[ii]->getCurrent();
        uncertParams[fuels.size()*6 + 6*ii+1] = commodities[ii]->getMean()*
                varParams->getCommodityMultipliers()(scenario->getCommodity());
        uncertParams[fuels.size()*6 + 6*ii+2] = commodities[ii]->getNoiseSD()*
                varParams->getCommoditySDMultipliers()(scenario->
                getCommoditySD());
        uncertParams[fuels.size()*6 + 6*ii+3] = commodities[ii]->
                getMRStrength()*varParams->getCommoditySDMultipliers()(scenario
                ->getCommoditySD());;
        uncertParams[fuels.size()*6 + 6*ii+4] = commodities[ii]->
                getPoissonJump()*varParams->getCommoditySDMultipliers()(
                scenario->getCommoditySD());
        uncertParams[fuels.size()*6 + 6*ii+5] = commodities[ii]->
                getJumpProb()*varParams->getCommoditySDMultipliers()(
                scenario->getCommoditySD());
    }

    // Transfer the data to the device
    cudaMemcpy(d_noPatches,noPatches,srp.size()*sizeof(int),
            cudaMemcpyHostToDevice);
    cudaMemcpy(d_initPops,initPops,patches*sizeof(float),
            cudaMemcpyHostToDevice);
    cudaMemcpy(d_transitions,transitions,transition*sizeof(float),
            cudaMemcpyHostToDevice);
    cudaMemcpy(d_survival,survival,transition*sizeof(float),
            cudaMemcpyHostToDevice);
    cudaMemcpy(d_speciesParams,speciesParams,srp.size()*8*sizeof(float),
            cudaMemcpyHostToDevice);
    cudaMemcpy(d_uncertParams,uncertParams,noUncertainties*6*sizeof(float),
            cudaMemcpyHostToDevice);
    cudaMemcpy(d_fuelCosts,fuelCosts.data(),fuelCosts.size()*sizeof(float),
            cudaMemcpyHostToDevice);

    // Free the host memory
    free(transitions);
    free(survival);
    free(initPops);
    free(capacities);
    free(speciesParams);
    free(uncertParams);

    // Exogenous parameters (fuels, commodities)
    // Ore composition is simply Gaussian for now
    float *d_randCont, *d_growthRate, *d_uBrownian, *d_uJumpSizes,
            *d_uJumps, *d_uResults, *d_uComposition, *d_flowRates,
            *d_uBrownianSpecies, *d_uJumpSizesSpecies, *d_uJumpsSpecies;
    int *d_controls;

    srand(time(NULL));
    int _seed = rand();
    curandGenerator_t gen;
    curandCreateGenerator(&gen, CURAND_RNG_PSEUDO_DEFAULT);
    curandSetPseudoRandomGeneratorSeed(gen, _seed);

    cudaMalloc((void**)&d_randCont,nYears*noPaths*sizeof(float));

    // 2. Random matrices for randomised control
    curandGenerateUniform(gen, d_randCont, nYears*noPaths);

    cudaMalloc((void**)&d_controls,nYears*noPaths*sizeof(int));

    int noBlocks = (int)(noPaths*nYears % maxThreadsPerBlock) ?
            (int)(noPaths*nYears/maxThreadsPerBlock  + 1) :
            (int)(noPaths*nYears/maxThreadsPerBlock);
    int noThreadsPerBlock = min(noPaths,maxThreadsPerBlock);

    randControls<<<noBlocks,noThreadsPerBlock>>>(noPaths,noControls,
            d_randCont,d_controls);
    // The flow rates corresponding to the random controls
    Eigen::MatrixXf flowRatesF = program->getFlowRates().cast<float>();
    cudaMalloc((void**)&d_flowRates,noControls*sizeof(float));
    cudaMemcpy(d_flowRates,flowRatesF.data(),noControls*sizeof(float),
            cudaMemcpyHostToDevice);

    // We no longer need the floating point random controls vector
    cudaFree(d_randCont);

    // Endogenous uncertainty
    cudaMalloc((void**)&d_growthRate,nYears*noPaths*patches*srp.size()*
            sizeof(float));
    cudaMalloc((void**)&d_uBrownianSpecies,nYears*noPaths*srp.size()*
            sizeof(float));
    cudaMalloc((void**)&d_uJumpSizesSpecies,nYears*noPaths*srp.size()*
            sizeof(float));
    cudaMalloc((void**)&d_uJumpsSpecies,nYears*noPaths*srp.size()*
            sizeof(float));

    // Exogenous uncertainty
    cudaMalloc((void**)&d_uBrownian,nYears*noPaths*noUncertainties*
            sizeof(float));
    cudaMalloc((void**)&d_uJumpSizes,nYears*noPaths*noUncertainties*
            sizeof(float));
    cudaMalloc((void**)&d_uJumps,nYears*noPaths*noUncertainties*
            sizeof(float));
    cudaMalloc((void**)&d_uResults,noUncertainties*nYears*noPaths*
            sizeof(float));
    cudaMalloc((void**)&d_uComposition,nYears*noPaths*(commodities.size())*
            sizeof(float));

    // 3. Random matrices for growth rate parameter for species
    curandGenerateNormal(gen, d_growthRate, nYears*noPaths*patches*srp.size(),
            0.0f,1.0f);

    curandGenerateNormal(gen, d_uBrownianSpecies, nYears*noPaths*srp.size(),
            0.0f,1.0f);

    curandGenerateNormal(gen, d_uJumpSizesSpecies, nYears*noPaths*srp.size(),
            0.0f,1.0f);

    curandGenerateUniform(gen, d_uJumpsSpecies, nYears*noPaths*srp.size());

    // 4. Random matrices for other uncertainties
    curandGenerateNormal(gen, d_uBrownian, nYears*noPaths*noUncertainties,0.0f,
            1.0f);

    curandGenerateNormal(gen, d_uJumpSizes, nYears*noPaths*noUncertainties,
            0.0f,1.0f);

    curandGenerateUniform(gen, d_uJumps, nYears*noPaths*noUncertainties);

    // 5. Ore composition paths
    for (int ii = 0; ii < commodities.size(); ii++) {
        curandGenerateNormal(gen, d_uComposition + ii*nYears*noPaths,
                nYears*noPaths,commodities[ii]->getOreContent(),
                commodities[ii]->getOreContentSD()*varParams->
                getCommodityPropSD()(scenario->getOreCompositionSD()));
    }

    // Destroy generator
    curandDestroyGenerator(gen);
    cudaDeviceSynchronize();

    // Finally, allocate space on the device for the path results. This is what
    // we use in our policy map.
    float *d_totalPops, *d_aars, *d_mcPops;
    cudaMalloc(&d_totalPops,srp.size()*(nYears+1)*noPaths*sizeof(float));
    cudaMalloc(&d_mcPops,(nYears+1)*noPaths*patches*sizeof(float));
    cudaMalloc((void**)&d_aars,srp.size()*(nYears+1)*noPaths*noControls*
            sizeof(float));

    // Compute forward paths (CUDA kernel)
    noBlocks = (int)(noPaths % maxThreadsPerBlock) ?
            (int)(noPaths/maxThreadsPerBlock + 1) :
            (int)(noPaths/maxThreadsPerBlock);

    forwardPathKernel<<<noBlocks,noThreadsPerBlock>>>(1,noPaths,nYears,
            srp.size(),patches,noControls,fuels.size(),noUncertainties,
            stepSize,d_tempPops,d_transitions,d_survival,d_speciesParams,
            d_capacities,d_aars,d_uncertParams,d_controls,d_uJumps,
            d_uBrownian,d_uJumpSizes,d_uJumpsSpecies,d_uBrownianSpecies,
            d_uJumpSizesSpecies,d_growthRate,d_uResults,d_totalPops);
    cudaDeviceSynchronize();

    // Free device memory that is no longer needed
    cudaFree(d_uBrownian);
    cudaFree(d_uJumpSizes);
    cudaFree(d_uJumps);
    cudaFree(d_uBrownianSpecies);
    cudaFree(d_uJumpSizesSpecies);
    cudaFree(d_uJumpsSpecies);

    // Determine the number of uncertainties. The uncertainties are the unit
    // payoff of the road (comprised of commodity and fuel prices, which are
    // pre-processed to determine covariances etc. and are treated as a single
    // uncertainty) and the adjusted population of each species under each
    // control.
    int noDims = srp.size() + 1;

    // Prepare the floating point versions of the output
    float *d_condExp;
    // Prepare the index of the fuel use by each vehicle class
    int *d_fuelIdx;

    cudaMalloc(&d_fuelIdx,fuelIdx.size()*sizeof(int));
    cudaMemcpy(d_fuelIdx,fuelIdx.data(),fuelIdx.size()*sizeof(int),
            cudaMemcpyHostToDevice);

    // Where to copy the results back to the host in floating point ready to
    // copy to the double precision outputs.
    std::vector<Eigen::MatrixXf> adjPopsF(nYears+1);

    for (int ii = 0; ii <= nYears; ii++) {
        adjPopsF[ii].resize(noPaths,srp.size());
    }

    // 2. Unit profits map inputs at each time step
    Eigen::MatrixXf unitProfitsF(nYears+1,noPaths);

    float *d_unitProfits;
    cudaMalloc((void**)&d_unitProfits,unitProfitsF.rows()*unitProfitsF.
            cols()*sizeof(float));

    // 3. Optimal profit-to-go outputs matrix (along each path)
    Eigen::MatrixXf condExpF(noPaths,nYears);

    cudaMalloc((void**)&d_condExp,condExp.rows()*condExp.cols()*sizeof(float));

    int* d_optCont;
    cudaMalloc((void**)&d_optCont,optCont.rows()*optCont.cols()*sizeof(int));

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
        {
            // I probably will not code this method
        }
        break;

        case Optimiser::ALGO5:
        {
            // I will not code this method
        }
        break;

        case Optimiser::ALGO6:
        // Full model with local linear kernel and forward path recomputation.
        // This method is the most accurate but is very slow.
        {
            // Make the grid for regressions. One for each control for each
            // time step mapped against the N-dimensional grid. We interpolate
            // when we compute the forward paths.
            float *d_regression, *d_adjPops;
            int dimRes = sim->getRoad()->getOptimiser()->getOtherInputs()->
                    getDimRes();

            cudaMalloc((void**)&d_regression,nYears*noControls*(dimRes*noDims +
                    pow(dimRes,noDims)*2)*sizeof(float));
            cudaMalloc((void**)&d_adjPops,adjPops[0].rows()*adjPops[0].cols()*
                    sizeof(float));

            // --- CUBLAS initialization
            cublasHandle_t *d_handles;
            cudaMalloc((void**)&d_handles,maxThreadsPerBlock*sizeof(
                    cublasHandle_t));
            createHandles<<<1,maxThreadsPerBlock>>>(d_handles,
                    maxThreadsPerBlock);
            cudaDeviceSynchronize();

            // The last step is simply the valid control with the highest
            // single period payoff
            optimalForwardPaths<<<noBlocks,noThreadsPerBlock>>>(nYears,noPaths,
                    nYears,srp.size(),patches,noControls,noUncertainties,
                    stepSize,unitCost,unitRevenue,rrr,fuelCosts.size(),
                    commodities.size(),dimRes,d_flowRates,d_fuelCosts,
                    d_tempPops,d_totalPops,d_transitions,d_survival,
                    d_speciesParams,d_growthRate,d_capacities,d_aars,
                    d_regression,d_uComposition,d_uResults,d_fuelIdx,d_condExp,
                    d_optCont,d_adjPops,d_unitProfits);
            cudaDeviceSynchronize();

            // Copy the adjusted populations to the output variable. This is
            // only provided for completeness. The algorithm does not use the
            // results as they pertain to the very last time step.
            cudaMemcpy(adjPopsF[nYears].data(),d_adjPops,srp.size()*noPaths*
                    sizeof(float),cudaMemcpyDeviceToHost);

            // Find the maximum and minimum x value along each path for the
            // dependant variables. This allocates space for the input
            // variables for the regressions.
            float *d_xmaxes, *d_xmins;
            cudaMalloc((void**)&d_xmaxes,noControls*noDims*sizeof(float));
            cudaMalloc((void**)&d_xmins,noControls*noDims*sizeof(float));

            // For each backward step not including the last period, we need to
            // determine the adjusted population for each species and the unit
            // payoffs.
            for (int ii = nYears-1; ii >= 0; ii--) {
                // Perform regression and save results
                int noBlocks2 = (int)((int)pow(dimRes,noDims)*noControls %
                        maxThreadsPerBlock) ? (int)(pow(dimRes,noDims)*
                        noControls/maxThreadsPerBlock + 1) : (int)(
                        pow(dimRes,noDims)*noControls/maxThreadsPerBlock);
                int maxThreadsPerBlock2 = min((int)pow(dimRes,noDims)*
                        noControls,maxThreadsPerBlock);

                float *d_xin, *d_xvals, *d_yvals;
                int* d_dataPoints;
                // The data points are arranged so that the number of rows
                // equals the number of dimensions and the number of columns
                // equals the number of data points.
                cudaMalloc((void**)&d_xin,noDims*noPaths*sizeof(float));
                cudaMalloc((void**)&d_xvals,noControls*noDims*noPaths*
                        sizeof(float));
                cudaMalloc((void**)&d_yvals,noControls*noPaths*sizeof(float));
                cudaMalloc((void**)&d_dataPoints,noControls*sizeof(int));

                // Compute the state values
                computePathStates<<<noBlocks,noThreadsPerBlock>>>(noPaths,
                        nYears,noDims,noControls,ii,unitCost,unitRevenue,
                        d_controls,fuels.size(),d_fuelCosts,d_uResults,
                        d_uComposition,noUncertainties,d_fuelIdx,
                        commodities.size(),d_aars,d_totalPops,d_xin);

                // This kernel does not take advantage of massive parallelism.
                // It is simply to allow us to call data that is already on
                // the device for allocating data for use in the regressions.
                allocateXYRegressionData<<<1,1>>>(noPaths,noControls,noDims,
                        ii,d_controls,d_xin,d_condExp,d_dataPoints,d_xvals,
                        d_yvals);
                cudaDeviceSynchronize();

                // Get the minimum and maximum X value for each dimension for
                // each control.
                computeStateMinMax<<<1,1>>>(noControls,noDims,noPaths,
                        d_dataPoints,d_xvals,d_xmins,d_xmaxes);

                // Allocate the k nearest neighbours for each design point in
                // order to do the regression stage after this. This component is
                // the slowest component.
                int *dataPoints;
                dataPoints = (int*)malloc(noControls*sizeof(int));
                cudaMemcpy(dataPoints,&d_dataPoints,noControls*sizeof(int),
                        cudaMemcpyDeviceToHost);

                for (int jj = 0; jj < noControls; jj++) {
                    // Use 5% of the nearest points for now
                    int k = dataPoints[jj]/5;

                    // We first need to perform a k nearest neighbour search
                    float *ref, *query, *dist;
                    int *ind;
                    ref = (float*)malloc(dataPoints[jj]*noDims*sizeof(float));
                    query = (float*)malloc(pow(dimRes,noDims)*noDims*
                            sizeof(float));
                    dist = (float*)malloc(pow(dimRes,noDims)*k*sizeof(float));
                    ind = (int*)malloc(pow(dimRes,noDims)*k*sizeof(int));

                    float *d_queryPts, *d_dist;
                    int *d_ind;
                    cudaMalloc((void**)&d_queryPts,pow(dimRes,noDims)*noDims*
                            sizeof(float));
                    cudaMalloc((void**)&d_dist,pow(dimRes,noDims)*k*sizeof(
                            float));
                    cudaMalloc((void**)&d_ind,pow(dimRes,noDims)*k*sizeof(
                            int));

                    createQueryPoints<<<noBlocks2,maxThreadsPerBlock2>>>((int)
                            pow(dimRes,noDims),noDims,dimRes,jj,noControls,ii,
                            d_xmins,d_xmaxes,d_regression,d_queryPts);
                    cudaMemcpy(query,d_queryPts,pow(dimRes,noDims)*noDims*
                            sizeof(float),cudaMemcpyDeviceToHost);
                    cudaFree(d_queryPts);

                    cudaMemcpy(ref,d_xvals + jj*noPaths*noDims,
                            dataPoints[jj]*noDims*sizeof(float),
                            cudaMemcpyDeviceToHost);

                    // Compute the knn searches
                    knn_cublas_with_indexes::knn(ref,dataPoints[jj],query,pow(
                            dimRes,noDims)*noDims,noDims,k,dist,ind);

                    cudaMemcpy(d_dist,dist,pow(dimRes,noDims)*k*sizeof(float),
                            cudaMemcpyHostToDevice);

                    cudaMemcpy(d_ind,ind,pow(dimRes,noDims)*k*sizeof(float),
                            cudaMemcpyHostToDevice);

                    free(ref);
                    free(query);
                    free(dist);
                    free(ind);
                    // Perform the regression for this control at this time at
                    // each of the query points.
                    multiLocLinReg<<<noBlocks2,maxThreadsPerBlock2>>>((int)
                            pow(dimRes,noDims)*noControls,noDims,dimRes,nYears,
                            noControls,ii,jj,k,d_dataPoints,d_xvals,d_yvals,
                            d_regression,d_xmins,d_xmaxes,d_dist,d_ind,
                            d_handles);
                    cudaDeviceSynchronize();

                    cudaFree(d_dist);
                    cudaFree(d_ind);
                }

                free(dataPoints);

                // Recompute forward paths
                optimalForwardPaths<<<noBlocks,noThreadsPerBlock>>>(ii,noPaths,
                        nYears,srp.size(),patches,noControls,noUncertainties,
                        stepSize,unitCost,unitRevenue,rrr,fuelCosts.size(),
                        commodities.size(),dimRes,d_flowRates,d_fuelCosts,
                        d_tempPops,d_totalPops,d_transitions,d_survival,
                        d_speciesParams,d_growthRate,d_capacities,d_aars,
                        d_regression,d_uComposition,d_uResults,d_fuelIdx,
                        d_condExp,d_optCont,d_adjPops,d_unitProfits);
                cudaDeviceSynchronize();

                // Copy the adjusted populations for this time step to the
                // output variables. The conditional expectations, optimal
                // controls and unit profits are copied as well if we are
                // producing the policy map for the optimal road.
                cudaMemcpy(adjPopsF[ii].data(),d_adjPops,srp.size()*noPaths*
                        sizeof(float),cudaMemcpyDeviceToHost);

                cudaFree(d_xvals);
                cudaFree(d_yvals);
                cudaFree(d_dataPoints);
            }

            // Free memory
            destroyHandles<<<1,maxThreadsPerBlock>>>(d_handles,
                    maxThreadsPerBlock);
            cudaFree(d_regression);
            cudaFree(d_xmaxes);
            cudaFree(d_xmins);
            cudaFree(d_adjPops);
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

    // Copy the conditional expectations, optimal controls and unit profits for
    // to the output variables to host memory and then to double precision
    // to the output variables (where needed).
    cudaMemcpy(optCont.data(),d_optCont,optCont.rows()*optCont.cols()*sizeof(
            int),cudaMemcpyDeviceToHost);

    cudaMemcpy(condExpF.data(),d_condExp,condExp.rows()*condExp.cols()*sizeof(
            float),cudaMemcpyDeviceToHost);
    condExp = condExpF.cast<double>();

    cudaMemcpy(unitProfitsF.data(),d_unitProfits,unitProfitsF.rows()*
            unitProfitsF.cols()*sizeof(float),cudaMemcpyDeviceToHost);
    unitProfits = unitProfitsF.cast<double>();

    // Free remaining device memory
    cudaFree(d_unitProfits);
    cudaFree(d_transitions);
    cudaFree(d_survival);
    cudaFree(d_initPops);
    cudaFree(d_tempPops);
    cudaFree(d_capacities);
    cudaFree(d_speciesParams);
    cudaFree(d_fuelCosts);
    cudaFree(d_growthRate);
    cudaFree(d_uResults);
    cudaFree(d_uComposition);
    cudaFree(d_flowRates);
    cudaFree(d_controls);

    cudaFree(d_condExp);
    cudaFree(d_optCont);
    cudaFree(d_fuelIdx);

    // Remove these here?
    cudaFree(d_totalPops);
    cudaFree(d_aars);
    cudaFree(d_mcPops);
}

void SimulateGPU::buildSurrogateROVCUDA(RoadGAPtr op) {

    // Get device properties
    int device = 0;
    struct cudaDeviceProp properties;
    cudaGetDeviceProperties(&properties, device);
    maxMultiProcessors = properties.multiProcessorCount;
    maxThreadsPerBlock = properties.maxThreadsPerBlock;

    // --- CUBLAS initialization
    cublasHandle_t *d_handles;
    cudaMalloc((void**)&d_handles,maxThreadsPerBlock*sizeof(
            cublasHandle_t));
    createHandles<<<1,maxThreadsPerBlock>>>(d_handles,
            maxThreadsPerBlock);
    cudaDeviceSynchronize();

    // Pertinent parameters
    int dimRes = op->getSurrDimRes();
    int noDims = op->getSpecies().size()+1;
    int samples = op->getNoSamples();
    // Use 5% of the nearest points for now
    int k = samples/5;

    // Convert to floating point
    Eigen::VectorXf surrogateF(dimRes*noDims+pow(dimRes,noDims));
    Eigen::VectorXf predictors(noDims*samples);
    Eigen::VectorXf values = op->getValues().segment(0,samples).cast<float>();
    Eigen::VectorXf valuesSD = op->getValuesSD().segment(0,samples)
            .cast<float>();

    // We also need to transpose the data so that the individual observations
    // are in columns
    for (int ii = 0; ii < samples; ii++) {
        predictors.segment(ii*noDims,noDims-1) = op->getIARS().block(ii,0,
                1,noDims-1).cast<float>();
        predictors((ii+1)*noDims-1) = (float)op->getUse()(ii);
    }

    // Call regression kernel for computing the mean and standard
    // deviation
    int noBlocks = (int)((int)pow(dimRes,noDims) % maxThreadsPerBlock) ?
            (int)(pow(dimRes,noDims)/maxThreadsPerBlock + 1) : (int)(
            pow(dimRes,noDims)/maxThreadsPerBlock);
    maxThreadsPerBlock = min((int)pow(dimRes,noDims),maxThreadsPerBlock);

    // Arrange the incoming information
    float *d_xmaxes, *d_xmins, *d_xvals, *d_yvals, *d_surrogate;
    cudaMalloc((void**)&d_xmaxes,noDims*sizeof(float));
    cudaMalloc((void**)&d_xmins,noDims*sizeof(float));

    cudaMalloc((void**)&d_xvals,noDims*samples*sizeof(float));
    cudaMalloc((void**)&d_yvals,samples*sizeof(float));
    cudaMemcpy(d_xvals,predictors.data(),noDims*samples*sizeof(float),
            cudaMemcpyHostToDevice);

    cudaMalloc((void**)&d_surrogate,(dimRes*noDims+pow(dimRes,noDims))*sizeof(
            float));

    // Get the minimum and maximum X value for each dimension for
    // each control.
    computeStateMinMax<<<1,1>>>(1,noDims,samples,&samples,d_xvals,d_xmins,
            d_xmaxes);
    cudaDeviceSynchronize();

    // Allocate the k nearest neighbours for each design point in
    // order to do the regression stage after this. This component is
    // the slowest component.

    // Use 5% of the nearest points for now
    // We first need to perform a k nearest neighbour search
    float *query, *dist;
    int *ind;
    query = (float*)malloc(pow(dimRes,noDims)*noDims*sizeof(float));
    dist = (float*)malloc(pow(dimRes,noDims)*k*sizeof(float));
    ind = (int*)malloc(pow(dimRes,noDims)*k*sizeof(int));

    float *d_queryPts, *d_dist;
    int *d_ind;
    cudaMalloc((void**)&d_queryPts,pow(dimRes,noDims)*noDims*sizeof(float));
    cudaMalloc((void**)&d_dist,pow(dimRes,noDims)*k*sizeof(float));
    cudaMalloc((void**)&d_ind,pow(dimRes,noDims)*k*sizeof(int));

    createQueryPoints<<<noBlocks,maxThreadsPerBlock>>>((int)pow(dimRes,noDims),
            noDims,dimRes,0,1,0,d_xmins,d_xmaxes,d_surrogate,d_queryPts);
    cudaMemcpy(query,d_queryPts,pow(dimRes,noDims)*noDims*sizeof(float),
            cudaMemcpyDeviceToHost);
    cudaFree(d_queryPts);

    // Compute the knn searches
    knn_cublas_with_indexes::knn(predictors.data(),samples,query,pow(
            dimRes,noDims)*noDims,noDims,k,dist,ind);

    cudaMemcpy(d_dist,dist,pow(dimRes,noDims)*k*sizeof(float),
            cudaMemcpyHostToDevice);

    cudaMemcpy(d_ind,ind,pow(dimRes,noDims)*k*sizeof(float),
            cudaMemcpyHostToDevice);

    free(query);
    free(dist);
    free(ind);

    // Mean ///////////////////////////////////////////////////////////////////
    cudaMemcpy(d_yvals,values.data(),samples*sizeof(float),
            cudaMemcpyHostToDevice);

    multiLocLinReg<<<noBlocks,maxThreadsPerBlock>>>(samples,noDims,
            dimRes,1,1,0,0,k,&samples,predictors.data(),values.data(),
            d_surrogate,d_xmins,d_xmaxes,d_dist,d_ind,d_handles);
    cudaDeviceSynchronize();

    cudaMemcpy(surrogateF.data(),d_surrogate,dimRes*noDims+pow(dimRes,noDims)*
            sizeof(float),cudaMemcpyDeviceToHost);

    // Save the surrogate to the RoadGA object
    op->getSurrogateROV()[2*op->getScenario()->getCurrentScenario()][
            op->getScenario()->getRun()] = surrogateF
            .cast<double>();

    // Standard deviation /////////////////////////////////////////////////////
    cudaMemcpy(d_yvals,valuesSD.data(),samples*sizeof(float),
            cudaMemcpyHostToDevice);

    multiLocLinReg<<<noBlocks,maxThreadsPerBlock>>>(samples,noDims,
            dimRes,1,1,0,0,k,&samples,predictors.data(),valuesSD.data(),
            d_surrogate,d_xmins,d_xmaxes,d_dist,d_ind,d_handles);
    cudaDeviceSynchronize();

    cudaMemcpy(surrogateF.data(),d_surrogate,dimRes*noDims+pow(dimRes,noDims)*
            sizeof(float),cudaMemcpyDeviceToHost);

    // Save the surrogate to the RoadGA object
    op->getSurrogateROV()[2*op->getScenario()->getCurrentScenario()+1][
            op->getScenario()->getRun()] = surrogateF.cast<double>();

    // Free remaining memory
    destroyHandles<<<1,maxThreadsPerBlock>>>(d_handles,
            maxThreadsPerBlock);
    cudaFree(d_xmaxes);
    cudaFree(d_xmins);
    cudaFree(d_xvals);
    cudaFree(d_yvals);
    cudaFree(d_surrogate);
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
