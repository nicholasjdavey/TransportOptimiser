#include <stdio.h>
#include <cuda.h>
#include <curand.h>
#include <curand_kernel.h>
#include <cusolverDn.h>
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include "../transportbase.h"

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

//        printf("%4d, %5d, %8.0f, %5.0f, %5.0f, %5.0f, %5.0f\n",idx,blockSizeX,
//                results[5*idx],results[5*idx+1],results[5*idx+2],
//                results[5*idx+3],results[5*idx+4]);
    }
}

// The mte kernel represents a single path for mte
__global__ void mteKernel(int noPaths, int nYears, int noPatches, float grm,
        float grsd, float *initPops, float* caps, float* mmm, float* eps,
        float* drf) {
    // Global index for finding the thread number
    int ii = blockIdx.x*blockDim.x + threadIdx.x;

    // Only perform matrix multiplication sequentially for now. Later, if
    // so desired, we can use dynamic parallelism because the card in the
    // machine has CUDA compute capability 3.5
    if (ii < noPaths) {
        // Initialise the temporary vector
        float *pops;
        pops = (float*)malloc(noPatches*sizeof(float));
        float *popsOld;
        popsOld = (float*)malloc(noPatches*sizeof(float));

        // Initialise the prevailing population vector
        for (int jj = 0; jj < noPatches; jj ++) {
            pops[jj] = 1.0f;
            popsOld[jj] = initPops[jj];
        }

        for (int jj = 0; jj < nYears; jj++) {
            // Movement and mortality
            for (int kk = 0; kk < noPatches; kk++) {
                pops[kk] = 0.0;
                for (int ll = 0; ll < noPatches; ll++) {
                    pops[kk] += popsOld[ll]*mmm[kk*noPatches+ll];
                }
            }

            // Natural birth and death
            for (int kk = 0; kk < noPatches; kk++) {
                float gr = grsd*drf[ii*(nYears*noPatches) + jj*noPatches + kk]
                        + grm;
                popsOld[kk] = pops[kk]*(1.0f + gr*(caps[kk]-pops[kk])/caps[kk]/
                        100.0);
            }
        }

        eps[ii] = 0.0f;
        for (int jj = 0; jj < noPatches; jj++) {
            eps[ii] += popsOld[jj];
        }
        free(pops);
        free(popsOld);
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
        noSpecies, int noPatches, int noControls, int noFuels,
        int noUncertainties, float timeStep, float* pops, float* transitions,
        float* survival, float* speciesParams, float* rgr, float* caps, float*
        aars, float* uncertParams, int* controls, float* uJumps, float*
        uBrownian, float* uJumpSizes, float* uResults, float* totalPops) {

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
                    // Population growth
                    float gr = speciesParams[jj*3]*rgr[idx*noSpecies*noPatches*
                            nYears + ii*noSpecies*noPatches + jj*noPatches +
                            kk] + speciesParams[jj*3+1];
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
    }
}

__global__ void allocateXYRegressionData(int noPaths, int noControls, int*
        controls, int *dataPoints, float *condExp, float *xvals, float
        *yvals) {

    for (int ii = 0; ii < noControls; ii++) {
        dataPoints[ii] = 0;
    }

    for (int ii = 0; ii < noPaths; ii++) {
    }
}

__global__ void optimalForwardPaths(int start, int noPaths, int nYears, int
        noSpecies, int noPatches, int noControls, int noUncertainties, float
        timeStep, float* pops, float* totalPops, float* transitions, float*
        survival, float* speciesParams, float* rgr, float* caps, float* aars,
        float* uncertParams, float* regression, float* uJumps, float*
        uBrownian, float* uJumpSizes, float* uResults, int* fuelIdx) {

    // Global thread index
    int idx = blockIdx.x*blockDim.x + threadIdx.x;

    // We do not need to recompute the exogenous uncertainties (everything
    // else other than animal populations)
    if (idx < noPaths) {
        if (start == nYears) {
            // At the last period we run the road if the adjusted population
            // for a particular control (pop-pop*aar_jj) is greater than the
            // minimum permissible population. This becomes the optimal
            // payoff, which we then regress onto the adjusted population to
            // determine the expected payoff at time T given a prevailing end
            // adjusted population. Everything is considered deterministic at
            // this stage.
            float* payoffs;
            payoffs = (float*)malloc(noControls*sizeof(float));
            bool* valid;
            valid = (bool*)malloc(noControls*sizeof(float));

            for (int ii = 0; ii < noControls; ii++) {
                // Compute the single period financial payoff for each control
                // for this period and the adjusted profit. If the adjusted
                // profit is above the threshold, then the payoff is invalid.
                valid[ii] = true;
                for (int jj = 0; jj < noSpecies; jj++) {
                    float adjPop = totalPops[start*noSpecies*noPaths + idx*
                            noSpecies + jj]*aars[jj*nYears*noPaths*noControls
                            + idx*noControls + ii];

                    if (adjPop < speciesParams[noSpecies*jj + 2]) {
                        valid = false;
                        break;
                    }
                }

                // Compute the payoff for the control if valid
                if (valid) {

                }
            }
        } else {

        }
    }
}

// Multiple global linear regression
__global__ void multiLinReg() {

}

// Multiple local linear regression
__global__ void multiLocLinReg(int noPoints, int noDims, int dimRes, int nYears,
        int noControls, int year, int control, int* dataPoints, float *xvals,
        float *yvals, float *d_regression) {

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

        // 1. First find the k nearest neighbours to the query point
        // Epanechnikov kernel

        // Gaussian kernel

        // N-nearest neighbour kernel (this is slowest)

        // Fixed window width

        // 2. Second, build the matrices used in the calculation
        // A - Input design matrix
        // B - Input known matrix
        // C - Output matrix of coefficients
        float *A, *B, *C;

        A = (float*)malloc(pow(noDims+1,2)*sizeof(float));
        B = (float*)malloc((noDims+1)*sizeof(float));
        C = (float*)malloc((noDims+1)*sizeof(float));

        // Initialise values to zero
        for (int ii = 0; ii <= noDims; ii++) {
            B[ii] = 0.0;

            for (int kk = 0; kk < dataPoints[control]; kk++) {
                B[ii] += yvals[ii]*(xvals[kk*noDims+ii] - d_regression[year*
                        noControls*(dimRes*noDims + (int)pow(dimRes,noDims)*2)
                        + control*(dimRes*noDims + (int)pow(dimRes,noDims)*2)
                        + ii*dimRes + dimIdx[ii]]);
            }

            for (int jj = 0; jj <= noDims; jj++) {
                A[ii*(noDims+1)+jj] = 0;

                for (int kk = 0; kk <= dataPoints[control]; kk++) {
                    A[ii*(noDims+1)+jj] += (xvals[kk*noDims+ii] - d_regression[
                            year*noControls*(dimRes*noDims + (int)pow(dimRes,
                            noDims)*2) + control*(dimRes*noDims + (int)pow(
                            dimRes,noDims)*2) + ii*dimRes + dimIdx[ii]])*(
                            xvals[kk*noDims+jj] - d_regression[year*noControls*
                            (dimRes*noDims + (int)pow(dimRes,noDims)*2) +
                            control*(dimRes*noDims + (int)pow(dimRes,noDims)*2)
                            + jj*dimRes + dimIdx[jj]]);
                }
            }
        }

        // 3. Use cuBlas to solve the set of linear equations to determine the
        // coefficients matrix.
        // --- CUBLAS initialization
        cublasHandle_t cublas_handle;
        cublasStatus_t status = cublasCreate_v2(&cublas_handle);

        // a. LU Decomposition
        // --- Creating the array of pointers needed as input/output to the
        // batched getrf
        float **inout_pointers = (float **)malloc(sizeof(float*));
        inout_pointers[0] = A;

        int *pivotArray;
        int *infoArray;
        cudaMalloc((void**)&pivotArray,(noDims+1)*sizeof(int));
        cudaMalloc((void**)&infoArray,sizeof(int));
        cublasSgetrfBatched(cublas_handle,noDims+1,inout_pointers,noDims,
                pivotArray,infoArray,1);

        // 4. Save the coefficients to the regression matrix

        // Free memory
        free(A);
        free(B);
    }
    free(dimIdx);
}


// WRAPPERS ///////////////////////////////////////////////////////////////////

void SimulateGPU::expPV(UncertaintyPtr uncertainty) {
    // Get device properties
    int device = 0;
    struct cudaDeviceProp properties;
    cudaGetDeviceProperties(&properties, device);
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
    cudaMalloc((void **)&d_brownian, sizeof(float)*nYears*noPaths);
    cudaMalloc((void **)&d_jumpSizes, sizeof(float)*nYears*noPaths);
    cudaMalloc((void **)&d_jumps, sizeof(float)*nYears*noPaths);
    cudaMalloc((void **)&d_results, sizeof(float)*nYears*noPaths);

    curandCreateGenerator(&gen, CURAND_RNG_PSEUDO_DEFAULT);
    curandSetPseudoRandomGeneratorSeed(gen, _seed);

    curandGenerateNormal(gen, d_brownian, nYears*noPaths, 0.0f, uncertainty->
            getNoiseSD()*timeStep*vp->getCommoditySDMultipliers()(
            sc->getCommoditySD()));
    curandGenerateNormal(gen, d_jumpSizes, nYears*noPaths,
            -pow(uncertainty->getPoissonJump()*vp->getCommoditySDMultipliers()(
            sc->getCommoditySD()),2)/2,pow(uncertainty->
            getPoissonJump()*vp->getCommoditySDMultipliers()(sc->getCommoditySD()
            ),2));
    curandGenerateUniform(gen, d_jumps, nYears*noPaths);

    curandDestroyGenerator(gen);
    cudaDeviceSynchronize();

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

    cudaDeviceSynchronize();

    cudaMemcpy(results,d_results,noPaths*sizeof(float),cudaMemcpyDeviceToHost);

    for (int ii = 0; ii < noPaths; ii++) {
        total += results[ii];
    }

    uncertainty->setExpPV((double)total/((double)noPaths));

    total = 0.0;
    for (int ii = 0; ii < noPaths; ii++) {
        total += pow(results[ii] - uncertainty->getExpPV(),2);
    }

    uncertainty->setExpPVSD(sqrt(total));

    cudaFree(d_brownian);
    cudaFree(d_jumpSizes);
    cudaFree(d_jumps);
    cudaFree(d_results);
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

    cudaMalloc(&d_A,a*b*sizeof(float));
    cudaMemcpy(d_A,Af,a*b*sizeof(float),cudaMemcpyHostToDevice);

    cudaMalloc(&d_B,c*d*sizeof(float));
    cudaMemcpy(d_B,Bf,c*d*sizeof(float),cudaMemcpyHostToDevice);

    cudaMalloc(&d_C,a*d*sizeof(float));

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

    // Retrieve result and free data
    cudaMemcpy(C.data(),d_C,a*d*sizeof(float),cudaMemcpyDeviceToHost);

    cudaDeviceSynchronize();
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

    cudaMalloc(&d_A,a*b*sizeof(float));
    cudaMemcpy(d_A,Af.data(),a*b*sizeof(float),cudaMemcpyHostToDevice);

    cudaMalloc(&d_B,c*d*sizeof(float));
    cudaMemcpy(d_B,Bf.data(),c*d*sizeof(float),cudaMemcpyHostToDevice);

    cudaMalloc(&d_C,a*d*sizeof(float));

    // declare the number of blocks per grid and the number of threads per block
    dim3 threads(BLOCK_SIZE,VECTOR_SIZE);
    dim3 grid(d/(BLOCK_SIZE*VECTOR_SIZE), a/BLOCK_SIZE);

    matrixMultiplicationKernel<<<grid,threads>>>(d_A,d_B,d_C,a,b,d);

    // Retrieve result and free data
    cudaMemcpy(Cf.data(),d_C,a*d*sizeof(float),cudaMemcpyDeviceToHost);

    C = Cf.cast<double>();

    cudaDeviceSynchronize();
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);

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

    cudaMalloc(&d_A,a*b*sizeof(float));
    cudaMemcpy(d_A,Af.data(),a*b*sizeof(float),cudaMemcpyHostToDevice);

    cudaMalloc(&d_B,a*b*sizeof(float));
    cudaMemcpy(d_B,Bf.data(),a*b*sizeof(float),cudaMemcpyHostToDevice);

    cudaMalloc(&d_C,a*b*sizeof(float));

    // declare the number of blocks per grid and the number of threads per
    // block
    dim3 dimBlock(32,32);
    dim3 dimGrid(b/dimBlock.x,a/dimBlock.y);

    matrixMultiplicationKernelEW<<<dimGrid,dimBlock>>>(d_A,d_B,d_C,a,b);

    // Retrieve result and free data
    cudaMemcpy(Cf.data(),d_C,a*b*sizeof(float),cudaMemcpyDeviceToHost);

    C = Cf.cast<double>();

    cudaDeviceSynchronize();
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
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

    cudaMalloc(&d_A,a*b*sizeof(float));
    cudaMemcpy(d_A,Af.data(),a*b*sizeof(float),cudaMemcpyHostToDevice);

    cudaMalloc(&d_B,a*b*sizeof(float));
    cudaMemcpy(d_B,Bf.data(),a*b*sizeof(float),cudaMemcpyHostToDevice);

    cudaMalloc(&d_C,a*b*sizeof(float));

    // declare the number of blocks per grid and the number of threads per
    // block
    dim3 dimBlock(32,32);
    dim3 dimGrid(b/dimBlock.x,a/dimBlock.y);

    matrixDivisionKernelEW<<<dimGrid,dimBlock>>>(d_A,d_B,d_C,a,b);

    // Retrieve result and free data
    cudaMemcpy(Cf.data(),d_C,a*b*sizeof(float),cudaMemcpyDeviceToHost);

    C = Cf.cast<double>();

    cudaDeviceSynchronize();
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
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

    cudaMalloc(&d_XY1,XY1.rows()*XY1.cols()*sizeof(float));
    cudaMemcpy(d_XY1,XY1f.data(),XY1.rows()*XY1.cols()*sizeof(float),
            cudaMemcpyHostToDevice);
    cudaMalloc(&d_XY2,XY2.rows()*XY2.cols()*sizeof(float));
    cudaMemcpy(d_XY2,XY2f.data(),XY2.rows()*XY2.cols()*sizeof(float),
            cudaMemcpyHostToDevice);

    cudaMalloc(&d_X4_X3,XY2.rows()*sizeof(float));
    cudaMemcpy(d_X4_X3,X4_X3.data(),XY2.rows()*sizeof(float),
            cudaMemcpyHostToDevice);
    cudaMalloc(&d_Y4_Y3,XY2.rows()*sizeof(float));
    cudaMemcpy(d_Y4_Y3,Y4_Y3.data(),XY2.rows()*sizeof(float),
            cudaMemcpyHostToDevice);
    cudaMalloc(&d_X2_X1,XY1.rows()*sizeof(float));
    cudaMemcpy(d_X2_X1,X2_X1.data(),XY1.rows()*sizeof(float),
            cudaMemcpyHostToDevice);
    cudaMalloc(&d_Y2_Y1,XY1.rows()*sizeof(float));
    cudaMemcpy(d_Y2_Y1,Y2_Y1.data(),XY1.rows()*sizeof(float),
            cudaMemcpyHostToDevice);
    cudaMalloc(&d_adjacency,XY1.rows()*XY2.rows()*sizeof(int));
    cudaMalloc(&d_cross,XY1.rows()*sizeof(int));

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
    cudaDeviceSynchronize();

    cudaError_t error = cudaGetLastError();
    if (error != cudaSuccess) {
      fprintf(stderr, "ERROR: %s \n", cudaGetErrorString(error));
    }

    // Sum the number
    noBlocks = (XY1.rows() % maxThreadsPerBlock)? (int)(XY1.rows()/
            maxThreadsPerBlock + 1) : (int)(XY1.rows()/maxThreadsPerBlock);
    roadCrossingsKernel<<<noBlocks,maxThreadsPerBlock>>>(XY1.rows(),
            XY2.rows(),d_adjacency,d_cross);
    cudaDeviceSynchronize();

    // Retrieve results
    cudaMemcpy(crossings.data(),d_cross,XY1.rows()*sizeof(int),
            cudaMemcpyDeviceToHost);

    cudaDeviceSynchronize();
    // Free memory
    cudaFree(d_X4_X3);
    cudaFree(d_Y4_Y3);
    cudaFree(d_X2_X1);
    cudaFree(d_Y2_Y1);
    cudaFree(d_cross);
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
    cudaMalloc((void **)&d_results,xres*yres*noRegions*5*sizeof(float));

    cudaMalloc((void **)&d_labelledImage,H*W*sizeof(int));
    cudaMemcpy(d_labelledImage,labelledImage.data(),H*W*sizeof(int),
            cudaMemcpyHostToDevice);

    cudaMalloc((void **)&d_populations,H*W*sizeof(float));
    cudaMemcpy(d_populations,popsFloat.data(),H*W*sizeof(float),
            cudaMemcpyHostToDevice);

    int noBlocks = ((xres*yres*noRegions) % maxThreadsPerBlock)? (int)(xres*
            yres*noRegions/maxThreadsPerBlock + 1) : (int)(xres*yres*noRegions/
            maxThreadsPerBlock);
    int noThreadsPerBlock = min(maxThreadsPerBlock,xres*yres*noRegions);

    patchComputation<<<noBlocks,noThreadsPerBlock>>>(xres*yres*noRegions,
            W, H, skpx, skpy, xres,yres,(float)subPatchArea,(float)xspacing,
            (float)yspacing,(float)habTyp->getMaxPop(),noRegions,
            d_labelledImage,d_populations,d_results);
    cudaDeviceSynchronize();

    cudaError_t error = cudaGetLastError();
    if (error != cudaSuccess) {
      fprintf(stderr, "ERROR: %s \n", cudaGetErrorString(error));
    }

    cudaMemcpy(results,d_results,xres*yres*noRegions*5*sizeof(float),
               cudaMemcpyDeviceToHost);

    // Now turn the results into patches
    for (int ii = 0; ii < xres*yres*noRegions; ii++) {
        if (results[5*ii] > 0) {
            // Create new patch to add to patches vector
            HabitatPatchPtr hab(new HabitatPatch());
            hab->setArea((double)results[5*ii]);
            hab->setCX((double)results[ii+3]);
            hab->setCY((double)results[ii+4]);
            hab->setPopulation((double)results[5*ii+2]);
            hab->setCapacity((double)results[5*ii+1]);
            initPop += (double)results[5*ii];
            initPops(noPatches) = (double)results[5*ii];
            patches[noPatches++] = hab;
        }
    }

    cudaFree(d_populations);
    cudaFree(d_labelledImage);
    cudaFree(d_results);
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
    ExperimentalScenarioPtr sc = sim->getRoad()->getOptimiser()->getScenario();
    VariableParametersPtr vp = sim->getRoad()->getOptimiser()->
            getVariableParams();

    // Get the important values for the road first and convert them to
    // formats that the kernel can use

    for (int ii = 0; ii < srp.size(); ii++) {

        // Species parameters
        float stepSize = (float)sim->getRoad()->getOptimiser()->getEconomic()->
                getTimeStep();
        float grm = (float)(srp[ii]->getSpecies()->getGrowthRateMean()*
                stepSize*vp->getGrowthRatesMultipliers()(sc->getPopGR()));
        float grsd = (float)(srp[ii]->getSpecies()->getGrowthRateSD()*
                stepSize*vp->getGrowthRateSDMultipliers()(sc->getPopGRSD()));
        int nPatches = capacities[ii].size();

        float *eps, *d_initPops, *d_eps, *d_caps, *d_mmm;

        // RANDOM MATRIX
        float *d_random_floats;
        curandGenerator_t gen;
        srand(time(NULL));
        int _seed = rand();
        //allocate space for 100 floats on the GPU
        //could also do this with thrust vectors and pass a raw pointer
        cudaMalloc((void **)&d_random_floats, sizeof(float) *nYears*noPaths*
                nPatches);
        curandCreateGenerator(&gen, CURAND_RNG_PSEUDO_DEFAULT);
        curandSetPseudoRandomGeneratorSeed(gen, _seed);
        curandGenerateNormal(gen, d_random_floats, nYears*noPaths*nPatches,
                0.0f,1.0f);
        curandDestroyGenerator(gen);
        cudaDeviceSynchronize();

        // INITIAL POPULATIONS
        Eigen::MatrixXf initPopsF = initPops[ii].cast<float>();
        cudaMalloc(&d_initPops,initPops[ii].size()*sizeof(float));
        cudaMemcpy(d_initPops,initPopsF.data(),
                initPops[ii].size()*sizeof(float),cudaMemcpyHostToDevice);

        // END POPULATIONS
        eps = (float*)malloc(noPaths*sizeof(float));
        cudaMalloc(&d_eps, noPaths*sizeof(float));

        for (int jj = 0; jj < noPaths; jj++) {
            eps[jj] = 0.0f;
        }

        cudaMemcpy(d_eps,eps,noPaths*sizeof(float),cudaMemcpyHostToDevice);

        // CAPACITIES
        Eigen::VectorXf capsF = capacities[ii].cast<float>();
        cudaMalloc(&d_caps,capacities[ii].size()*sizeof(float));
        cudaMemcpy(d_caps,capsF.data(),capacities[ii].size()*sizeof(float),
                cudaMemcpyHostToDevice);

        // MOVEMENT AND MORTALITY MATRIX
        // We use the highest flow rate in the vector of survival matrices
        const Eigen::MatrixXd& transProbs = srp[ii]->getTransProbs();
        const Eigen::MatrixXd& survProbs = srp[ii]->getSurvivalProbs()[
                srp[ii]->getSurvivalProbs().size()-1];
        Eigen::MatrixXf mmm = (transProbs.array()*survProbs.array()).
                cast<float>();

        cudaMalloc(&d_mmm,mmm.rows()*mmm.cols()*sizeof(float));
        cudaMemcpy(d_mmm,mmm.data(),mmm.rows()*mmm.cols()*
                sizeof(float),cudaMemcpyHostToDevice);

        ///////////////////////////////////////////////////////////////////////
        // Perform N simulation paths. Currently, there is no species
        // interaction, so we run each kernel separately and do not need to use
        // the Thrust library.
        int noBlocks = (int)(noPaths % maxThreadsPerBlock)?
                (int)(noPaths/maxThreadsPerBlock + 1) :
                (int)(noPaths/maxThreadsPerBlock);
        int noThreadsPerBlock = min(noPaths,maxThreadsPerBlock);

        mteKernel<<<noBlocks,noThreadsPerBlock>>>
                (noPaths,nYears,capacities[ii].size(),grm,grsd,d_initPops,
                d_caps,d_mmm,d_eps,d_random_floats);
        cudaDeviceSynchronize();

        // Retrieve results
        cudaMemcpy(eps,d_eps,srp.size()*sizeof(float),cudaMemcpyDeviceToHost);

        for (int jj = 0; jj < noPaths; jj++) {
            endPops(jj,ii) = eps[jj];
        }

        // Free memory
        cudaDeviceSynchronize();
        cudaFree(d_random_floats);
        cudaFree(d_initPops);
        cudaFree(d_eps);
        cudaFree(d_caps);
        cudaFree(d_mmm);
        free(eps);
    }
}

void SimulateGPU::simulateROVCUDA(SimulatorPtr sim,
        std::vector<SpeciesRoadPatchesPtr>& srp,
        std::vector<std::vector<Eigen::MatrixXd> > &aars,
        std::vector<Eigen::MatrixXd> &totalPops, Eigen::MatrixXd& condExp,
        Eigen::MatrixXi& optCont) {
    // Currently there is no species interaction. This can be a future question
    // and would be an interesting extension on how it can be implemented,
    // what the surrogate looks like and how the patches are formed.

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

    double unitProfit = sim->getRoad()->getAttributes()->getUnitVarRevenue();
    double unitCost = sim->getRoad()->getAttributes()->getUnitVarCosts();
    double stepSize = optimiser->getEconomic()->getTimeStep();

    // Get the important values for the road first and convert them to formats
    // that the kernel can use

    // Initialise CUDA memory /////////////////////////////////////////////////

    // 1. Transition and survival matrices for each species and each control
    float *transitions, *survival, *initPops, *capacities, *speciesParams,
            *uncertParams, *d_transitions, *d_survival, *d_initPops,
            *d_tempPops, *d_capacities, *d_speciesParams, *d_uncertParams;

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
    speciesParams = (float*)malloc(srp.size()*2*sizeof(float));
    uncertParams = (float*)malloc(noUncertainties*8*sizeof(float));

    cudaMalloc((void**)&d_noPatches,srp.size()*sizeof(int));
    cudaMalloc((void**)&d_initPops,patches*sizeof(float));
    cudaMalloc((void**)&d_capacities,patches*sizeof(float));
    cudaMalloc((void**)&d_transitions,transition*sizeof(float));
    cudaMalloc((void**)&d_survival,transition*noControls*sizeof(float));
    cudaMalloc((void**)&d_speciesParams,srp.size()*3*sizeof(float));
    cudaMalloc((void**)&d_uncertParams,noUncertainties*6*sizeof(float));
    cudaMalloc((void**)&d_tempPops,noPaths*nYears*patches*srp.size()*
            sizeof(float));

    int counter1 = 0;
    int counter2 = 0;
    int counter3 = 0;

    // Read in the information into the correct format
    for (int ii = 0; ii < srp.size(); ii++) {
        memcpy(initPops+counter1,srp[ii]->getInitPops().data(),
                srp[ii]->getHabPatches().size()*sizeof(float));
        memcpy(capacities+counter1,srp[ii]->getCapacities().data(),
                srp[ii]->getHabPatches().size()*sizeof(float));

        speciesParams[counter1] = srp[ii]->getSpecies()->getGrowthRateMean()*
                varParams->getGrowthRatesMultipliers()(scenario->getPopGR());
        speciesParams[counter1+1] = srp[ii]->getSpecies()->getGrowthRateSD()*
                varParams->getGrowthRateSDMultipliers()(scenario->getPopGRSD());
        speciesParams[counter1+2] = srp[ii]->getSpecies()->getThreshold()*
                varParams->getPopulationLevels()(scenario->getPopLevel());

        counter1 += 3;

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
    cudaMemcpy(d_speciesParams,speciesParams,srp.size()*3*sizeof(float),
            cudaMemcpyHostToDevice);
    cudaMemcpy(d_uncertParams,uncertParams,noUncertainties*6*sizeof(float),
            cudaMemcpyHostToDevice);

    // Exogenous parameters (fuels, commodities)
    // Ore composition is simply Gaussian for now
    float *d_randCont, *d_growthRate, *d_uBrownian, *d_uJumpSizes,
            *d_uJumps, *d_uResults, *d_uComposition;
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

    // We no longer need the floating point random controls vector
    cudaFree(d_randCont);

    // Endogenous uncertainty
    cudaMalloc((void**)&d_growthRate,nYears*noPaths*patches*srp.size()*
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
            d_growthRate,d_capacities,d_aars,d_uncertParams,d_controls,
            d_uJumps,d_uBrownian,d_uJumpSizes,d_uResults,d_totalPops);
    cudaDeviceSynchronize();

    // Determine the number of uncertainties. The uncertainties are the unit
    // payoff of the road (comprised of commodity and fuel prices, which are
    // pre-processed to determine covariances etc. and are treated as a single
    // uncertainty) and the adjusted population of each species under each
    // control.
    int noDims = srp.size() + 1;

    // Prepare the floating point versions of the output
    float *d_condExp;
    int *d_fuelIdx;

    cudaMalloc(&d_fuelIdx,fuelIdx.size()*sizeof(int));
    cudaMemcpy(d_fuelIdx,fuelIdx.data(),fuelIdx.size()*sizeof(int),
            cudaMemcpyHostToDevice);

    Eigen::MatrixXf condExpF(condExp.rows(),condExp.cols());
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
            // time step mapped against the N-dimensional grid.
            float *d_regression;
            int dimRes = sim->getRoad()->getOptimiser()->getOtherInputs()->
                    getDimRes();

            cudaMalloc((void**)&d_regression,nYears*noControls*(dimRes*noDims +
                    pow(dimRes,noDims)*2)*sizeof(float));

            // The first backward step is simply the single period payoff for
            // each control

            // For each further backward step
            for (int ii = nYears - 1; ii >= 0; ii--) {
                // Perform regression and save results
                int noBlocks2 = (int)((int)pow(dimRes,noDims)*noControls %
                        maxThreadsPerBlock) ? (int)(pow(dimRes,noDims)*
                        noControls/maxThreadsPerBlock + 1) : (int)(
                        pow(dimRes,noDims)*noControls/maxThreadsPerBlock);
                int maxThreadsPerBlock2 = min((int)pow(dimRes,noDims)*
                        noControls,maxThreadsPerBlock);

                // Build the regression data for each control on the device
                float *d_xvals, *d_yvals;
                int* d_dataPoints;
                cudaMalloc((void**)&d_xvals,noDims*noPaths*sizeof(float));
                cudaMalloc((void**)&d_yvals,noPaths*sizeof(float));
                cudaMalloc((void**)&d_dataPoints,noControls*sizeof(int));

                // This kernel does not take advantage of massive parallelism.
                // It is simply to allow us to call data that is already on
                // the device.
                allocateXYRegressionData<<<1,1>>>(noPaths, noControls,
                        d_controls, d_dataPoints, d_condExp, d_xvals, d_yvals);

                for (int jj = 0; jj < noControls; jj++) {
                    // Perform the regression for this control at this time
                    multiLocLinReg<<<noBlocks2,maxThreadsPerBlock2>>>((int)
                            pow(dimRes,noDims)*noControls,noDims,dimRes,nYears,
                            noControls,ii,jj,d_dataPoints,d_xvals,d_yvals,
                            d_regression);
                    cudaDeviceSynchronize();
                }

                // Recompute forward paths
                optimalForwardPaths<<<noBlocks,noThreadsPerBlock>>>(ii,noPaths,
                        nYears,srp.size(),patches,noControls,noUncertainties,
                        stepSize,d_tempPops,d_totalPops,d_transitions,
                        d_survival,d_speciesParams,d_growthRate,d_capacities,
                        d_aars,d_uncertParams,d_regression,d_uJumps,
                        d_uBrownian,d_uJumpSizes,d_uResults,d_fuelIdx);
                cudaDeviceSynchronize();

                cudaFree(d_xvals);
                cudaFree(d_yvals);
                cudaFree(d_dataPoints);
            }

            // At the final initial time step, take an average of the path
            // values

            // Free memory
            cudaFree(d_regression);
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

    // Copy device results to host memory
    cudaMemcpy(optCont.data(),d_optCont,optCont.rows()*optCont.cols()*sizeof(
            int),cudaMemcpyDeviceToHost);
    cudaMemcpy(condExpF.data(),d_condExp,condExp.rows()*condExp.cols()*sizeof(
            float),cudaMemcpyDeviceToHost);
    condExp = condExpF.cast<double>();

    // Free remaining device memory
    cudaFree(d_condExp);
    cudaFree(d_optCont);
    cudaFree(d_fuelIdx);
}
