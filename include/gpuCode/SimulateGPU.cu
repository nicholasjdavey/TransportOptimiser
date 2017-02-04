#include <stdio.h>
#include <curand.h>
#include <curand_kernel.h>
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

// The rov kernel represents a single path for rov
__global__ void rovKernel() {
      printf("Hello from mykernel\n");
}

// WRAPPERS ///////////////////////////////////////////////////////////////////

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
        std::vector<HabitatPatchPtr>& patches, double& initPop, int&
        noPatches) {

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

    int noBlocks = (xres*yres*noRegions % maxThreadsPerBlock)? (int)(xres*yres*
            noRegions/maxThreadsPerBlock +1) : (int)(xres*yres*noRegions/
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

    // Get the important values for the road first and convert them to
    // formats that the kernel can use

    for (int ii = 0; ii < srp.size(); ii++) {

        // Species parameters
        float grm = (float)srp[ii]->getSpecies()->getGrowthRateMean();
        float grsd = (float)srp[ii]->getSpecies()->getGrowthRateSD();
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
                (noPaths/maxThreadsPerBlock + 1) :
                (noPaths/maxThreadsPerBlock);
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

void SimulateGPU::simulateROVCUDA() {
    // Get device properties
    int device = 0;
    struct cudaDeviceProp properties;
    cudaGetDeviceProperties(&properties, device);
    maxMultiProcessors = properties.multiProcessorCount;
    maxThreadsPerBlock = properties.maxThreadsPerBlock;
}
