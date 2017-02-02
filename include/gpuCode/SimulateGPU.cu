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
__global__ void matrixMultiplicationKernel(float* A, float* B, float* C, int a,
        int b, int c, int d) {

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

// The patch kernel represents a single cell for generating habitat patches
// The results matrix contains the following:
//
__global__ void patchComputation(int W, int H, int skpx, int skpy, int xres,
        int yres, float subPatchArea, float xspacing, float yspacing, float
        capacity, int uniqueRegions, int* labelledImage, float* pops,
        float* results) {

    // Get global index of thread
    int idx = blockIdx.x*blockDim.x + threadIdx.x;

    // Get large grid cell subscripts of thread
    int blockIdxY = (int)(((int)(idx/uniqueRegions))/xres);
    int blockIdxX = (int)(idx/uniqueRegions) - blockIdxY*xres;
    int regionNo = idx - blockIdxY*xres*yres -blockIdxX*yres;

    int blockSizeX;
    int blockSizeY;

    if ((blockIdxX+1)*skpx <= H) {
        blockSizeX = skpx;
    } else {
        blockSizeX = H-blockIdx.x*skpx;
    }

    if ((blockIdxY+1)*skpy <= W) {
        blockSizeY = skpy;
    } else {
        blockSizeY = W-blockIdx.y*skpy;
    }

    // Iterate through each region for this large grid cell
    float area = 0.0f;
    float cap = 0.0f;
    float pop = 0.0f;
    float cx = 0.0f;
    float cy = 0.0f;

    for (int ii = 0; ii < blockSizeX; ii++) {
        for (int jj = 0; jj < blockSizeY; jj++) {
            int subIdx = blockIdxY*xres*skpx*skpy + blockIdxX*skpx*skpy
                    + jj*(H - blockIdxX*skpx) + ii;
            area += (float)(labelledImage[subIdx] == regionNo);
        }
    }

    if (area > 0) {
        for (int ii = 0; ii < blockSizeX; ii++) {
            for (int jj = 0; jj < blockSizeY; jj++) {
                int subIdx = blockIdxY*xres*skpx*skpy + blockIdxX*skpx*skpy
                        + jj*(H - blockIdxX*skpx) + ii;
                pop += pops[subIdx];
                cx += jj*(float)(labelledImage[subIdx] == regionNo);
                cy += ii*(float)(labelledImage[subIdx] == regionNo);
            }
        }
    }

    // Store results to output matrix
    results[5*idx+2] = pop;
    results[5*idx+3] = xspacing*(cx/area + blockIdxX*skpx);
    results[5*idx+4] = yspacing*(cy/area + blockIdxY*skpy);

    cap = area*capacity;
    area = area*subPatchArea;
    results[5*idx] = area;
    results[5*idx+1] = cap;
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

void SimulateGPU::eigenMatrixMult(const Eigen::MatrixXf& A, const
        Eigen::MatrixXf& B, Eigen::MatrixXf& C) {

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

    cudaMalloc(&d_A,a*b*sizeof(float));
    cudaMemcpy(d_A,A.data(),a*b*sizeof(float),cudaMemcpyHostToDevice);

    cudaMalloc(&d_B,c*d*sizeof(float));
    cudaMemcpy(d_B,B.data(),c*d*sizeof(float),cudaMemcpyHostToDevice);

    cudaMalloc(&d_C,a*d*sizeof(float));
    cudaMemcpy(d_C,C.data(),a*d*sizeof(float),cudaMemcpyHostToDevice);

    // declare the number of blocks per grid and the number of threads per block
    dim3 threadsPerBlock(a, d);
    dim3 blocksPerGrid(1, 1);
        if (a*d > maxThreadsPerBlock){
            threadsPerBlock.x = maxThreadsPerBlock;
            threadsPerBlock.y = maxThreadsPerBlock;
            blocksPerGrid.x = ceil(double(a)/double(threadsPerBlock.x));
            blocksPerGrid.y = ceil(double(d)/double(threadsPerBlock.y));
        }

    matrixMultiplicationKernel<<<blocksPerGrid,threadsPerBlock>>>(d_A,d_B,d_C,
            a,b,c,d);

    // Retrieve result and free data
    cudaMemcpy(C.data(),d_C,a*d*sizeof(float),cudaMemcpyDeviceToHost);

    cudaDeviceSynchronize();
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
}

double SimulateGPU::buildPatches(int W, int H, int skpx, int skpy, int xres,
        int yres, int noRegions, double xspacing, double yspacing, double
        subPatchArea, HabitatTypePtr habTyp, Eigen::MatrixXi& labelledImage,
        Eigen::MatrixXf& populations, std::vector<HabitatPatchPtr>& patches) {

    float *results, *d_results;
    results = (float*)malloc(xres*yres*noRegions*sizeof(float));
    cudaMalloc(&d_results,xres*yres*noRegions*sizeof(float));

    patchComputation<<<xres*yres*noRegions/maxThreadsPerBlock+1,
            maxThreadsPerBlock>>>(W, H, skpx, skpy, xres, yres,
            (float)subPatchArea,(float)xspacing,(float)yspacing,
            (float)habTyp->getMaxPop(),noRegions,
            labelledImage.data(),populations.data(),d_results);
    cudaDeviceSynchronize();

    std::cout << xres*yres*noRegions/maxThreadsPerBlock+1 << std::endl;
    cudaError_t error = cudaGetLastError();
    if (error != cudaSuccess) {
      fprintf(stderr, "ERROR: %s \n", cudaGetErrorString(error));
    }

    cudaMemcpy(results,d_results,xres*yres*noRegions*sizeof(float),
               cudaMemcpyDeviceToHost);

    // Now turn the results into patches
    int counter = 0;
    double initPop = 0.0;
    for (int ii = 0; ii < xres*yres*noRegions; ii++) {
        if (results[xres*yres*noRegions*ii] > 0) {
            // Create new patch to add to patches vector
            HabitatPatchPtr hab(new HabitatPatch());
            hab->setArea((double)results[xres*yres*noRegions*ii]);
            hab->setCX((double)results[xres*yres*noRegions*ii+3]);
            hab->setCY((double)results[xres*yres*noRegions*ii+4]);
            hab->setPopulation((double)results[xres*yres*noRegions*ii+2]);
            hab->setCapacity((double)results[xres*yres*noRegions*ii+1]);
            initPop += (double)results[xres*yres*noRegions*ii];
            patches[counter++] = hab;
        }
    }

    cudaFree(d_results);
    free(results);

    patches.resize(--counter);

    return initPop;
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
        mteKernel<<<(int)ceil(noPaths/maxThreadsPerBlock),maxThreadsPerBlock>>>
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
