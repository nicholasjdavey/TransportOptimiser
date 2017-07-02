// CUDA PARALLEL PROCESSING COMMANDS //////////////////////////////////////
// For performance, we only use floats as CUDA is much faster in single-
// precision than double.

#ifndef SIMULATEGPU_H
#define SIMULATEGPU_H

#define CHECK_RESULT 1
#define ENABLE_NAIVE 1

// Thread block size
#define BLOCK_SIZE 32

// outer product vetor size is VECTOR_SIZE * BLOCK_SIZE
#define VECTOR_SIZE 32

// External CUDA C routines
//extern "C" int iDivUp(int, int);
//extern "C" void gpuErrchk(cudaError_t);
//extern "C" void cusolveSafeCall(cusolverStatus_t);
//extern "C" void cublasSafeCall(cublasStatus_t);
//extern "C" cublasStatus_t cublasCreate_v2(cublasHandle_t*);

/**
 * Namespace for wrapping CUDA-enabling functions for use in C++ code
 */

class Optimiser;
typedef std::shared_ptr<Optimiser> OptimiserPtr;

class RoadGA;
typedef std::shared_ptr<RoadGA> RoadGAPtr;

namespace SimulateGPU {

    // CUDA WRAPPERS //////////////////////////////////////////////////////////

    /**
     * Computes the expected present value for an uncertain parameter (e.g.
     * a commodity) given fixed usage over time.
     *
     * @param uncertainty as UncertaintyPtr
     */
    void expPV(UncertaintyPtr uncertainty);

    /**
     * Multiplication of two floating point matrices (naive)
     * @param (input) A as Eigen::MatrixXd&
     * @param (input) B as Eigen::MatrixXd&
     * @param (output) C as Eigen::MatrixXd&
     */
    void eMMN(const Eigen::MatrixXd& A, const Eigen::MatrixXd& B,
            Eigen::MatrixXd& C);

    /**
     * Multiplication of two floating point matrices
     *
     * This is computationally more effective than the naive approach shown
     * above.
     * @param (input) A as Eigen::MatrixXd&
     * @param (input) B as Eigen::MatrixXd&
     * @param (output) C as Eigen::MatrixXd&
     */
    void eMM(const Eigen::MatrixXd& A, const Eigen::MatrixXd& B,
            Eigen::MatrixXd& C);

    /**
     * Element-wise multiplication of two floating point matrices
     * @param (input) A as Eigen::MatrixXd&
     * @param (input) B as Eigen::MatrixXd&
     * @param (output) C as Eigen::MatrixXd&
     */
    void ewMM(const Eigen::MatrixXd& A, const Eigen::MatrixXd& B,
            Eigen::MatrixXd& C);

    /**
     * Element-wise dividion of two floating point matrices
     * @param (input) A as Eigen::MatrixXd&
     * @param (input) B as Eigen::MatrixXd&
     * @param (output) C as Eigen::MatrixXd&
     */
    void ewMD(const Eigen::MatrixXd &A, const Eigen::MatrixXd &B,
            Eigen::MatrixXd &C);

    /**
     * Computes the number of times the lines in XY1 intersect the curve
     * represented by XY2.
     *
     * @param XY1 as const Eigen::MatrixXd&
     * @param XY2 as const Eigen::MatrixXd&
     * @param crossings as Eigen::VectorXi&
     */
    void lineSegmentIntersect(const Eigen::MatrixXd& XY1, const
            Eigen::MatrixXd& XY2, Eigen::VectorXi &crossings);

    /**
     * Generates the habitat patches using CUDA for a specific habitat type
     *
     * This function also updates the count of patches and the overall
     * population accounted for.
     *
     * @param (input) W as int
     * @param (input) H as int
     * @param (input) skpx as int
     * @param (input) skpy as int
     * @param (input) xres as int
     * @param (input) yres as int
     * @param (input) noRegions as int
     * @param (input) xspacing as double
     * @param (input) yspacing as double
     * @param (input) subPatchArea as double
     * @param (input) habTyp as HabitatTypePtr
     * @param (input) labelledImage as const Eigen::MatrixXi&
     * @param (input) populations as const Eigen::MatrixXf&
     * @param (output) patches as std::vector<HabitatPatchPtr>&
     * @param (output) initPop as double
     * @param (output) noPatches as int
     */
    void buildPatches(int W, int H, int skpx, int skpy, int xres, int yres,
            int noRegions, double xspacing, double yspacing, double
            subPatchArea, HabitatTypePtr habTyp, const Eigen::MatrixXi&
            labelledImage, const Eigen::MatrixXd &populations,
            std::vector<HabitatPatchPtr>& patches, double& initPop,
            Eigen::VectorXd &initPops, Eigen::VectorXd &capacities, int& noPatches);

    /**
     * Runs the simulation for the fixed traffic flow model in CUDA
     * @param (input) sim as SimulatorPtr
     * @param (input) srp as std::vector<SpeciesRoadPatchesPtr>&
     * @param (input) initPops as std::vector<Eigen::VectorXd>&
     * @param (input) capacities as std::vector<Eigen::VectorXd>&
     * @param (output) endPops as Eigen::MatrixXd& (output)
     * @return Computation status as Optimiser::ComputationStatus
     */
    void simulateMTECUDA(SimulatorPtr sim,
            std::vector<SpeciesRoadPatchesPtr>& srp,
            std::vector<Eigen::VectorXd>& initPops,
            std::vector<Eigen::VectorXd>& capacities,
            Eigen::MatrixXd &endPops);

    /**
     * Runs the simulation for the controlled traffic flow model in CUDA.
     *
     * @brief simulateROVCUDA
     * @param sim as SimulatorPtr
     * @param srp as std::vector<SpeciesRoadPatchesPtr>&
     * @param adjPops std::vector<Eigen::MatrixXd>&
     * @param unitProfits Eigen::MatrixXd&
     * @param condExp as Eigen::MatrixXd&
     * @param optCont as Eigen::MatrixXi&
     */
    void simulateROVCUDA(SimulatorPtr sim,
            std::vector<SpeciesRoadPatchesPtr>& srp,
            std::vector<Eigen::MatrixXd> &adjPops, Eigen::MatrixXd &
            unitProfits, Eigen::MatrixXd& condExp, Eigen::MatrixXi& optCont);

    /**
     * Performs MTE simulations for all sample roads used to build the
     * surrogate.
     *
     * @note This routine should be faster than simulateMTECUDA as the random
     * variables are generated once at the beginning and used across all
     * sample roads
     *
     * @note This routine only computes the local linear version of the
     * surrogate
     */
    void surrogateMTECUDA();

    /**
     * Performs ROV simulations for all sample roads used to build the
     * surrogate.
     *
     * @note This routine should be faster than simulateMTECUDA as the random
     * variables are generated once at the beginning and used across all
     * sample roads
     */
    void surrogateROVCUDA();

    /**
     * Builds and returns the surrogate model for the MTE scenario using CUDA
     *
     * @param op as RoadGAPtr
     * @param speciesID as int
     */
    void buildSurrogateMTECUDA(RoadGAPtr op, int speciesID);

    /**
     * Builds the ROV surrogate using CUDA
     *
     * @param op as OptimiserPtr
     * @param surrogate as Eigen::VectorXd&
     */
    void buildSurrogateROVCUDA(RoadGAPtr op);

    /**
     * Interpolates a surrogate model at many points using the GPU. This is
     * used for plotting the surrogate models.
     *
     * @param surrogate as Eigen::VectorXd&
     * @param predictors as Eigen::VectorXd&
     * @param results as Eigen::VectorXd&
     * @param dimRes as int
     * @param noDims as int
     */
    void interpolateSurrogateMulti(Eigen::VectorXd& surrogate,
            Eigen::VectorXd &predictors, Eigen::VectorXd &results, int dimRes,
            int noDims);

    //  HELPER ROUTINES (COMPUTED ON CPU) /////////////////////////////////////

    /**
     * Converts a dense matrix to a sparse matrix, providing the output values
     * and corresponding row index of each element.
     *
     * @param denseIn as float*
     * @param rows as int
     * @param cols as int
     * @param sparseOut as float*
     * @param elemsPerCol as int*
     * @param rowIdx as int*
     * @param totalElements as int&
     */
    void dense2Sparse(float* denseIn, int rows, int cols, float* sparseOut,
            int* elemsPerCol, int* rowIdx, int &totalElements);

    /**
     * CPU-based multiple linear regression (global)
     *
     * @param noPoints as int
     * @param noDims as int
     * @param (in) xvals as float*
     * @param (in) yvals as float*
     * @param (out) X as float*
     */
    void multiLinReg(int noPoints, int noDims, float *xvals, float *yvals,
            float *X);

    /**
     * Solves a system of linear equations
     *
     * @param dims as int
     * @param (in) A as float*
     * @param (in) B as float*
     * @param (out) C as float*
     */
    void solveLinearSystem(int dims, float *A, float *B, float *C);

//    /**
//     * Allocates data to the different regression data based on the contrl
//     *
//     * @param noPaths as int
//     * @param Controls as int
//     * @param noDims as int
//     * @param controls as int*
//     * @param xin as float*
//     * @param condExp as float*
//     * @param dataPoints as int*
//     * @param xvals as float*
//     * @param yvals as float*
//     */
//    void allocateXYRegressionData(int noPaths, int noControls, int noDims, int*
//            controls, float* xin, float* condExp, int* dataPoints, float*
//            xvals, float* yvals);

//    // CALLING CUSOLVER ///////////////////////////////////////////////////////////
//    /*******************/
//    /* iDivUp FUNCTION */
//    /*******************/
//    extern "C" int iDivUp(int a, int b){ return ((a % b) != 0) ? (a / b + 1) : (a / b); }

//    /********************/
//    /* CUDA ERROR CHECK */
//    /********************/
//    // --- Credit to http://stackoverflow.com/questions/14038589/what-is-the-canonical-way-to-check-for-errors-using-the-cuda-runtime-api
//    void gpuAssert(cudaError_t code, char *file, int line, bool abort=true)
//    {
//       if (code != cudaSuccess)
//       {
//          fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
//          if (abort) { exit(code); }
//       }
//    }

//    extern "C" void gpuErrchk(cudaError_t ans) { gpuAssert((ans), __FILE__, __LINE__); }

//    /**************************/
//    /* CUSOLVE ERROR CHECKING */
//    /**************************/
//    static const char *_cusolverGetErrorEnum(cusolverStatus_t error)
//    {
//        switch (error)
//        {
//            case CUSOLVER_STATUS_SUCCESS:
//                return "CUSOLVER_SUCCESS";

//            case CUSOLVER_STATUS_NOT_INITIALIZED:
//                return "CUSOLVER_STATUS_NOT_INITIALIZED";

//            case CUSOLVER_STATUS_ALLOC_FAILED:
//                return "CUSOLVER_STATUS_ALLOC_FAILED";

//            case CUSOLVER_STATUS_INVALID_VALUE:
//                return "CUSOLVER_STATUS_INVALID_VALUE";

//            case CUSOLVER_STATUS_ARCH_MISMATCH:
//                return "CUSOLVER_STATUS_ARCH_MISMATCH";

//            case CUSOLVER_STATUS_EXECUTION_FAILED:
//                return "CUSOLVER_STATUS_EXECUTION_FAILED";

//            case CUSOLVER_STATUS_INTERNAL_ERROR:
//                return "CUSOLVER_STATUS_INTERNAL_ERROR";

//            case CUSOLVER_STATUS_MATRIX_TYPE_NOT_SUPPORTED:
//                return "CUSOLVER_STATUS_MATRIX_TYPE_NOT_SUPPORTED";

//        }

//        return "<unknown>";
//    }

//    inline void __cusolveSafeCall(cusolverStatus_t err, const char *file, const int line)
//    {
//        if(CUSOLVER_STATUS_SUCCESS != err) {
//            fprintf(stderr, "CUSOLVE error in file '%s', line %d\n %s\nerror %d: %s\nterminating!\n",__FILE__, __LINE__,err, \
//                                    _cusolverGetErrorEnum(err)); \
//            cudaDeviceReset(); assert(0); \
//        }
//    }

//    extern "C" void cusolveSafeCall(cusolverStatus_t err) { __cusolveSafeCall(err, __FILE__, __LINE__); }

//    /*************************/
//    /* CUBLAS ERROR CHECKING */
//    /*************************/
//    static const char *_cublasGetErrorEnum(cublasStatus_t error)
//    {
//        switch (error)
//        {
//            case CUBLAS_STATUS_SUCCESS:
//                return "CUBLAS_STATUS_SUCCESS";

//            case CUBLAS_STATUS_NOT_INITIALIZED:
//                return "CUBLAS_STATUS_NOT_INITIALIZED";

//            case CUBLAS_STATUS_ALLOC_FAILED:
//                return "CUBLAS_STATUS_ALLOC_FAILED";

//            case CUBLAS_STATUS_INVALID_VALUE:
//                return "CUBLAS_STATUS_INVALID_VALUE";

//            case CUBLAS_STATUS_ARCH_MISMATCH:
//                return "CUBLAS_STATUS_ARCH_MISMATCH";

//            case CUBLAS_STATUS_MAPPING_ERROR:
//                return "CUBLAS_STATUS_MAPPING_ERROR";

//            case CUBLAS_STATUS_EXECUTION_FAILED:
//                return "CUBLAS_STATUS_EXECUTION_FAILED";

//            case CUBLAS_STATUS_INTERNAL_ERROR:
//                return "CUBLAS_STATUS_INTERNAL_ERROR";

//            case CUBLAS_STATUS_NOT_SUPPORTED:
//                return "CUBLAS_STATUS_NOT_SUPPORTED";

//            case CUBLAS_STATUS_LICENSE_ERROR:
//                return "CUBLAS_STATUS_LICENSE_ERROR";
//    }

//        return "<unknown>";
//    }

//    inline void __cublasSafeCall(cublasStatus_t err, const char *file, const int line)
//    {
//        if(CUBLAS_STATUS_SUCCESS != err) {
//            fprintf(stderr, "CUBLAS error in file '%s', line %d\n %s\nerror %d: %s\nterminating!\n",__FILE__, __LINE__,err, \
//                                    _cublasGetErrorEnum(err)); \
//            cudaDeviceReset(); assert(0); \
//        }
//    }

//    extern "C" void cublasSafeCall(cublasStatus_t err) { __cublasSafeCall(err, __FILE__, __LINE__); }

}

#endif
