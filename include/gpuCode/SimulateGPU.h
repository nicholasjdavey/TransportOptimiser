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

/**
 * Namespace for wrapping CUDA-enabling functions for use in C++ code
 */
namespace SimulateGPU {

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
            std::vector<HabitatPatchPtr>& patches, double& initPop, int&
            noPatches);

    /**
     * Runs the simulation for the fixed traffic flow model in CUDA
     * @param (input) sim as SimulatorPtr
     * @param (input) srp as std::vector<SpeciesRoadPatchesPtr>&
     * @param (input) initPops as std::vector<Eigen::VectorXd>&
     * @param (input) capacities as std::vector<Eigen::VectorXd>&
     * @param (output) endPops as Eigen::MatrixXd& (output)
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
     * @param method as Optimiser::ROVType
     * @param srp as std::vector<SpeciesRoadPatchesPtr>&
     * @param initPops as std::vector<Eigen::VectorXd>&
     * @param capacities std::vector<Eigen::VectorXd>&
     * @param aars Eigen::MatrixXd&
     * @param totalPops as Eigen::MatrixXd&
     * @param condExp as Eigen::MatrixXd&
     * @param optCont as Eigen::MatrixXi&
     */
    void simulateROVCUDA(SimulatorPtr sim, Optimiser::ROVType method,
            std::vector<SpeciesRoadPatchesPtr>& srp,
            std::vector<Eigen::VectorXd>& initPops,
            std::vector<Eigen::VectorXd>& capacities,
            std::vector<std::vector<Eigen::MatrixXd>>& aars,
            std::vector<Eigen::MatrixXd>& totalPops,
            Eigen::MatrixXd& condExp, Eigen::MatrixXi& optCont);

}

#endif
