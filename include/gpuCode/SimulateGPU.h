// CUDA PARALLEL PROCESSING COMMANDS //////////////////////////////////////
// For performance, we only use floats as CUDA is much faster in single-
// precision than double.

#ifndef SIMULATEGPU_H
#define SIMULATEGPU_H

/**
 * Namespace for wrapping CUDA-enabling functions for use in C++ code
 */
namespace SimulateGPU {

    /**
     * Multiplication of two floating point matrices
     * @param A as Eigen::MatrixXd& (input)
     * @param B as Eigen::MatrixXd& (input)
     * @param C as Eigen::MatrixXd& (output)
     */
    void eigenMatrixMult(const Eigen::MatrixXf& A, const
            Eigen::MatrixXf& B, Eigen::MatrixXf& C);

    /**
     * Generates the habitat patches using CUDA
     * @param W as int
     * @param H as int
     * @param skpx as int
     * @param skpy as int
     * @param xres as int
     * @param yres as int
     * @param noRegions as int
     * @param xspacing as double
     * @param yspacing as double
     * @param subPatchArea as double
     * @param habTyp as HabitatTypePtr
     * @param labelledImage as Eigen::MatrixXi&
     * @param populations as Eigen::MatrixXf&
     * @param patches as std::vector<HabitatPatchPtr>&
     * @return Total population as double
     */
    double buildPatches(int W, int H, int skpx, int skpy, int xres, int yres,
            int noRegions, double xspacing, double yspacing, double
            subPatchArea, HabitatTypePtr habTyp, Eigen::MatrixXi&
            labelledImage, Eigen::MatrixXf& populations,
            std::vector<HabitatPatchPtr>& patches);

    /**
     * Runs the simulation for the fixed traffic flow model in CUDA
     * @param sim as SimulatorPtr
     * @param srp as std::vector<SpeciesRoadPatchesPtr>&
     * @param initPops as std::vector<Eigen::VectorXd>&
     * @param capacities as std::vector<Eigen::VectorXd>&
     * @param endPops as Eigen::MatrixXd& (output)
     */
    void simulateMTECUDA(SimulatorPtr sim,
            std::vector<SpeciesRoadPatchesPtr>& srp,
            std::vector<Eigen::VectorXd>& initPops,
            std::vector<Eigen::VectorXd>& capacities,
            Eigen::MatrixXd &endPops);

    /**
     * Runs the simulation for the controlled traffic flow model in CUDA
     * @param sim as SimulatorPtr
     * @param srp as std::vector<SpeciesRoadPatchesPtr>&
     * @param initPops as std::vector<Eigen::VectorXd>&
     * @param capacities as std::vector<Eigen::VectorXd>&
     * @param endPops as Eigen::MatrixXd&
     */
    void simulateROVCUDA();
}

#endif
