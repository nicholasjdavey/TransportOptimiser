#include "../transportbase.h"

Simulator::Simulator() : MonteCarloROV() {
}

Simulator::Simulator(RoadPtr road) {
    this->road = road;
}

Simulator::~Simulator() {
}

SimulatorPtr Simulator::me() {
    return shared_from_this();
}

void Simulator::simulateMTE() {
    // This routine computes the animal costs based on mean time to extinction.
    // We use Monte Carlo simulation (if any uncertainty) where we run the road
    // at full flow until the end of the design horizon. This is used for
    // computation, not for visualisation.

    RoadPtr road = this->road.lock();
    OptimiserPtr optimiser = road->getOptimiser();
    ThreadManagerPtr threader = optimiser->getThreadManager();
    bool gpu = optimiser->getGPU();
    ExperimentalScenarioPtr scenario = optimiser->getScenario();
    VariableParametersPtr varParams = optimiser->getVariableParams();

    TrafficProgramPtr program = (road->getOptimiser()
            ->getPrograms())[scenario->getProgram()];
    int controls = program->getFlowRates().size();
    unsigned long noPaths = optimiser->getOtherInputs()->getNoPaths();

    std::vector<SpeciesRoadPatchesPtr> srp = road->getSpeciesRoadPatches();
    Eigen::VectorXd iar(srp.size());

    // First, compute the IARs for the road
    if (threader != nullptr) {
        // If we have access to multi-threading, use it for computing the
        // iars
        std::vector<std::future<double>> results(srp.size());

        for (int ii = 0; ii < srp.size(); ii++) {
            // Push onto the pool with a lambda expression
            results[ii] = threader->push([ii,srp](int id){
                srp[ii]->computeInitialAAR();
                return srp[ii]->getInitAAR()(srp[ii]->getInitAAR().size()-1);
            });
        }

        for (int ii = 0; ii < srp.size(); ii++) {
            // Retrieve results
            iar(ii) = results[ii].get();
        }

    } else {

        for (int ii = 0; ii < srp.size(); ii++) {
            srp[ii]->computeInitialAAR();
            iar(ii) = srp[ii]->getInitAAR()(srp[ii]->getInitAAR().size()-1);
        }
    }

    road->getAttributes()->setIAR(iar);

    // Initialise the populations and capacities vectors
    std::vector<Eigen::VectorXd> initPops(srp.size());
    std::vector<Eigen::VectorXd> capacities(srp.size());

    for (int ii = 0; ii < srp.size(); ii++) {
        initPops[ii].resize(srp[ii]->getHabPatches().size());
        capacities[ii].resize(srp[ii]->getHabPatches().size());

        for (int jj = 0; jj < srp[ii]->getHabPatches().size(); jj++) {
            initPops[ii](jj) = srp[ii]->getHabPatches()[jj]->getPopulation();
            capacities[ii](jj) = srp[ii]->getHabPatches()[jj]->getCapacity();
        }
    }

    // Next, simulate the animal model. For this, we compute each species at
    // each time step. If we are doing visualisation, we only compute one
    // path.

    // We need simulation data for creating surrogates

    // We check to see if there is any uncertainty first to ensure we
    // really need to perform Monte Carlo simulation

//    (varParams->getLambda()(scenario->getLambda()) == 0) &&
//                (varParams->getHabPref()(scenario->getHabPref()) == 0) &&
//                (varParams->getBeta()(scenario->getRangingCoeff()) == 0) &&

    if (varParams->getGrowthRateSDMultipliers()(scenario->getPopGRSD()) == 0) {

        Eigen::RowVectorXd endPopulations(srp.size());

        // No need for Monte Carlo simulation
        std::vector<Eigen::VectorXd> finalPops =
                initPops;
        this->simulateMTEPath(srp,initPops,
                capacities,finalPops);
        for (int ii = 0; ii < srp.size();
                ii++) {
            endPopulations(ii) = finalPops[ii].sum();
        }

    } else {
        // Create a matrix to store the end populations of each species
        // in each run
        Eigen::MatrixXd endPopulations(noPaths,srp.size());

        if (threader != nullptr) {
            // If we have access to GPU-enabled computing, we exploit it
            // here and at the calling function in the GA optimiser, we
            // implement the standard multi-threading on the host. If not,
            // we do not implement multi-threading at the higher level and
            // instead implement it here.
            if (gpu) {
                // Call the external, CUDA-compiled code
                SimulateGPU::simulateMTECUDA(this->me(),srp,initPops,
                        capacities,endPopulations);

            } else {
                // We evaluate the actual road using Monte Carlo simulation
                std::vector<std::future<Eigen::RowVectorXd>> results(noPaths);

                for (unsigned long ii = 0; ii < noPaths; ii++) {
                    // Push onto the thread pool with a lambda expression
                    results[ii] = threader->push([this, srp, initPops, capacities]
                            (int id){
                        Eigen::RowVectorXd endPops(srp.size());
                        std::vector<Eigen::VectorXd> finalPops =
                                initPops;
                        this->simulateMTEPath(srp,initPops,
                                capacities,finalPops);
                        for (int jj = 0; jj < srp.size();
                                jj++) {
                            endPops(jj) = finalPops[jj].sum();
                        }
                        return endPops;
                    });
                }

                for (unsigned long ii = 0; ii < noPaths; ii++) {
                    endPopulations.row(ii) = results[ii].get();
                }
            }

        } else {

            // Compute serially. We can also use GPU computing here but the
            // earlier calling function does not use multi-threading on the
            // host.
            if (gpu) {
                // Call the external, CUDA-compiled code
                SimulateGPU::simulateMTECUDA(this->me(),srp,initPops,
                        capacities,endPopulations);
            } else {

                for (unsigned long ii = 0; ii < noPaths; ii++) {

                    for (int jj = 0; jj < srp.size(); jj++) {
                        Eigen::RowVectorXd endPops(srp.size());
                        std::vector<Eigen::VectorXd> finalPops =
                                initPops;
                        this->simulateMTEPath(srp,initPops,capacities,
                                finalPops);

                        for (int kk = 0; kk < srp.size(); kk++) {
                            endPops(jj) = finalPops[jj].sum();
                        }

                        endPopulations.row(ii) = endPops;
                    }
                }
            }
        }

        // Prepare to save results to road object
        Eigen::VectorXd endPopMean(srp.size());
        Eigen::VectorXd endPopSD(srp.size());
        Eigen::MatrixXd variances(noPaths,srp.size());

        for (int ii = 0; ii < srp.size(); ii++) {
            endPopMean(ii) = endPopulations.col(ii).mean();
            variances.col(ii) = endPopulations.col(ii).array() -
                    endPopMean(ii);
            endPopSD(ii) = sqrt(variances.col(ii).array().square().mean());
            srp[ii]->setEndPopMean(endPopMean(ii));
            srp[ii]->setEndPopSD(endPopSD(ii));
        }

        road->getAttributes()->setEndPopMTE(endPopMean);
        road->getAttributes()->setEndPopMTESD(endPopSD);
        road->getCosts()->setPenalty(0.0);
        road->getAttributes()->setTotalValueSD(0.0);
    }
}

void Simulator::simulateMTE(std::vector<Eigen::MatrixXd>& visualiseResults) {
    // This routine computes the animal costs based on mean time to extinction
    // where we run the road at full flow until the end of the design horizon.
    RoadPtr road = this->road.lock();
    OptimiserPtr optimiser = road->getOptimiser();
    ThreadManagerPtr threader = optimiser->getThreadManager();

    std::vector<SpeciesRoadPatchesPtr> srp = road->
            getSpeciesRoadPatches();
    Eigen::VectorXd iar(srp.size());

    // First, compute the IARs for the road
    if (threader != nullptr) {
        // If we have access to multi-threading, use it for computing the
        // iars
        std::vector<std::future<double>> results(srp.size());

        for (int ii = 0; ii < srp.size(); ii++) {
            // Push onto the pool with a lambda expression
            results[ii] = threader->push([ii,srp](int id){
                srp[ii]->computeInitialAAR();
                return srp[ii]->getInitAAR()(srp[ii]->getInitAAR().size()-1);
            });
        }

        for (int ii = 0; ii < srp.size(); ii++) {
            // Retrieve results
            iar(ii) = results[ii].get();
        }

    } else {

        for (int ii = 0; ii < srp.size(); ii++) {
            srp[ii]->computeInitialAAR();
            iar(ii) = srp[ii]->getInitAAR()(srp[ii]->getInitAAR().size()-1);
        }
    }

    road->getAttributes()->setIAR(iar);

    // Initialise the populations and capacities vectors
    std::vector<Eigen::VectorXd> initPops(srp.size());
    std::vector<Eigen::VectorXd> capacities(srp.size());

    for (int ii = 0; ii < srp.size(); ii++) {
        initPops[ii].resize(srp[ii]->getHabPatches().size());
        capacities[ii].resize(srp[ii]->getHabPatches().size());

        for (int jj = 0; jj < srp.size(); jj++) {
            initPops[ii](jj) = srp[ii]->getHabPatches()[jj]->getPopulation();
            capacities[ii](jj) = srp[ii]->getHabPatches()[jj]->getCapacity();
        }
    }

    // When visualising, we also need to know where the road cells are, so we
    // compute these for the habitat patches

    // Compute one path for visualising the results
    this->simulateMTEPath(srp,initPops,capacities,visualiseResults);
}

void Simulator::simulateROVCR() {
    // For now, this method only considers a single species

    // SIM ROVCR (we cannot use the base Monte Carlo routine for this method)

    // Simulate forward paths
    // Save all paths:
    //  1. Values of all uncertainties
    //  2. Control taken
    //  3. Adjusted populations

    // Save optimal control map produced:
    //  1. Optimal values
    //  2. Optimal controls

    RoadPtr road = this->road.lock();
    OptimiserPtr optimiser = road->getOptimiser();
    ThreadManagerPtr threader = optimiser->getThreadManager();
    bool gpu = optimiser->getGPU();
    ExperimentalScenarioPtr scenario = optimiser->getScenario();
    VariableParametersPtr varParams = optimiser->getVariableParams();
    Optimiser::ROVType method = optimiser->getROVMethod();

    TrafficProgramPtr program = (road->getOptimiser()->getPrograms())[
            scenario->getProgram()];
    int controls = program->getFlowRates().size();
    int noPaths = optimiser->getOtherInputs()->getNoPaths();
    int nYears = optimiser->getEconomic()->getYears();

    std::vector<SpeciesRoadPatchesPtr> srp = road->getSpeciesRoadPatches();
    Eigen::VectorXd iar(srp.size());

    // First, compute the IAR for each species for this road
    if (threader != nullptr) {
        // If we have access to multi-threading, use it for computing the
        // iars
        std::vector<std::future<double>> results(srp.size());

        for (int ii = 0; ii < srp.size(); ii++) {
            // Push onto the pool with a lambda expression
            results[ii] = threader->push([ii,srp](int id){
                srp[ii]->computeInitialAAR();
                return srp[ii]->getInitAAR()(srp[ii]->getInitAAR().size()-1);
            });
        }

        for (int ii = 0; ii < srp.size(); ii++) {
            // Retrieve results
            iar(ii) = results[ii].get();
        }

    } else {

        for (int ii = 0; ii < srp.size(); ii++) {
            srp[ii]->computeInitialAAR();
            iar(ii) = srp[ii]->getInitAAR()(srp[ii]->getInitAAR().size()-1);
        }
    }

    road->getAttributes()->setIAR(iar);

    // Initialise the populations and capacities vectors
    std::vector<Eigen::VectorXd> initPops(srp.size());
    std::vector<Eigen::VectorXd> capacities(srp.size());

    for (int ii = 0; ii < srp.size(); ii++) {
        initPops[ii].resize(srp[ii]->getHabPatches().size());
        capacities[ii].resize(srp[ii]->getHabPatches().size());

        for (int jj = 0; jj < srp.size(); jj++) {
            initPops[ii](jj) = srp[ii]->getHabPatches()[jj]->getPopulation();
            capacities[ii](jj) = srp[ii]->getHabPatches()[jj]->getCapacity();
        }
    }

    // VALUATION //////////////////////////////////////////////////////////////
    // Next, perform the valuation. If there is no growth rate uncertainty, we
    // only need one path.

//    std::vector<std::vector<Eigen::MatrixXd>> mcPops(srp.size());

//    for (int ii = 0; ii < srp.size(); ii++) {
//        mcPops[ii].resize(nYears);
//        for (int jj = 0; jj < nYears; jj++) {
//            mcPops[ii][jj].resize(noPaths,srp->getHabPatches().size());
//        }
//    }

    // We save the following information so that we can create the policy maps

    // 1. AAR maps for each path for each control at each time step
    std::vector<std::vector<Eigen::MatrixXd>> aars(srp.size());

    for (int ii = 0; ii < srp.size(); ii++) {
        aars[ii].resize(nYears);
        for (int jj = 0; jj < nYears; jj++) {
            aars[ii][jj].resize(noPaths,controls);
        }
    }

    // 2. Total population on each path for each time step
    std::vector<Eigen::MatrixXd> totalPops(srp.size());

    for (int ii = 0; ii < srp.size(); ii++) {
        totalPops[ii].resize(noPaths,nYears);
    }

    // 3. Optimal profit-to-go matrix (along each path)
    Eigen::MatrixXd condExp(noPaths,nYears);

    // 4. Optimal control matrix (along each path)
    Eigen::MatrixXi optCont(noPaths,nYears);

    if (varParams->getGrowthRateSDMultipliers()(scenario->getPopGRSD()) == 0) {

        Eigen::RowVectorXd endPopulations(srp.size());

        // Fill in code here

    } else {
        // Create
        // We have to use Monte Carlo simulation

        // Create a matrix to store the results
        if (threader != nullptr) {
            // If we have access to GPU-enabled computing, we exploit it
            // here and at the calling function in the GA optimiser, we
            // implement the standard multi-threading on the host. If not,
            // we do not implement multi-threading at the higher level and
            // instead implement it here.
            if (gpu) {
                // Call the external, CUDA-compiled code
                SimulateGPU::simulateROVCUDA(this->me(),srp,aars,totalPops,
                        condExp,optCont);

            } else {
                // Put multi-threaded code without the GPU here
                // Don't bother for now
            }

        } else {
            // We will always use multiple threads otherwise it is too slow

            // For completeness, add single-threaded code here
            // Don't bother for now
        }
    }
}

void Simulator::simulateROVCR(std::vector<Eigen::MatrixXd>& visualisePops,
        std::vector<int> &visualiseFlows) {

}

///////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////

void Simulator::naturalBirthDeath(const SpeciesRoadPatchesPtr species, const
        Eigen::VectorXd& capacities, Eigen::VectorXd& pops) {

//    RoadPtr road = this->road.lock();
//    OptimiserPtr optimiser = road->getOptimiser();
//    ThreadManagerPtr threader = optimiser->getThreadManager();
//    ExperimentalScenarioPtr scenario = optimiser->getScenario();
    SpeciesPtr spec = species->getSpecies();

//    TrafficProgramPtr program = (road->getOptimiser()
//            ->getPrograms())[scenario->getProgram()];
//    int controls = program->getFlowRates().size();

//    int timeSteps = road->getOptimiser()->getEconomic()->getYears();
//    double stepSize = road->getOptimiser()->getEconomic()->getTimeStep();

    // Initialise random number generator
//    unsigned seed1 = std::chrono::system_clock::now().time_since_epoch().
//            count();
//    std::mt19937 generator(seed1);
    std::normal_distribution<double> growth(spec->getGrowthRateMean(),
            spec->getGrowthRateSD());

    Eigen::VectorXd gr(pops.size());
    double stepSize = species->getRoad()->getOptimiser()->getEconomic()->
            getTimeStep();

    for (int ii = 0; ii < pops.size(); ii++) {
        gr(ii) = stepSize*growth(generator)*0.01;
    }

    gr = (gr.array()*(capacities.array() - pops.array())).array()/
            (capacities.array());

    pops = pops.array()*(1 + gr.array());
}

void Simulator::simulateMTEPath(const std::vector<SpeciesRoadPatchesPtr>&
        species, const std::vector<Eigen::VectorXd> &initPops, const
        std::vector<Eigen::VectorXd> &capacities, std::vector<Eigen::VectorXd>
        &finalPops) {
    std::vector<Eigen::VectorXd> pops = initPops;

    std::vector<Eigen::MatrixXd> reArrMat(species.size());
    RoadPtr road = this->road.lock();
    OptimiserPtr optimiser = road->getOptimiser();
    ExperimentalScenarioPtr scenario = optimiser->getScenario();

    int timeSteps = road->getOptimiser()->getEconomic()->getYears();

    TrafficProgramPtr program = (road->getOptimiser()
            ->getPrograms())[scenario->getProgram()];
    int controls = program->getFlowRates().size();

    // Initialise all the transformation matrices that take the population at
    // the start of a period (for a single species) to the population at the
    // end of the period by accounting for transition and mortality.
    for (int ii = 0; ii < species.size(); ii++) {
        reArrMat[ii] = (species[ii]->getTransProbs()).array()*(species[ii]->
                getSurvivalProbs()[controls-1]).array();
    }

    for (int ii = 0; ii < timeSteps; ii++) {

        // First, compute the effects of species interaction
        // Not implemented yet. TO DO

        // Next, compute animal movement and road mortality
        for (int jj = 0; jj < species.size(); jj++) {
            pops[jj] =reArrMat[jj].transpose()*pops[jj];
        }

        // Finally, account for natural birth and death
        for (int jj = 0; jj < species.size(); jj ++) {
            this->naturalBirthDeath(species[jj],capacities[jj],pops[jj]);
        }
    }

    for (int ii = 0; ii < species.size(); ii++) {
        finalPops[ii] = pops[ii];
    }
}

void Simulator::simulateMTEPath(const std::vector<SpeciesRoadPatchesPtr>&
        species, const std::vector<Eigen::VectorXd>& initPops, const
        std::vector<Eigen::VectorXd> &capacities, std::vector<Eigen::MatrixXd>&
        visualiseResults) {

    std::vector<Eigen::VectorXd> pops = initPops;

    std::vector<Eigen::MatrixXd> reArrMat(species.size());
    RoadPtr road = this->road.lock();
    OptimiserPtr optimiser = road->getOptimiser();
    ExperimentalScenarioPtr scenario = optimiser->getScenario();

    int timeSteps = road->getOptimiser()->getEconomic()->getYears();

    TrafficProgramPtr program = (road->getOptimiser()
            ->getPrograms())[scenario->getProgram()];
    int controls = program->getFlowRates().size();

    // Initialise all the transformation matrices that take the population at
    // the start of a period (for a single species) to the population at the
    // end of the period by accounting for transition and mortality.
    for (int jj = 0; jj < species.size(); jj++) {
        reArrMat[jj] = (species[jj]->getTransProbs())*(species[jj]->
                getSurvivalProbs()[controls-1]);

        visualiseResults[jj].col(0) = initPops[jj].col(0);
    }

    for (int jj = 0; jj < timeSteps; jj++) {

        // First, compute the effects of species competition
        // Not implemented yet. TO DO

        // Next, compute animal movement and road mortality
        for (int kk = 0; kk < species.size(); kk++) {
            pops[kk] = reArrMat[jj].transpose()*pops[kk];
        }

        // Finally, account for natural birth and death
        for (int kk = 0; kk < species.size(); kk ++) {
            this->naturalBirthDeath(species[kk],capacities[kk],pops[kk]);
        }

        // Save the results to the visualisation matrix
        for (int kk = 0; kk < species.size(); kk++) {
            visualiseResults[jj].col(jj+1) = pops[kk];
        }
    }
}

void Simulator::simulateROVCRPath(const std::vector<SpeciesRoadPatchesPtr>&
            species, const std::vector<Eigen::VectorXd>& initPops, const
            std::vector<Eigen::VectorXd>& capacities,
            std::vector<Eigen::VectorXd>& exogenousPaths,
            std::vector<Eigen::VectorXd>& endogenousPaths) {

}

void Simulator::simulateROVCRPath(const std::vector<SpeciesRoadPatchesPtr>&
            species, const std::vector<Eigen::VectorXd>& initPops, const
            std::vector<Eigen::VectorXd>& capacities,
            const std::vector<Eigen::VectorXd>& exogenousPaths,
            std::vector<Eigen::VectorXd>& endogenousPaths,
            std::vector<Eigen::MatrixXd> &visualiseResults) {

}

void Simulator::recomputeForwardPath(const std::vector<SpeciesRoadPatchesPtr>&
            species, const std::vector<Eigen::VectorXd>& initPops, const
            std::vector<Eigen::VectorXd>& capacities,
            const std::vector<Eigen::VectorXd>& exogenousPaths,
            const unsigned long timeStep, const std::vector<std::vector<
            alglib::spline1dinterpolant>> optPtG,
            std::vector<Eigen::VectorXd>& endogenousPaths) {

}
