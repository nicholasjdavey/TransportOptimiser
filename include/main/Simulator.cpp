#include "../transportbase.h"

Simulator::Simulator() : MonteCarloROV() {
}

Simulator::Simulator(RoadPtr road) {
    this->road = road;
}

Simulator::~Simulator() {
}

void Simulator::simulateMTE() {
    // This routine computes the animal costs based on mean time to extinction.
    // We use Monte Carlo simulation (if any uncertainty) where we run the road
    // at full flow until the end of the design horizon.

    RoadPtr road = this->road.lock();
    OptimiserPtr optimiser = road->getOptimiser();
    ThreadManagerPtr threader = optimiser->getThreadManager();
    ExperimentalScenarioPtr scenario = optimiser->getScenario();
    VariableParametersPtr varParams = optimiser->getVariableParams();

    TrafficProgramPtr program = (road->getOptimiser()
            ->getPrograms())[scenario->getProgram()];
    int controls = program->getFlowRates().size();
    unsigned long noPaths = optimiser->getOtherInputs()->getNoPaths();

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
            results[ii] = threader->push([ii,controls,srp](int id){
                Eigen::VectorXd iarsCont(controls);
                srp[ii]->createSpeciesModel();
                srp[ii]->computeInitialAAR(iarsCont);
                return iarsCont(controls-1);
            });
        }

        for (int ii = 0; ii < srp.size(); ii++) {
            // Retrieve results
            iar(ii) = results[ii].get();
        }

    } else {

        for (int ii = 0; ii < srp.size(); ii++) {
            Eigen::VectorXd iarsCont(controls);
            srp[ii]->createSpeciesModel();
            srp[ii]->computeInitialAAR(iarsCont);
            iar(ii) = iarsCont(controls-1);
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

    // Next, simulate the animal model. For this, we compute each species at
    // each time step. If we are doing visualisation, we only compute one
    // path.

    // We need simulation data for creating surrogates

    // We check to see if there is any uncertainty first to ensure we
    // really need to perform Monte Carlo simulation

    if ((varParams->getLambda()(scenario->getLambda()) == 0) &&
            (varParams->getHabPref()(scenario->getHabPref()) == 0) &&
            (varParams->getBeta()(scenario->getRangingCoeff()) == 0)) {
        // No need for Monte Carlo simulation

    } else {
        // We evaluate the actual road using Monte Carlo simulation
        std::vector<std::future<Eigen::RowVectorXd>> results(srp.size());

        // Create a matrix to store the end populations of each species
        // in each run
        Eigen::MatrixXd endPopulations(noPaths,srp.size());

        if (threader != nullptr) {
            for (unsigned long ii = 0; ii < noPaths; ii++) {
                // Push onto the thread pool with a lambda expression
                results[ii] = threader->push([this,srp, initPops, capacities]
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

        } else {
            // Compute serially
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

        // Prepare to save results to road object
        Eigen::VectorXd endPopMean(srp.size());
        Eigen::VectorXd endPopSD(srp.size());
        Eigen::MatrixXd variances(noPaths,srp.size());

        for (int ii = 0; ii < srp.size(); ii++) {
            endPopMean(ii) = endPopulations.col(ii).mean();
            variances.col(ii) = endPopulations.col(ii).array() -
                    endPopMean(ii);
            endPopSD(ii) = sqrt(variances.col(ii).array().square().mean());
        }

        road->getAttributes()->setEndPopMTE(endPopMean);
        road->getAttributes()->setEndPopMTESD(endPopSD);
    }
}

void Simulator::simulateMTE(std::vector<Eigen::MatrixXd>& visualiseResults) {
    // This routine computes the animal costs based on mean time to extinction
    // where we run the road at full flow until the end of the design horizon.
    RoadPtr road = this->road.lock();
    OptimiserPtr optimiser = road->getOptimiser();
    ThreadManagerPtr threader = optimiser->getThreadManager();
    ExperimentalScenarioPtr scenario = optimiser->getScenario();

    TrafficProgramPtr program = (road->getOptimiser()
            ->getPrograms())[scenario->getProgram()];
    int controls = program->getFlowRates().size();

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
            results[ii] = threader->push([ii,controls,srp](int id){
                Eigen::VectorXd iarsCont(controls);
                srp[ii]->createSpeciesModel();
                srp[ii]->computeInitialAAR(iarsCont);
                return iarsCont(controls-1);
            });
        }

        for (int ii = 0; ii < srp.size(); ii++) {
            // Retrieve results
            iar(ii) = results[ii].get();
        }

    } else {

        for (int ii = 0; ii < srp.size(); ii++) {
            Eigen::VectorXd iarsCont(controls);
            srp[ii]->createSpeciesModel();
            srp[ii]->computeInitialAAR(iarsCont);
            iar(ii) = iarsCont(controls-1);
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

    // Compute one path for visualising the results
    this->simulateMTEPath(srp,initPops,capacities,visualiseResults);
}

void Simulator::naturalBirthDeath(const SpeciesRoadPatchesPtr species, const
        Eigen::VectorXd& capacities, Eigen::VectorXd& pops) {

    RoadPtr road = this->road.lock();
    OptimiserPtr optimiser = road->getOptimiser();
    ThreadManagerPtr threader = optimiser->getThreadManager();
    ExperimentalScenarioPtr scenario = optimiser->getScenario();
    SpeciesPtr spec = species->getSpecies();

    TrafficProgramPtr program = (road->getOptimiser()
            ->getPrograms())[scenario->getProgram()];
    int controls = program->getFlowRates().size();

    int timeSteps = road->getOptimiser()->getEconomic()->getYears();
    double stepSize = road->getOptimiser()->getEconomic()->getTimeStep();

    // Initialise random number generator
    std::mt19937_64 generator;
    std::normal_distribution<double> growth(spec->getGrowthRateMean(),
            spec->getGrowthRateSD());

    Eigen::VectorXd gr(pops.size());

    for (int ii = 0; ii < pops.size(); ii++) {
        gr(ii) = growth(generator)*0.01;
    }

    gr = (pops.array() - capacities.array())/(capacities.array());

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
        reArrMat[ii] = (species[ii]->getTransProbs())*(species[ii]->
                getSurvivalProbs()[controls-1]);
    }

    for (int ii = 0; ii < timeSteps; ii++) {

        // First, compute the effects of species interaction
        // Not implemented yet. TO DO

        // Next, compute animal movement and road mortality
        for (int jj = 0; jj < species.size(); jj++) {
            pops[jj] =reArrMat[ii]*initPops[jj];
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
            pops[kk] =reArrMat[jj]*initPops[kk];
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
