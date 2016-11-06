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
    // We use Monte Carlo simulation (if any uncertainty) or just employ one
    // run (if no uncertainty) where we run the road at full flow until the end
    // of the design horizon.

    RoadPtr road = this->road.lock();
    OptimiserPtr optimiser = road->getOptimiser();
    ThreadManagerPtr threader = optimiser->getThreadManager();
    unsigned int paths = optimiser->getOtherInputs()->getNoPaths();
    ExperimentalScenarioPtr scenario = optimiser->getScenario();
    VariableParametersPtr varParams = optimiser->getVariableParams();

    TrafficProgramPtr program = (road->getOptimiser()
            ->getPrograms())[scenario->getProgram()];
    int controls = program->getFlowRates().size();
    int timeSteps = road->getOptimiser()->getEconomic()->getYears();
    double stepSize = road->getOptimiser()->getEconomic()->getTimeStep();

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

    // Next, simulate the animal model. For this, we compute each species at each
    // time step
    // Initialise the populations and capacities vectors
    std::vector<Eigen::VectorXd> pops(srp.size());
    std::vector<Eigen::VectorXd> capacities(srp.size());

    for (int ii = 0; ii < srp.size(); ii++) {
        pops[ii].resize(srp[ii]->getHabPatches().size());
        capacities[ii].resize(srp[ii]->getHabPatches().size());

        for (int jj = 0; jj < srp.size(); jj++) {
            pops[ii](jj) = srp[ii]->getHabPatches()[jj]->getPopulation();
            capacities[ii](jj) = srp[ii]->getHabPatches()[jj]->getCapacity();
        }
    }

    for (int ii = 0; ii < timeSteps; ii++) {

        // First, compute the effects of species competition
        // Not implemented yet

        // Next, compute animal movement and road mortality


        // Finally, account for natural birth and death
    }
}
