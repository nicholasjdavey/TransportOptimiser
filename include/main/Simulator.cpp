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
    TrafficProgramPtr program = (road->getOptimiser()
            ->getPrograms())[road->getOptimiser()->getScenario()
            ->getProgram()];
    int controls = program->getFlowRates().size();
    int timeSteps = road->getOptimiser()->getEconomic()->getYears();
    double stepSize = road->getOptimiser()->getEconomic()->getTimeStep();

    std::vector<SpeciesRoadPatchesPtr> srp = road->
            getSpeciesRoadPatches();

    // First, compute the IARs for the road
    Eigen::VectorXd iar(srp.size());

    for (int ii = 0; ii < srp.size(); ii++) {
        Eigen::VectorXd iarsCont(controls);
        srp[ii]->createSpeciesModel();
        srp[ii]->computeInitialAAR(iarsCont);

        iar(ii) = iarsCont(controls-1);
    }

    road->getAttributes()->setIAR(iar);

    // Next, simulate the animal model. For this, we compute each species at each
    // time step
    for (int ii = 0; ii < timeSteps; ii++) {

        // First, compute the effects of species competition

        // Next, compute animal movement and road mortality

        // Finally, account for natural birth and death
    }
}
