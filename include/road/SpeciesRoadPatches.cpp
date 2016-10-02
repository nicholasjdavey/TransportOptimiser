#include "../transportbase.h"

SpeciesRoadPatches::SpeciesRoadPatches(SpeciesPtr species, RoadPtr road) :
        Uncertainty() {
    this->species = species;
    this->road = road;
}

SpeciesRoadPatches::SpeciesRoadPatches(SpeciesPtr species, RoadPtr road,
        bool active, double mean, double stdDev, double rev,
        std::string nm) : Uncertainty(nm, mean, stdDev, rev, active) {

    this->species = species;
    this->road = road;
}

SpeciesRoadPatches::~SpeciesRoadPatches() {}

Eigen::MatrixXd SpeciesRoadPatches::habitatPatchDistances() {
    // First copy relevant HabitatPatch components to vectors for ease of use
    int hps = this->habPatch.size();

    Eigen::VectorXd xorg = Eigen::VectorXd::Zero(hps);
    Eigen::VectorXd yorg = Eigen::VectorXd::Zero(hps);

    for (int ii = 0; ii < hps; ii++) {
        xorg(ii) = this->habPatch[ii]->getCX();
        yorg(ii) = this->habPatch[ii]->getCY();
    }

    Eigen::MatrixXd xDests(hps,hps);
    Eigen::MatrixXd yDests(hps,hps);
    Eigen::MatrixXd xOrgs(hps,hps);
    Eigen::MatrixXd yOrgs(hps,hps);

    igl::repmat(xorg,1,hps,xDests);
    igl::repmat(yorg,1,hps,yDests);
    xOrgs = xDests.transpose();
    yOrgs = yDests.transpose();

    return ((xDests - xOrgs).array().pow(2) + (yDests - yOrgs).array().pow(2))
            .array().sqrt().matrix();
}

Eigen::MatrixXi SpeciesRoadPatches::roadCrossings() {
    Eigen::VectorXd* px = this->road->getRoadSegments()->getX();
    Eigen::VectorXd* py = this->road->getRoadSegments()->getY();
    Eigen::VectorXi* typ = this->road->getRoadSegments()
            ->getType();
    int noSegs = px->size()-1;

    // First copy relevant HabitatPatch components to vectors for ease of use
    Eigen::MatrixXd roadSegsVisible(noSegs,4);
    roadSegsVisible.block(0,0,noSegs,1) = px->segment(0,noSegs);
    roadSegsVisible.block(0,1,noSegs,1) = py->segment(0,noSegs);
    roadSegsVisible.block(0,2,noSegs,1) = px->segment(1,noSegs);
    roadSegsVisible.block(0,3,noSegs,1) = py->segment(1,noSegs);

    Eigen::MatrixXd typMat(noSegs,4);
    typMat.block(0,0,noSegs,1) = (typ->array() == (int)RoadSegments::ROAD)
            .cast<double>();
    typMat.block(0,2,noSegs,1) = (typ->array() == (int)RoadSegments::ROAD)
            .cast<double>();
    typMat.block(0,2,noSegs,1) = (typ->array() == (int)RoadSegments::ROAD)
            .cast<double>();
    typMat.block(0,2,noSegs,1) = (typ->array() == (int)RoadSegments::ROAD)
            .cast<double>();

    // We reduce the computation time by only considering cells that are close
    // enough to the originating cell (i.e. they are close enough to each other
    // that a transition is likely in the absence of a road). We use the 5th
    // percentile distance based on the simple gamma distribution used in
    // Rhodes et al. 2014.
    //
    // Need to alter to take into account the lambda currently being tested
    // from optimiser->variableParameters
    double lda = (*this->road->getOptimiser()->getVariableParams()
            ->getLambda())(this->road->getOptimiser()->getScenario()
            ->getLambda());
    double maxDist = log(0.05)/(-lda);
}
