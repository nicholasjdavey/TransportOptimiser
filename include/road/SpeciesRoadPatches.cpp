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
    // First copy relevant HabitatPatch components to vectors for ease of use
    Eigen::VectorXd* px;
}
