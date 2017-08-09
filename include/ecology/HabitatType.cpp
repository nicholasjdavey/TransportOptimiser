#include "../transportbase.h"

// Initialise static variables
Eigen::VectorXi HabitatType::allVegetations =
        Eigen::VectorXi::LinSpaced(0,4,1);

HabitatType::HabitatType() {
	this->type = HabitatType::OTHER;
}

HabitatType::HabitatType(HabitatType::habType typ, double maxPop,
        const Eigen::VectorXi &vegetations, double habPrefMean, double
       habPrefSD, double cost) {
    this->type = typ;
    this->maxPop = maxPop;
    this->cost = cost;
    this->vegetations = vegetations;
    this->habPrefMean = habPrefMean;
    this->habPrefSD = habPrefSD;
}

HabitatType::~HabitatType() {
}
