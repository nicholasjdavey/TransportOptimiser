#include "../transportbase.h"

VariableParameters::VariableParameters(const Eigen::VectorXd &popLevels,
        const Eigen::VectorXd &bridge, const Eigen::VectorXd &hp, const
        Eigen::VectorXd &l, const Eigen::VectorXd &b, const Eigen::VectorXd
        &pgr, const Eigen::VectorXd &f, const Eigen::VectorXd &c) {
	
	// Initialise object
    this->populationLevels = popLevels;
    this->animalBridge = bridge;
    this->habPref = hp;
    this->lambda = l;
    this->beta = b;
    this->popGR = pgr;
    this->fuel = f;
    this->commodity = c;
}
