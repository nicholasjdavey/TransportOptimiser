#include "../transportbase.h"

VariableParameters::VariableParameters(const Eigen::VectorXd& popLevels, const
        Eigen::VectorXi& bridge, const Eigen::VectorXd& hp, const
        Eigen::VectorXd& l, const Eigen::VectorXd& b, const Eigen::VectorXd&
        pgr, const Eigen::VectorXd& pgrsd, const Eigen::VectorXd& c, const
        Eigen::VectorXd& csd, Eigen::VectorXd &cpsd) {
	
	// Initialise object
    this->populationLevels = popLevels;
    this->animalBridge = bridge;
    this->habPref = hp;
    this->lambda = l;
    this->beta = b;
    this->popGR = pgr;
    this->popGRSD = pgrsd;
    this->commoditySD = csd;
    this->commodity = c;
    this->commodityPropSD = cpsd;
}

VariableParameters::~VariableParameters() {

}
