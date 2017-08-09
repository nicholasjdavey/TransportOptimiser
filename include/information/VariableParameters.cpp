#include "../transportbase.h"

VariableParameters::VariableParameters() {
    // Default constructor
    this->populationLevels = Eigen::VectorXd::Zero(0);
    this->animalBridge = Eigen::VectorXi::Zero(0);
    this->habPref = Eigen::VectorXd::Zero(0);
    this->lambda = Eigen::VectorXd::Zero(0);
    this->beta = Eigen::VectorXd::Zero(0);
    this->popGR = Eigen::VectorXd::Zero(0);
    this->popGRSD = Eigen::VectorXd::Zero(0);
    this->commoditySD = Eigen::VectorXd::Zero(0);
    this->commodity = Eigen::VectorXd::Zero(0);
    this->commodityPropSD = Eigen::VectorXd::Zero(0);
}

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
