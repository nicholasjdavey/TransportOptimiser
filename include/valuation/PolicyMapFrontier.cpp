#include "../transportbase.h"

PolicyMapFrontier::PolicyMapFrontier() {
}

PolicyMapFrontier::PolicyMapFrontier(unsigned long base,
		unsigned long proposed) {

	this->baseOption = base;
	this->proposedOption = proposed;
}

PolicyMapFrontier::PolicyMapFrontier(unsigned long base,
        unsigned long proposed, const Eigen::MatrixXd &lvls,
        const Eigen::VectorXd &unitProfit) {

	this->baseOption = base;
	this->proposedOption = proposed;
    this->stateLevels = lvls;
    this->unitProfit = unitProfit;
}

PolicyMapFrontier::~PolicyMapFrontier() {
}
