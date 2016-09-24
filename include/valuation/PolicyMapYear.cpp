#include "../transportbase.h"

PolicyMapYear::PolicyMapYear(std::vector<PolicyMapFrontierPtr>* frontiers,
		Eigen::MatrixXf* stateLevels, Eigen::VectorXd* expectedProfit) {

	this->frontiers = *frontiers;
	this->stateLevels = *stateLevels;
	this->expectedProfit = *expectedProfit;
}

PolicyMapYear::PolicyMapYear() {
}

PolicyMapYear::~PolicyMapYear() {}
