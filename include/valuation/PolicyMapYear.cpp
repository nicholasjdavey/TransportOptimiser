#include "../transportbase.h"

PolicyMapYear::PolicyMapYear(const std::vector<PolicyMapFrontierPtr>&
        frontiers, const Eigen::MatrixXd& stateLevels, const
        Eigen::VectorXd& expectedProfit) {

    this->frontiers = frontiers;
    this->stateLevels = stateLevels;
    this->expectedProfit = expectedProfit;
}

PolicyMapYear::PolicyMapYear() {
}

PolicyMapYear::~PolicyMapYear() {}
