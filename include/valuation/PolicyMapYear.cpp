#include "../transportbase.h"

PolicyMapYear::PolicyMapYear(const std::vector<PolicyMapFrontierPtr>&
        frontiers, const Eigen::MatrixXd& stateLevels, const
        Eigen::VectorXd& expectedProfit, const Eigen::VectorXi&
        optimalControls) {

    this->frontiers = frontiers;
    this->stateLevels = stateLevels;
    this->expectedProfit = expectedProfit;
    this->optimalControl = optimalControls;
}

PolicyMapYear::PolicyMapYear(unsigned long noDims, unsigned long noPaths) {
    this->stateLevels.resize(noPaths,noDims);
    this->expectedProfit.resize(noPaths);
    this->optimalControl.resize(noPaths);
}

PolicyMapYear::~PolicyMapYear() {}
