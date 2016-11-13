#include "../transportbase.h"

Program::Program(const Eigen::VectorXd& flowRates, const Eigen::MatrixXd&
        switching) {
    this->flowRates = flowRates;
    this->switching = switching;
}

Program::~Program() {
}
