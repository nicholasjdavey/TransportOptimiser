#include "../transportbase.h"

// Initialise static values
unsigned long Program::programs = 0;

Program::Program(const Eigen::VectorXd& flowRates, const Eigen::MatrixXf&
        switching) {

	this->number = ++programs;
    this->flowRates = flowRates;
    this->switching = switching;
}

Program::~Program() {
}
