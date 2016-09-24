#include "../transportbase.h"

// Initialise static values
unsigned long Program::programs = 0;

Program::Program(Eigen::VectorXd* flowRates, Eigen::MatrixXf* switching) {

	this->number = ++programs;
	this->flowRates = *flowRates;
	this->switching = *switching;
}

Program::~Program() {
}
