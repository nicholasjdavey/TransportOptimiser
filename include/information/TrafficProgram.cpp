#include "../transportbase.h"

TrafficProgram::TrafficProgram(Eigen::VectorXd* flowRates,
		Eigen::MatrixXf* switching) : Program(flowRates, switching) {
}

TrafficProgram::TrafficProgram(bool br, TrafficPtr traffic,
		Eigen::VectorXd* flowRates, Eigen::MatrixXf* switching)
		: Program(flowRates, switching) {
	this->bridge = br;
	this->traffic = traffic;
}

TrafficProgram::~TrafficProgram() {
}
