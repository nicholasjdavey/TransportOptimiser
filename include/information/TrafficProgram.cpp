#include "../transportbase.h"

TrafficProgram::TrafficProgram(const Eigen::VectorXd &flowRates,
        const Eigen::MatrixXf &switching) : Program(flowRates, switching) {
}

TrafficProgram::TrafficProgram(bool br, TrafficPtr traffic,
        const Eigen::VectorXd &flowRates, const Eigen::MatrixXf &switching)
		: Program(flowRates, switching) {
	this->bridge = br;
	this->traffic = traffic;
}

TrafficProgram::~TrafficProgram() {
}
