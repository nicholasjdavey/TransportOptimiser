#include "../transportbase.h"

MonteCarloROV::MonteCarloROV() {
	this->randGenerator = "";
	this->value = 0;
}

MonteCarloROV::~MonteCarloROV() {
}

void MonteCarloROV::simulateROVCR() {
    // Simulate forward paths using multiple threads. Save values of
    // each uncertainty
}

void MonteCarloROV::buildPolicyMap() {

}

void MonteCarloROV::simulateForwardPaths() {
    // For each time step, call the uncertainty computation for each
    // uncertainty
}

void MonteCarloROV::randomState(StatePtr currState) {

}
