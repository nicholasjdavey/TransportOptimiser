#include "../transportbase.h"

VariableParameters::VariableParameters(Eigen::VectorXd* popLevels, bool bridge,
		Eigen::VectorXd* hp, Eigen::VectorXd* l, Eigen::VectorXd* b,
		bool pgr, bool f, bool c) {
	
	// Initialise object
	this->populationLevels = *popLevels;
	this->animalBridge = bridge;
	this->habPref = *hp;
	this->lambda = *l;
	this->beta = *b;
	this->popGR = pgr;
	this->fuel = f;
	this->commodity = c;
}
