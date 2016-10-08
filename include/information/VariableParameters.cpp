#include "../transportbase.h"

VariableParameters::VariableParameters(Eigen::VectorXd* popLevels,
        Eigen::VectorXd *bridge, Eigen::VectorXd* hp, Eigen::VectorXd* l,
        Eigen::VectorXd* b, Eigen::VectorXd *pgr, Eigen::VectorXd *f,
        Eigen::VectorXd *c) {
	
	// Initialise object
	this->populationLevels = *popLevels;
    this->animalBridge = *bridge;
	this->habPref = *hp;
	this->lambda = *l;
	this->beta = *b;
    this->popGR = *pgr;
    this->fuel = *f;
    this->commodity = *c;
}
