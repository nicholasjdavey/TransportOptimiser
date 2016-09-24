#include "../transportbase.h"

State::State(double t, ProgramPtr p, std::vector<UncertaintyPtr>* ex,
			std::vector<UncertaintyPtr>* en) {

	this->time = t;
	this->program = p;
	this->exogenousUncertainties = *ex;
	this->endogenousUncertainties = *en;
}

State::~State() {}
