#include "../transportbase.h"

State::State(double t, ProgramPtr p, const std::vector<UncertaintyPtr> &ex,
            const std::vector<UncertaintyPtr> &en) {

	this->time = t;
	this->program = p;
    this->exogenousUncertainties = ex;
    this->endogenousUncertainties = en;
}

State::~State() {}
