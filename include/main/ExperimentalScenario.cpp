#include "../transportbase.h"

ExperimentalScenario::ExperimentalScenario(OptimiserPtr optimiser) {
    this->optimiser = optimiser;
    this->program = 0;
    this->popLevel = 0;
    this->habPref = 0;
    this->lambda = 0;
    this->rangingCoeff = 0;
    this->animalBridge = 0;
    this->popGR = 0;
    this->fuel = 0;
    this->commodity = 0;
}

ExperimentalScenario::ExperimentalScenario(OptimiserPtr optimiser, int program,
        int popLevel, int habPref, int lambda, int rangingCoeff,
        int animalBridge, int popGR, int fuel, int commodity) {

    this->optimiser = optimiser;
    this->program = program;
    this->popLevel = popLevel;
    this->habPref = habPref;
    this->lambda = lambda;
    this->rangingCoeff = rangingCoeff;
    this->animalBridge = animalBridge;
    this->popGR = popGR;
    this->fuel = fuel;
    this->commodity = commodity;
}

ExperimentalScenario::~ExperimentalScenario() {}
