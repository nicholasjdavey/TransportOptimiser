#include "../transportbase.h"

ExperimentalScenario::ExperimentalScenario(OptimiserPtr optimiser) {
    this->optimiser = optimiser;
    this->program = 0;
    this->popLevel = 0;
    this->habPrefSD = 0;
    this->lambdaSD = 0;
    this->rangingCoeffSD = 0;
    this->animalBridge = 0;
    this->popGR = 0;
    this->fuel = 0;
    this->commodity = 0;
}

ExperimentalScenario::ExperimentalScenario(OptimiserPtr optimiser, int program,
        int popLevel, int habPrefSD, int lambdaSD, int rangingCoeffSD,
        int animalBridge, int popGR, int fuel, int commodity) {

    this->optimiser = optimiser;
    this->program = program;
    this->popLevel = popLevel;
    this->habPrefSD = habPrefSD;
    this->lambdaSD = lambdaSD;
    this->rangingCoeffSD = rangingCoeffSD;
    this->animalBridge = animalBridge;
    this->popGR = popGR;
    this->fuel = fuel;
    this->commodity = commodity;
}

ExperimentalScenario::~ExperimentalScenario() {}
