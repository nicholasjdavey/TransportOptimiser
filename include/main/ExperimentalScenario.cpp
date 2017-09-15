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
    this->popGRSD;
    this->commodity = 0;
    this->commoditySD = 0;
    this->currentScenario = 0;
    this->oreCompSD = 0;
    this->compRoad = 0;
    this->run = 0;
}

ExperimentalScenario::ExperimentalScenario(OptimiserPtr optimiser, int program,
        int popLevel, int habPref, int lambda, int rangingCoeff,
        int animalBridge, int popGR, int popGRSD, int commodity,
        int commoditySD, int ore, int cr, int run) {

    this->optimiser = optimiser;
    this->program = program;
    this->popLevel = popLevel;
    this->habPref = habPref;
    this->lambda = lambda;
    this->rangingCoeff = rangingCoeff;
    this->animalBridge = animalBridge;
    this->popGR = popGR;
    this->popGRSD = popGRSD;
    this->commodity = commodity;
    this->commoditySD = commoditySD;
    this->run = run;
    this->compRoad = cr;
    this->oreCompSD = ore;
}

ExperimentalScenario::~ExperimentalScenario() {}

void ExperimentalScenario::computeScenarioNumber() {

    OptimiserPtr optimiser = this->optimiser.lock();
    int hps = optimiser->getVariableParams()->getHabPref().size();
    int ls = optimiser->getVariableParams()->getLambda().size();
    int bs = optimiser->getVariableParams()->getBeta().size();
    int grms = optimiser->getVariableParams()->
            getGrowthRatesMultipliers().size();
    int grsdms = optimiser->getVariableParams()->
            getGrowthRateSDMultipliers().size();
    int cms = optimiser->getVariableParams()->
            getCommodityMultipliers().size();
    int csdms = optimiser->getVariableParams()->
            getCommoditySDMultipliers().size();
    int ocsdm = optimiser->getVariableParams()->
            getCommodityPropSD().size();
    int abs = optimiser->getVariableParams()->getBridge().size();

    // Experimental scenario also needs to save the surrogate model and
    // performance metrics. Put this in the optimiser class.

    int ii = this->getPopLevel();
    int jj = this->getHabPref();
    int kk = this->getLambda();
    int ll = this->getRangingCoeff();
    int mm = this->getPopGR();
    int nn = this->getPopGRSD();
    int oo = this->getCommodity();
    int pp = this->getCommoditySD();
    int qq = this->getAnimalBridge();
    int rr = this->getOreCompositionSD();

    this->currentScenario = ii*hps*ls*bs*grms*grsdms*cms*csdms*ocsdm*abs +
            jj*ls*bs*grms*grsdms*cms*csdms*ocsdm*abs + kk*bs*grms*grsdms*cms*
            csdms*ocsdm*abs + ll*grms*grsdms*cms*csdms*ocsdm*abs + mm*grsdms*
            cms*csdms*ocsdm*abs + nn*cms*csdms*ocsdm*abs + oo*csdms*ocsdm*abs
            + pp*ocsdm*abs + qq*abs + rr;
}
