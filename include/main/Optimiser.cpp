#include "../transportbase.h"

Optimiser::Optimiser() {
    // Initialise nothing. All parameters must be assigned manually.
}

Optimiser::Optimiser(double mr, double cf, unsigned long gens, unsigned
        long popSize, double stopTol, double confInt, double confLvl, unsigned
        long habGridRes, std::string solScheme, unsigned long noRuns,
        Optimiser::Type type, unsigned long sg) {

//	std::vector<ProgramPtr>* programs(new std::vector<ProgramPtr>());
    unsigned long const hardware_threads = std::thread::hardware_concurrency();
//    ThreadManagerPtr threader(new ThreadManager(hardware_threads));
    this->threader = threader;
    this->mutationRate = mr;
    this->crossoverFrac = cf;
    this->generations = gens;
    this->noRuns = noRuns;
    this->populationSizeGA = popSize;
    this->stoppingTol = stopTol;
    this->confInt = confInt;
    this->confLvl = confLvl;
    this->habGridRes = habGridRes;
    this->solutionScheme = solScheme;
    this->stallGenerations = sg;
    this->stallGen = 0;
    this->type = type;
}

Optimiser::~Optimiser() {
}

OptimiserPtr Optimiser::me() {
    return shared_from_this();
}

void Optimiser::initialiseStorage() {
    //	std::vector<RoadPtr>* crp(new std::vector<RoadPtr>());

    Eigen::MatrixXd currPop(this->populationSizeGA,3*(this->designParams->
            getIntersectionPoints()+2));
    this->currentRoadPopulation = currPop;

    unsigned long noTests = (this->variableParams->getPopulationLevels().
            size())*(this->variableParams->getHabPref().size())*(this->
            variableParams->getLambda().size())*(this->variableParams->
            getBeta().size())*(this->variableParams->
            getGrowthRatesMultipliers().size())*
            (this->variableParams->getGrowthRateSDMultipliers().size())*
            (this->variableParams->getCommodityMultipliers().size())*
            (this->variableParams->getCommoditySDMultipliers().size())*
            (this->variableParams->getAnimalBridge().size());

    std::vector< std::vector<RoadPtr> > br(this->noRuns);

    for(unsigned int ii=0; ii<this->noRuns;ii++) {
            std::vector<RoadPtr> brr(this->noRuns);
            br.push_back(brr);
    }
    this->bestRoads = br;

    this->surrogate.resize(noTests,std::vector<std::vector<
            alglib::spline1dinterpolant>>(noRuns,std::vector<
            alglib::spline1dinterpolant>(this->getSpecies().size())));
}

void Optimiser::computeHabitatMaps() {
    for (SpeciesPtr species: this->species) {
        species->generateHabitatMap(this->me());
    }
}

void Optimiser::computeExpPv() {
    // Commodities
    for (CommodityPtr commodity : this->getEconomic()->getCommodities()) {
        commodity->computeExpPV();
    }

    // Fuels
    for (CommodityPtr fuel : this->getEconomic()->getFuels()) {
        fuel->computeExpPV();
    }
}

void Optimiser::optimiseRoad() {

    // FOR SIMPLEPENALTY  AND MTE /////////////////////////////////////////////
    // If Fuel and commodity prices are stochastic, compute the mean expected
    // PV of operating one unit of traffic consistently over the horizon:
    // i.e. call <Uncertainty>->computeExpPV() for each uncertainty before
    // starting the computation.

    // FOR ROV ////////////////////////////////////////////////////////////////
    // Uncertainties are calculated when simulating each road

    // FOR MTE AND ROV ////////////////////////////////////////////////////////
    // Compute using surrogate function, which we learn at each iteration.
    // Perform sampling to ensure a good distribution of AARs.
}

void Optimiser::evaluateSurrogateModelMTE(RoadPtr road, Eigen::VectorXd
        &pops, Eigen::VectorXd& popsSD) {

    std::vector<SpeciesRoadPatchesPtr> speciesRoadPatches =
            road->getSpeciesRoadPatches();

    for (int ii = 0; ii < this->species.size(); ii++) {

        const Eigen::VectorXd& initAAR = speciesRoadPatches[ii]->getInitAAR();

        pops(ii) = alglib::spline1dcalc(this->surrogate[2*this->scenario->
                getCurrentScenario()][this->scenario->getRun()][ii],initAAR(
                initAAR.size()-1));
        popsSD(ii) = alglib::spline1dcalc(this->surrogate[2*this->scenario->
                getCurrentScenario()+1][this->scenario->getRun()][ii],initAAR(
                initAAR.size()-1));
    }
}

void Optimiser::evaluateSurrogateModelROVCR(RoadPtr road, double use, double
        usesd) {

}
