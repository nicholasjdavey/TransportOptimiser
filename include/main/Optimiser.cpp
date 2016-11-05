#include "../transportbase.h"

Optimiser::Optimiser() {
    // Initialise nothing. All parameters must be assigned manually.
}

Optimiser::Optimiser(const std::vector<TrafficProgramPtr>& programs,
        OtherInputsPtr oInputs, DesignParametersPtr desParams,
        EarthworkCostsPtr earthworks, UnitCostsPtr unitCosts,
        VariableParametersPtr varParams, const std::vector<SpeciesPtr>&
        species, EconomicPtr economic, TrafficPtr traffic, RegionPtr region,
        double mr, unsigned long cf, unsigned long gens, unsigned long popSize,
        double stopTol, double confInt, double confLvl, unsigned long
        habGridRes, std::string solScheme, unsigned long noRuns,
        Optimiser::Type type, double elite) {

//	std::vector<RoadPtr>* crp(new std::vector<RoadPtr>());
    this->type = type;
    Eigen::MatrixXd currPop(popSize,3*(desParams->getIntersectionPoints()+2));
    this->currentRoadPopulation = currPop;

    unsigned long noTests = (varParams->getPopulationLevels().size())*
            (varParams->getHabPref().size())*(varParams->getLambda().
            size())*(varParams->getBeta().size())*
            (varParams->getGrowthRatesMultipliers().size())*
            (varParams->getGrowthRateSDMultipliers().size())*
            (varParams->getCommodityMultipliers().size())*
            (varParams->getCommoditySDMultipliers().size())*
            (varParams->getAnimalBridge().size());

    std::vector< std::vector<RoadPtr> > br(noTests);

    for(unsigned int ii=0; ii<noTests;ii++) {
            std::vector<RoadPtr> brr(noRuns);
            br.push_back(brr);
    }
    this->bestRoads = br;

//	std::vector<ProgramPtr>* programs(new std::vector<ProgramPtr>());
    unsigned long const hardware_threads = std::thread::hardware_concurrency();
    ThreadManagerPtr threader(new ThreadManager(hardware_threads));
    this->threader = threader;
    this->programs = programs;
    this->otherInputs = oInputs;
    this->designParams = desParams;
    this->earthworks = earthworks;
    this->economic = economic;
    this->traffic = traffic;
    this->region = region;
    this->earthworks = earthworks;
    this->unitCosts = unitCosts;
    this->species = species;
    this->mutationRate = mr;
    this->crossoverFrac = cf;
    this->variableParams = varParams;
    this->generations = gens;
    this->noRuns = noRuns;
    this->populationSizeGA = popSize;
    this->stoppingTol = stopTol;
    this->confInt = confInt;
    this->confLvl = confLvl;
    this->habGridRes = habGridRes;
    this->solutionScheme = solScheme;
}

Optimiser::~Optimiser() {
}

OptimiserPtr Optimiser::me() {
    return shared_from_this();
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

        Eigen::VectorXd initAAR = speciesRoadPatches[ii]->getInitAAR();

        pops(ii) = alglib::spline1dcalc(this->surrogate[2*this->scenario->
                getCurrentScenario()][this->scenario->getRun()][ii],initAAR(
                initAAR.size()));
        popsSD(ii) = alglib::spline1dcalc(this->surrogate[2*this->scenario->
                getCurrentScenario()+1][this->scenario->getRun()][ii],initAAR(
                initAAR.size()));
    }
}

void Optimiser::evaluateSurrogateModelROVCR(RoadPtr road, double use, double
        usesd) {

}
