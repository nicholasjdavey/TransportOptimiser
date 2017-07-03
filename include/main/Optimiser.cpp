#include "../transportbase.h"

Optimiser::Optimiser() {
    // Initialise nothing. All parameters must be assigned manually.
}

Optimiser::Optimiser(double mr, double cf, unsigned long gens, unsigned
        long popSize, double stopTol, double confInt, double confLvl, unsigned
        long habGridRes, unsigned long surrDimRes, std::string solScheme,
        unsigned long noRuns, Optimiser::Type type, unsigned long sg, double
        msr, bool gpu, Optimiser::ROVType method,
        Optimiser::InterpolationRoutine interp) {

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
    this->maxSampleRate = msr;
    this->gpu = gpu;
    this->method = method;
    this->surrDimRes = surrDimRes;
    this->interp = interp;
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
            br[ii] = brr;
    }
    this->bestRoads = br;

    if (this->interp == Optimiser::CUBIC_SPLINE) {
        this->surrogate.resize(noTests,std::vector<std::vector<
                alglib::spline1dinterpolant>>(noRuns,std::vector<
                alglib::spline1dinterpolant>(this->getSpecies().size())));
    } else if (this->interp == Optimiser::MULTI_LOC_LIN_REG) {
        this->surrogateML.resize(noTests,std::vector<std::vector<
                Eigen::VectorXd>>(noRuns,std::vector<Eigen::VectorXd>(this->
                getSpecies().size())));
    }
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

void Optimiser::optimise(bool plot) {

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

    int commSD = this->getScenario()->getCommoditySD();

    if ((this->getVariableParams()
            ->getCommoditySDMultipliers()(commSD) != 0.0)) {
        // Find the expected present value of each commodity
        std::vector<CommodityPtr> commodities = this->economic->
                getCommodities();
        std::vector<CommodityPtr> fuels = this->economic->getFuels();

        for (int ii = 0; ii < commodities.size(); ii++) {
            commodities[ii]->computeExpPV();
        }

        for (int ii = 0; ii < fuels.size(); ii++) {
            fuels[ii]->computeExpPV();
        }
    }

    Costs::computeUnitRevenue(this->me());

    switch (this->type) {
    case Optimiser::CONTROLLED:
        {
            // We cannot compute expected present values beforehand as usage
            // is not uniform due to the control aspect.
        }
        break;

    default:
        {
            // Nothing special to do
        }
        break;
    }
}

void Optimiser::evaluateSurrogateModelMTE(RoadPtr road, Eigen::VectorXd &pops,
        Eigen::VectorXd& popsSD) {

    std::vector<SpeciesRoadPatchesPtr> speciesRoadPatches = road->
            getSpeciesRoadPatches();

    for (int ii = 0; ii < this->species.size(); ii++) {

        const Eigen::VectorXd& initAAR = speciesRoadPatches[ii]->getInitAAR();

        // We will replace this with our own interpolation routine
        pops(ii) = alglib::spline1dcalc(this->surrogate[2*this->scenario->
                getCurrentScenario()][this->scenario->getRun()][ii],initAAR(
                initAAR.size()-1));
        popsSD(ii) = alglib::spline1dcalc(this->surrogate[2*this->scenario->
                getCurrentScenario()+1][this->scenario->getRun()][ii],initAAR(
                initAAR.size()-1));
    }
}

void Optimiser::evaluateSurrogateModelMTEML(RoadPtr road, Eigen::VectorXd
        &pops, Eigen::VectorXd& popsSD) {

    std::vector<SpeciesRoadPatchesPtr> speciesRoadPatches = road->
            getSpeciesRoadPatches();

    for (int ii = 0; ii < this->species.size(); ii++) {

        Eigen::VectorXd initAAR(1);
        initAAR(0) = speciesRoadPatches[ii]->getInitAAR()(speciesRoadPatches[
                ii]->getInitAAR().size()-1);

        // Use our own interpolation routine
        pops(ii) = Utility::interpolateSurrogate(this->surrogateML[2*this->
                scenario->getCurrentScenario()][this->scenario->getRun()][ii],
                initAAR,this->surrDimRes);
        popsSD(ii) = Utility::interpolateSurrogate(this->surrogateML[2*this->
                scenario->getCurrentScenario()+1][this->scenario->getRun()][
                ii],initAAR,this->surrDimRes);
    }
}

void Optimiser::evaluateSurrogateModelROVCR(RoadPtr road, double& value,
        double& valuesd) {

    // PREDICTORS /////////////////////////////////////////////////////////////
    Eigen::VectorXd predictors(this->species.size() + 1);

    // INITIAL AARS
    for (int ii = 0; ii < this->species.size(); ii++) {
        const Eigen::VectorXd& initAAR = road->getSpeciesRoadPatches()[ii]->
                getInitAAR();
        predictors(ii) = initAAR(initAAR.size()-1);
    }

    // INITIAL PERIOD UNIT PROFIT
    // For now, we use the initial unit profit. This is saved as a road
    // attribute.

    // Fixed cost per unit traffic
    double unitCost = road->getAttributes()->getUnitVarCosts();
    // Fuel consumption per vehicle class per unit traffic (L)
    Eigen::VectorXd fuelCosts = road->getCosts()->getUnitFuelCost();
    Eigen::VectorXd currentFuelPrice(fuelCosts.size());

    for (int ii = 0; ii < fuelCosts.size(); ii++) {
        currentFuelPrice(ii) = (road->getOptimiser()->getTraffic()->
                getVehicles())[ii]->getFuel()->getCurrent();
    }

    unitCost += fuelCosts.transpose()*currentFuelPrice;

    // As the revenue per unit traffic is the same for each road, we leave it
    // out for now.
    // Load per unit traffic
    // double unitRevenue = road->getCosts()->getUnitRevenue();

    road->getAttributes()->setInitialUnitCost(unitCost);
    predictors(species.size()) = unitCost;

    // Interpolate the multivariate grid using these values

    // MEAN
    value = Utility::interpolateSurrogate(this->surrogateML[2*this->scenario->
            getCurrentScenario()][this->scenario->getRun()][0],predictors,
            this->surrDimRes);

    // STANDARD DEVIATION
    valuesd = Utility::interpolateSurrogate(this->surrogateML[2*this->
            scenario->getCurrentScenario()+1][this->scenario->getRun()][0],
            predictors,this->surrDimRes);
}
