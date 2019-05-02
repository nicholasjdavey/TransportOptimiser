#include "../transportbase.h"

Optimiser::Optimiser() {
    // Initialise nothing. All parameters must be assigned manually.

    // Determine the number of devices in the machine
    this->gpus = SimulateGPU::deviceCount();

    // This may never be called if CPU threading is not enabled
    ThreadManagerPtr threaderGPU(new ThreadManager(this->gpus));
    this->setThreadManagerGPU(threaderGPU);
}

Optimiser::Optimiser(double mr, double cf, unsigned long gens, unsigned
        long popSize, double stopTol, double confInt, double confLvl, unsigned
        long habGridRes, unsigned long surrDimRes, std::string solScheme,
        unsigned long noRuns, Optimiser::Type type, unsigned long sg, double
        msr, unsigned long learnSamples, bool gpu, Optimiser::ROVType method,
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
    this->learnSamples = learnSamples;

    // Determine the number of devices in the machine
    this->gpus = SimulateGPU::deviceCount();

    ThreadManagerPtr threaderGPU(new ThreadManager(this->gpus));
    this->setThreadManagerGPU(threaderGPU);
}

Optimiser::~Optimiser() {
}

OptimiserPtr Optimiser::me() {
    return shared_from_this();
}

void Optimiser::initialiseExperimentStorage() {
    int prog = this->getPrograms().size();
    int pl = this->variableParams->getPopulationLevels().rows();
    int hp = this->variableParams->getHabPref().rows();
    int lambda = this->variableParams->getLambda().rows();
    int beta = this->variableParams->getBeta().rows();
    int ab = this->variableParams->getAnimalBridge().rows();
    int grm = this->variableParams->getGrowthRatesMultipliers().rows();
    int grsd = this->variableParams->getGrowthRateSDMultipliers().rows();
    int comm = this->variableParams->getCommodityMultipliers().rows();
    int commSDM = this->variableParams->getCommoditySDMultipliers().rows();
    int commProp = this->variableParams->getCommodityPropSD().rows();
    int compRoad = this->variableParams->getCompRoad().rows();

    unsigned long noTests = prog*pl*hp*lambda*beta*ab*grm*grsd*comm*commSDM*
            commProp*compRoad;

    std::vector< std::vector<RoadPtr> > br(noTests);

    for(unsigned int ii=0; ii< noTests;ii++) {
            std::vector<RoadPtr> brr(this->noRuns);
            br[ii] = brr;
    }
    this->bestRoads = br;

    if (this->interp == Optimiser::CUBIC_SPLINE) {
        this->surrogate.resize(noTests*2,std::vector<std::vector<
                alglib::spline1dinterpolant>>(noRuns,std::vector<
                alglib::spline1dinterpolant>(this->getSpecies().size())));
    } else if (this->interp == Optimiser::MULTI_LOC_LIN_REG) {
        this->surrogateML.resize(noTests*2,std::vector<std::vector<
                Eigen::VectorXd>>(noRuns,std::vector<Eigen::VectorXd>(this->
                getSpecies().size())));
    }
}

void Optimiser::initialiseStorage() {
    //	std::vector<RoadPtr>* crp(new std::vector<RoadPtr>());

    Eigen::MatrixXd currPop(this->populationSizeGA,3*(this->designParams->
            getIntersectionPoints()+2));
    this->currentRoadPopulation = currPop;
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

    // If we have a comparison road, we know that the maximum operating value
    // for ROV cannot exceed the total operating cost for this road (this would
    // imply that the company is PAID to use fuel, which is illogical).
    if (this->comparisonRoad != nullptr) {
        AttributesPtr attributes(new Attributes(comparisonRoad));
        comparisonRoad->setAttributes(attributes);

        comparisonRoad->computeVarProfitICFixedFlow();

        this->maxROVBenefit = comparisonRoad->getAttributes()->getVarProfitIC()
                *this->getVariableParams()->getCompRoad()(this->getScenario()->
                getCompRoad());
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

void Optimiser::saveRunPopulation() {

}

void Optimiser::saveExperimentalResults() {
    // Save Experimental Scenario
    // Ensure results folder is created in the correct location
    std::string scenarioFolder = this->getRootFolder() + "/" + "Scenario_" +
            std::to_string(this->scenario->getCurrentScenario());

    std::string scenarioFile = scenarioFolder + "/Scenario_Configuration";

    if(!(boost::filesystem::exists(scenarioFolder))) {
        boost::filesystem::create_directory(scenarioFolder);
    }
    std::ifstream outputFileCheck;
    std::ofstream outputFile;

    outputFileCheck.open(scenarioFile);

    if (!(outputFileCheck.good())) {
        outputFileCheck.close();

        outputFile.open(scenarioFile);
        outputFile << "####################################################################################################" << std::endl;
        outputFile << "##################################### SCENARIO CONFIGURATION #######################################" << std::endl;
        outputFile << "####################################################################################################" << std::endl;
        outputFile << std::endl;

        // Save scenario details
        // Input files
        outputFile << "ROOT FOLDER                  : " << this->rootFolder << std::endl;
        outputFile << "X VALUES FILE                : " << this->xValuesFile << std::endl;
        outputFile << "Y VALUES FILE                : " << this->yValuesFile << std::endl;
        outputFile << "Z VALUES FILE                : " << this->zValuesFile << std::endl;
        outputFile << "VEGETATION FILE              : " << this->vegetationFile << std::endl;
        outputFile << "ACQUISITION FILE             : " << this->acquisitionFile << std::endl;
        outputFile << "SOIL FILE                    : " << this->soilFile << std::endl;
        outputFile << "COMMODITIES FILES            : ";

        for (int ii = 0; ii < this->commoditiesFiles.size(); ii++) {
            outputFile << this->commoditiesFiles[ii];

            if (ii < (this->commoditiesFiles.size() - 1)) {
                outputFile << ",";
            }
        }
        outputFile << std::endl;
        outputFile << "FUELS FILES                  : ";

        for (int ii = 0; ii < this->fuelsFiles.size(); ii++) {
            outputFile << this->fuelsFiles[ii];

            if (ii < (this->fuelsFiles.size() - 1)) {
                outputFile << ",";
            }
        }
        outputFile << std::endl;
        outputFile << "VEHICLES FILES               : ";

        for (int ii = 0; ii < this->vehiclesFiles.size(); ii++) {
            outputFile << this->vehiclesFiles[ii];

            if (ii < (this->vehiclesFiles.size() - 1)) {
                outputFile << ",";
            }
        }
        outputFile << std::endl;
        outputFile << "SPECIES FILES                : ";

        for (int ii = 0; ii < this->speciesFiles.size(); ii++) {
            outputFile << this->speciesFiles[ii];

            if (ii < (this->speciesFiles.size() - 1)) {
                outputFile << ",";
            }
        }
        outputFile << std::endl << std::endl;

        // Scenario configuration
        outputFile << "# See variable parameters in the input file for the values corresponding to the below indices" << std::endl;
        outputFile << "PROGRAM                      : " << this->scenario->getProgram() << std::endl;
        outputFile << "POPULATION THRESHOLD         : " << this->scenario->getPopLevel() << std::endl;
        outputFile << "HABITAT PREFERENCE           : " << this->scenario->getHabPref() << std::endl;
        outputFile << "MOVEMENT PROPENSITY          : " << this->scenario->getLambda() << std::endl;
        outputFile << "RANGING COEFFICIENT          : " << this->scenario->getRangingCoeff() << std::endl;
        outputFile << "ANIMAL BRIDGE USE            : " << this->scenario->getAnimalBridge() << std::endl;
        outputFile << "POPULATION GROWTH RATE       : " << this->scenario->getPopGR() << std::endl;
        outputFile << "POPULATION GROWTH RATE SD    : " << this->scenario->getPopGRSD() << std::endl;
        outputFile << "COMMODITY MEAN               : " << this->scenario->getCommodity() << std::endl;
        outputFile << "COMMODITY SD                 : " << this->scenario->getCommoditySD() << std::endl;
        outputFile << "ORE COMPOSITION SD           : " << this->scenario->getOreCompositionSD() << std::endl;
        outputFile << "SCENARIO NUMBER              : " << this->scenario->getCurrentScenario() << std::endl;
    }

    outputFile.close();
}

void Optimiser::saveBestRoadResults() {
    // Computes the values
    // (First compute using MTE)
    this->computeBestRoadResults();

    // Save generic road data
    RoadPtr road = this->bestRoads[this->scenario->getCurrentScenario()][this->
            scenario->getRun()];

    // Ensure results folder is created in the correct location
    std::string scenarioFolder = this->getRootFolder() + "/" + "Scenario_" +
            std::to_string(this->scenario->getCurrentScenario());
    std::string runFolder = scenarioFolder + "/" + std::to_string(this->
            scenario->getRun());

    if(!(boost::filesystem::exists(scenarioFolder))) {
        boost::filesystem::create_directory(scenarioFolder);
    }

    if(!(boost::filesystem::exists(runFolder))) {
        boost::filesystem::create_directory(runFolder);
    }

    std::ifstream outputFileCheck;
    std::ofstream outputFile;

    // Save Base Road Results:
    std::string bestRoad = runFolder + "/" + "best_road";
    outputFileCheck.open(bestRoad);

    if (outputFileCheck.good()) {
        outputFileCheck.close();
        std::remove(bestRoad.c_str());
    } else {
        outputFileCheck.close();
    }

    outputFile.open(bestRoad,std::ios::out);

    outputFile << "####################################################################################################" << std::endl;
    outputFile << "######################################## BEST ROAD RESULTS #########################################" << std::endl;
    outputFile << "####################################################################################################" << std::endl;
    outputFile << std::endl;

    // Reason for stopping
    std::string reason;

    switch (this->stop) {
        case Optimiser::STALLED:
            reason = "Maximum stall generations reached";
            break;
        case Optimiser::COMPLETE:
            reason = "Stopping tolerance achieved";
            break;
        case Optimiser::MAX_GENS:
            reason = "Maximum generations reached";
            break;
        default:
            reason = "Unidentified error during run";
    }

    outputFile << "STOPPING REASON                : " << reason << std::endl << std::endl;

    // Intersection Points
    outputFile << "####################################################################################################" << std::endl;
    outputFile << "# INTERSECTION POINTS ##############################################################################" << std::endl;
    outputFile << "X INTERSECTION POINTS: ";
    for (int ii = 0; ii < road->getXCoords().size(); ii++) {
        outputFile << road->getXCoords()(ii);
        if (ii < (road->getXCoords().size() - 1)) {
            outputFile << ",";
        }
    }
    outputFile << std::endl;

    outputFile << "Y INTERSECTION POINTS: ";
    for (int ii = 0; ii < road->getYCoords().size(); ii++) {
        outputFile << road->getYCoords()(ii);
        if (ii < (road->getYCoords().size() - 1)) {
            outputFile << ",";
        }
    }
    outputFile << std::endl;

    outputFile << "Z INTERSECTION POINTS: ";
    for (int ii = 0; ii < road->getZCoords().size(); ii++) {
        outputFile << road->getZCoords()(ii);
        if (ii < (road->getZCoords().size() - 1)) {
            outputFile << ",";
        }
    }
    outputFile << std::endl << std::endl;

    // Horizontal Alignment
    outputFile << "####################################################################################################" << std::endl;
    outputFile << "# HORIZONTAL ALIGNMENT #############################################################################" << std::endl;
    outputFile << "DELTAS:                ";
    for (int ii = 0; ii < road->getHorizontalAlignment()->getDeltas().size(); ii++) {
        outputFile << road->getHorizontalAlignment()->getDeltas()(ii);
        if (ii < (road->getHorizontalAlignment()->getDeltas().size() - 1)) {
            outputFile << ",";
        }
    }
    outputFile << std::endl;

    outputFile << "RADII:                 ";
    for (int ii = 0; ii < road->getHorizontalAlignment()->getRadii().size(); ii++) {
        outputFile << road->getHorizontalAlignment()->getRadii()(ii);
        if (ii < (road->getHorizontalAlignment()->getRadii().size() - 1)) {
            outputFile << ",";
        }
    }
    outputFile << std::endl;

    outputFile << "REQUIRED RADII:        ";
    for (int ii = 0; ii < road->getHorizontalAlignment()->getRadiiReq().size(); ii++) {
        outputFile << road->getHorizontalAlignment()->getRadiiReq()(ii);
        if (ii < (road->getHorizontalAlignment()->getRadiiReq().size() - 1)) {
            outputFile << ",";
        }
    }
    outputFile << std::endl;

    outputFile << "POINTS OF CURVATURE X: ";
    for (int ii = 0; ii < road->getHorizontalAlignment()->getPOCX().size(); ii++) {
        outputFile << road->getHorizontalAlignment()->getPOCX()(ii);
        if (ii < (road->getHorizontalAlignment()->getPOCX().size() - 1)) {
            outputFile << ",";
        }
    }
    outputFile << std::endl;

    outputFile << "POINTS OF CURVATURE Y: ";
    for (int ii = 0; ii < road->getHorizontalAlignment()->getPOCY().size(); ii++) {
        outputFile << road->getHorizontalAlignment()->getPOCY()(ii);
        if (ii < (road->getHorizontalAlignment()->getPOCY().size() - 1)) {
            outputFile << ",";
        }
    }
    outputFile << std::endl;

    outputFile << "POINTS OF TANGENCY X:  ";
    for (int ii = 0; ii < road->getHorizontalAlignment()->getPOTX().size(); ii++) {
        outputFile << road->getHorizontalAlignment()->getPOTX()(ii);
        if (ii < (road->getHorizontalAlignment()->getPOTX().size() - 1)) {
            outputFile << ",";
        }
    }
    outputFile << std::endl;

    outputFile << "POINTS OF TANGENCY Y:  ";
    for (int ii = 0; ii < road->getHorizontalAlignment()->getPOTY().size(); ii++) {
        outputFile << road->getHorizontalAlignment()->getPOTY()(ii);
        if (ii < (road->getHorizontalAlignment()->getPOTY().size() - 1)) {
            outputFile << ",";
        }
    }
    outputFile << std::endl;

    outputFile << "MIDPOINT X:            ";
    for (int ii = 0; ii < road->getHorizontalAlignment()->getMidX().size(); ii++) {
        outputFile << road->getHorizontalAlignment()->getMidX()(ii);
        if (ii < (road->getHorizontalAlignment()->getMidX().size() - 1)) {
            outputFile << ",";
        }
    }
    outputFile << std::endl;

    outputFile << "MIDPOINT Y:            ";
    for (int ii = 0; ii < road->getHorizontalAlignment()->getMidY().size(); ii++) {
        outputFile << road->getHorizontalAlignment()->getMidY()(ii);
        if (ii < (road->getHorizontalAlignment()->getMidY().size() - 1)) {
            outputFile << ",";
        }
    }
    outputFile << std::endl;

    outputFile << "CENTRE OF CURVATURE X: ";
    for (int ii = 0; ii < road->getHorizontalAlignment()->getDelX().size(); ii++) {
        outputFile << road->getHorizontalAlignment()->getDelX()(ii);
        if (ii < (road->getHorizontalAlignment()->getDelX().size() - 1)) {
            outputFile << ",";
        }
    }
    outputFile << std::endl;

    outputFile << "CENTRE OF CURVATURE Y: ";
    for (int ii = 0; ii < road->getHorizontalAlignment()->getDelY().size(); ii++) {
        outputFile << road->getHorizontalAlignment()->getDelY()(ii);
        if (ii < (road->getHorizontalAlignment()->getDelY().size() - 1)) {
            outputFile << ",";
        }
    }
    outputFile << std::endl;

    outputFile << "VELOCITIES:            ";
    for (int ii = 0; ii < road->getHorizontalAlignment()->getVelocities().size(); ii++) {
        outputFile << road->getHorizontalAlignment()->getVelocities()(ii);
        if (ii < (road->getHorizontalAlignment()->getVelocities().size() - 1)) {
            outputFile << ",";
        }
    }
    outputFile << std::endl << std::endl;

    // Vertical Alignment
    outputFile << "####################################################################################################" << std::endl;
    outputFile << "# VERTICAL ALIGNMENT ###############################################################################" << std::endl;
    outputFile << "DISTANCES ALONG CURVE: ";
    for (int ii = 0; ii < road->getVerticalAlignment()->getSDistances().size(); ii++) {
        outputFile << road->getVerticalAlignment()->getSDistances()(ii);
        if (ii < (road->getVerticalAlignment()->getSDistances().size() - 1)) {
            outputFile << ",";
        }
    }
    outputFile << std::endl;

    outputFile << "PTS VERT CURVATURE:    ";
    for (int ii = 0; ii < road->getVerticalAlignment()->getPVCs().size(); ii++) {
        outputFile << road->getVerticalAlignment()->getPVCs()(ii);
        if (ii < (road->getVerticalAlignment()->getPVCs().size() - 1)) {
            outputFile << ",";
        }
    }
    outputFile << std::endl;

    outputFile << "PTS VERTICAL TANGENCY: ";
    for (int ii = 0; ii < road->getVerticalAlignment()->getPVTs().size(); ii++) {
        outputFile << road->getVerticalAlignment()->getPVTs()(ii);
        if (ii < (road->getVerticalAlignment()->getPVTs().size() - 1)) {
            outputFile << ",";
        }
    }
    outputFile << std::endl;

    outputFile << "PTS VERT CURV ELEV:    ";
    for (int ii = 0; ii < road->getVerticalAlignment()->getEPVCs().size(); ii++) {
        outputFile << road->getVerticalAlignment()->getEPVCs()(ii);
        if (ii < (road->getVerticalAlignment()->getEPVCs().size() - 1)) {
            outputFile << ",";
        }
    }
    outputFile << std::endl;

    outputFile << "PTS VERT TAN ELEV:     ";
    for (int ii = 0; ii < road->getVerticalAlignment()->getEPVTs().size(); ii++) {
        outputFile << road->getVerticalAlignment()->getEPVTs()(ii);
        if (ii < (road->getVerticalAlignment()->getEPVTs().size() - 1)) {
            outputFile << ",";
        }
    }
    outputFile << std::endl;

    outputFile << "VELOCITIES:            ";
    for (int ii = 0; ii < road->getVerticalAlignment()->getVelocities().size(); ii++) {
        outputFile << road->getVerticalAlignment()->getVelocities()(ii);
        if (ii < (road->getVerticalAlignment()->getVelocities().size() - 1)) {
            outputFile << ",";
        }
    }
    outputFile << std::endl;

    outputFile << "LENGTHS OF CURVATURE:  ";
    for (int ii = 0; ii < road->getVerticalAlignment()->getLengths().size(); ii++) {
        outputFile << road->getVerticalAlignment()->getLengths()(ii);
        if (ii < (road->getVerticalAlignment()->getLengths().size() - 1)) {
            outputFile << ",";
        }
    }
    outputFile << std::endl;

    outputFile << "GRADES:                ";
    for (int ii = 0; ii < road->getVerticalAlignment()->getGrades().size(); ii++) {
        outputFile << road->getVerticalAlignment()->getGrades()(ii);
        if (ii < (road->getVerticalAlignment()->getGrades().size() - 1)) {
            outputFile << ",";
        }
    }
    outputFile << std::endl;

    outputFile << "STOP SIGHT DISTANCES:  ";
    for (int ii = 0; ii < road->getVerticalAlignment()->getSSDs().size(); ii++) {
        outputFile << road->getVerticalAlignment()->getSSDs()(ii);
        if (ii < (road->getVerticalAlignment()->getSSDs().size() - 1)) {
            outputFile << ",";
        }
    }
    outputFile << std::endl;

    outputFile << "# QUADRATIC COEFFICIENTS" << std::endl;
    for (int ii = 0; ii < 3; ii++) {
        if (ii == 0) {
            outputFile << "X^0:                   ";
        } else if (ii == 1) {
            outputFile << "X^1:                   ";
        } else {
            outputFile << "X^2:                   ";
        }

        for (int jj = 0; jj < road->getVerticalAlignment()->getPolyCoeffs().rows(); jj++) {
            outputFile << road->getVerticalAlignment()->getPolyCoeffs()(jj,ii);
            if (ii < (road->getVerticalAlignment()->getPolyCoeffs().rows() - 1)) {
                outputFile << ",";
            }
        }
        outputFile << std::endl;
    }
    outputFile << std::endl;

    // Segments
    outputFile << "####################################################################################################" << std::endl;
    outputFile << "# SEGMENTS #########################################################################################" << std::endl;
    outputFile << "X SEGMENT COORDINATES: ";
    for (int ii = 0; ii < road->getRoadSegments()->getX().size(); ii++) {
        outputFile << road->getRoadSegments()->getX()(ii);
        if (ii < (road->getRoadSegments()->getX().size() - 1)) {
            outputFile << ",";
        }
    }
    outputFile << std::endl;

    outputFile << "Y SEGMENT COORDINATES: ";
    for (int ii = 0; ii < road->getRoadSegments()->getY().size(); ii++) {
        outputFile << road->getRoadSegments()->getY()(ii);
        if (ii < (road->getRoadSegments()->getY().size() - 1)) {
            outputFile << ",";
        }
    }
    outputFile << std::endl;

    outputFile << "Z SEGMENT COORDINATES: ";
    for (int ii = 0; ii < road->getRoadSegments()->getZ().size(); ii++) {
        outputFile << road->getRoadSegments()->getZ()(ii);
        if (ii < (road->getRoadSegments()->getZ().size() - 1)) {
            outputFile << ",";
        }
    }
    outputFile << std::endl;

    outputFile << "NATURAL ELEVATIONS:    ";
    for (int ii = 0; ii < road->getRoadSegments()->getE().size(); ii++) {
        outputFile << road->getRoadSegments()->getE()(ii);
        if (ii < (road->getRoadSegments()->getE().size() - 1)) {
            outputFile << ",";
        }
    }
    outputFile << std::endl;

    outputFile << "DISTANCE ALONG ROAD:   ";
    for (int ii = 0; ii < road->getRoadSegments()->getDists().size(); ii++) {
        outputFile << road->getRoadSegments()->getDists()(ii);
        if (ii < (road->getRoadSegments()->getDists().size() - 1)) {
            outputFile << ",";
        }
    }
    outputFile << std::endl;

    outputFile << "ROAD WIDTH AT POINTS:  ";
    for (int ii = 0; ii < road->getRoadSegments()->getWidths().size(); ii++) {
        outputFile << road->getRoadSegments()->getWidths()(ii);
        if (ii < (road->getRoadSegments()->getWidths().size() - 1)) {
            outputFile << ",";
        }
    }
    outputFile << std::endl;

    outputFile << "VELOCITIES:            ";
    for (int ii = 0; ii < road->getRoadSegments()->getVelocities().size();
            ii++) {
        outputFile << road->getRoadSegments()->getVelocities()(ii);
        if (ii < (road->getRoadSegments()->getVelocities().size() - 1)) {
            outputFile << ",";
        }
    }
    outputFile << std::endl;

    outputFile << "SEGMENT TYPE:          ";
    for (int ii = 0; ii < road->getRoadSegments()->getType().size(); ii++) {
        outputFile << road->getRoadSegments()->getType()(ii);
        if (ii < (road->getRoadSegments()->getType().size() - 1)) {
            outputFile << ",";
        }
    }
    outputFile << std::endl << std::endl;

    // Road Cells (row-major index)
    outputFile << "####################################################################################################" << std::endl;
    outputFile << "# ROAD CELLS #######################################################################################" << std::endl;
    outputFile << "# CELL POINTS DATA: " << std::endl;
    outputFile << "X:                     ";
    for (int ii = 0; ii < road->getRoadCells()->getX().size(); ii++) {
        outputFile << road->getRoadCells()->getX()(ii);
        if (ii < (road->getRoadCells()->getX().size() - 1)) {
            outputFile << ",";
        }
    }
    outputFile << std::endl;

    outputFile << "X:                     ";
    for (int ii = 0; ii < road->getRoadCells()->getX().size(); ii++) {
        outputFile << road->getRoadCells()->getX()(ii);
        if (ii < (road->getRoadCells()->getX().size() - 1)) {
            outputFile << ",";
        }
    }
    outputFile << std::endl;

    outputFile << "Y:                     ";
    for (int ii = 0; ii < road->getRoadCells()->getY().size(); ii++) {
        outputFile << road->getRoadCells()->getY()(ii);
        if (ii < (road->getRoadCells()->getY().size() - 1)) {
            outputFile << ",";
        }
    }
    outputFile << std::endl;

    outputFile << "Z:                     ";
    for (int ii = 0; ii < road->getRoadCells()->getZ().size(); ii++) {
        outputFile << road->getRoadCells()->getZ()(ii);
        if (ii < (road->getRoadCells()->getZ().size() - 1)) {
            outputFile << ",";
        }
    }
    outputFile << std::endl;

    outputFile << "WIDTHS:                ";
    for (int ii = 0; ii < road->getRoadCells()->getWidths().size(); ii++) {
        outputFile << road->getRoadCells()->getWidths()(ii);
        if (ii < (road->getRoadCells()->getWidths().size() - 1)) {
            outputFile << ",";
        }
    }
    outputFile << std::endl;

    outputFile << "X:                     ";
    for (int ii = 0; ii < road->getRoadCells()->getX().size(); ii++) {
        outputFile << road->getRoadCells()->getX()(ii);
        if (ii < (road->getRoadCells()->getX().size() - 1)) {
            outputFile << ",";
        }
    }
    outputFile << std::endl;

    outputFile << "VEGETATION:            ";
    for (int ii = 0; ii < road->getRoadCells()->getVegetation().size(); ii++) {
        outputFile << road->getRoadCells()->getVegetation()(ii);
        if (ii < (road->getRoadCells()->getVegetation().size() - 1)) {
            outputFile << ",";
        }
    }
    outputFile << std::endl;

    outputFile << "LENGTHS:               ";
    for (int ii = 0; ii < road->getRoadCells()->getLengths().size(); ii++) {
        outputFile << road->getRoadCells()->getLengths()(ii);
        if (ii < (road->getRoadCells()->getLengths().size() - 1)) {
            outputFile << ",";
        }
    }
    outputFile << std::endl;

    outputFile << "AREAS:                 ";
    for (int ii = 0; ii < road->getRoadCells()->getAreas().size(); ii++) {
        outputFile << road->getRoadCells()->getAreas()(ii);
        if (ii < (road->getRoadCells()->getAreas().size() - 1)) {
            outputFile << ",";
        }
    }
    outputFile << std::endl;

    outputFile << "TYPES:                 ";
    for (int ii = 0; ii < road->getRoadCells()->getTypes().size(); ii++) {
        outputFile << road->getRoadCells()->getTypes()(ii);
        if (ii < (road->getRoadCells()->getTypes().size() - 1)) {
            outputFile << ",";
        }
    }
    outputFile << std::endl;

    outputFile << "CELL REFERENCES:       ";
    for (int ii = 0; ii < road->getRoadCells()->getCellRefs().size(); ii++) {
        outputFile << road->getRoadCells()->getCellRefs()(ii);
        if (ii < (road->getRoadCells()->getCellRefs().size() - 1)) {
            outputFile << ",";
        }
    }
    outputFile << std::endl << std::endl;

    outputFile << "# UNIQUE CELLS OCCUPIED:  ";
    outputFile << "LENGTHS:               ";
    for (int ii = 0; ii < road->getRoadCells()->getUniqueCells().size(); ii++)
    {
        outputFile << road->getRoadCells()->getUniqueCells()(ii);
        if (ii < (road->getRoadCells()->getUniqueCells().size() - 1)) {
            outputFile << ",";
        }
    }
    outputFile << std::endl << std::endl;

    // Attributes
    outputFile << "####################################################################################################" << std::endl;
    outputFile << "# ROAD ATTRIBUTES ##################################################################################" << std::endl;
    outputFile << "# INITIAL ANIMALS AT RISK UNDER FULL FLOW (ROWS = SPECIES) " << std::endl;
    for (int ii = 0; ii < road->getAttributes()->getIAR().rows(); ii++) {
        for (int jj = 0; jj < road->getAttributes()->getIAR().cols(); jj++) {
            outputFile << road->getAttributes()->getIAR()(ii,jj);
            if (ii < (road->getAttributes()->getIAR().rows() - 1)) {
                outputFile << ",";
            }
        }
        outputFile << std::endl;
    }
    outputFile << std::endl;

    outputFile << "# END POPULATIONS (MTE)" << std::endl;
    for (int ii = 0; ii < road->getAttributes()->getEndPopMTE().size(); ii++) {
        outputFile << road->getAttributes()->getEndPopMTE()(ii);
        if (ii < (road->getAttributes()->getEndPopMTE().size() - 1)) {
            outputFile << ",";
        }
    }
    outputFile << std::endl << std::endl;

    outputFile << "# END POPULATION (MTE) STANDARD DEVIATIONS" << std::endl;
    for (int ii = 0; ii < road->getAttributes()->getEndPopMTESD().size(); ii++) {
        outputFile << road->getAttributes()->getEndPopMTESD()(ii);
        if (ii < (road->getAttributes()->getEndPopMTESD().size() - 1)) {
            outputFile << ",";
        }
    }
    outputFile << std::endl << std::endl;

    outputFile << "# OTHER ROAD ATTRIBUTES ############################################################################" << std::endl;
    outputFile << "Fixed Costs                                   : " << road->getAttributes()->getFixedCosts() << std::endl;
    outputFile << "Unit Variable Costs                           : " << road->getAttributes()->getUnitVarCosts() << std::endl;
    outputFile << "Tonnes of Ore Per Unit Traffic                : " << road->getAttributes()->getUnitVarRevenue() << std::endl;
    outputFile << "Length                                        : " << road->getAttributes()->getLength() << std::endl;
    outputFile << "Initial Unit Cost                             : " << road->getAttributes()->getInitialUnitCost() << std::endl;
    outputFile << "Total Value Mean                              : " << road->getAttributes()->getTotalValueMean() << std::endl;
    outputFile << "Total Value Standard Deviation                : " << road->getAttributes()->getTotalValueSD() << std::endl;
    outputFile << "Operating Profit (Full Flow)                  : " << road->getAttributes()->getVarProfitIC() << std::endl;
    outputFile << "Operating Profit (ROV)                        : " << road->getAttributes()->getTotalUtilisationROV() << std::endl;
    outputFile << "Operating Profit (ROV) Standard Dev           : " << road->getAttributes()->getTotalUtilisationROVSD() << std::endl;
    outputFile << std::endl;

    // Costs
    outputFile << "# COSTS ############################################################################################" << std::endl;
    outputFile << "Fixed Accident Costs                          : " << road->getCosts()->getAccidentFixed() << std::endl;
    outputFile << "Variable Accident Costs                       : " << road->getCosts()->getAccidentVariable() << std::endl;
    outputFile << "Earthwork Costs                               : " << road->getCosts()->getEarthwork() << std::endl;
    outputFile << "Fixed Length Costs                            : " << road->getCosts()->getLengthFixed() << std::endl;
    outputFile << "Variable Length Costs                         : " << road->getCosts()->getLengthVariable() << std::endl;
    outputFile << "Location Costs                                : " << road->getCosts()->getLocation() << std::endl;
    outputFile << "Penalty Costs                                 : " << road->getCosts()->getPenalty() << std::endl;
    outputFile << "Haul Load Per Unit Traffic                    : " << road->getCosts()->getUnitRevenue() << std::endl;
    outputFile << "# Fuel Usage Per Unit Traffic " << std::endl;
    for (int ii = 0; ii < this->traffic->getVehicles().size(); ii++) {
        outputFile << this->traffic->getVehicles()[ii]->getName() << " : " << road->getCosts()->getUnitFuelCost()(ii);
        if (ii < (this->traffic->getVehicles().size() - 1)) {
            outputFile << std::endl;
        }
    }
    outputFile << std::endl << std::endl;

    // Species Patch Data
    outputFile << "####################################################################################################" << std::endl;
    outputFile << "# SPECIES PATCH DATA ###############################################################################" << std::endl;

    for (int ii = 0; ii < road->getSpeciesRoadPatches().size(); ii++) {
//        // For Each Species, save Species Road Patches Results:
//        std::string srpFile = runFolder + "/" + "species_road_patches_" +
//                std::to_string(ii);

        outputFile << "####################################################################################################" << std::endl;
        outputFile << "#SPECIES " << ii << std::endl;
        outputFile << "NO PATCHES                                    : " <<
                road->getSpeciesRoadPatches()[ii]->getInitPops().size() <<
                std::endl;

        outputFile << "X CENTROIDS                                   : ";
        for (int jj = 0; jj < road->getSpeciesRoadPatches()[ii]->getInitPops()
                .size(); jj++) {
            outputFile << road->getSpeciesRoadPatches()[ii]->getHabPatches()
                    [jj]->getCX();

            if (jj < (road->getSpeciesRoadPatches()[ii]->getInitPops().size()
                    - 1)) {
                outputFile << ",";
            }
        }
        outputFile << std::endl;

        outputFile << "Y CENTROIDS                                   : ";
        for (int jj = 0; jj < road->getSpeciesRoadPatches()[ii]->getInitPops()
                .size(); jj++) {
            outputFile << road->getSpeciesRoadPatches()[ii]->getHabPatches()
                    [jj]->getCY();

            if (jj < (road->getSpeciesRoadPatches()[ii]->getInitPops().size()
                    - 1)) {
                outputFile << ",";
            }
        }
        outputFile << std::endl;

        outputFile << "AREAS                                         : ";
        for (int jj = 0; jj < road->getSpeciesRoadPatches()[ii]->getInitPops()
                .size(); jj++) {
            outputFile << road->getSpeciesRoadPatches()[ii]->getHabPatches()
                    [jj]->getArea();

            if (jj < (road->getSpeciesRoadPatches()[ii]->getInitPops().size()
                    - 1)) {
                outputFile << ",";
            }
        }
        outputFile << std::endl;

        outputFile << "CAPACITIES                                    : ";
        for (int jj = 0; jj < road->getSpeciesRoadPatches()[ii]->getInitPops()
                .size(); jj++) {
            outputFile << road->getSpeciesRoadPatches()[ii]->getHabPatches()
                    [jj]->getCapacity();

            if (jj < (road->getSpeciesRoadPatches()[ii]->getInitPops().size()
                    - 1)) {
                outputFile << ",";
            }
        }
        outputFile << std::endl;

        outputFile << "PREFERENCE TYPE                               : ";
        for (int jj = 0; jj < road->getSpeciesRoadPatches()[ii]->getInitPops()
                .size(); jj++) {
            outputFile << static_cast<int>(road->getSpeciesRoadPatches()[ii]->
                    getHabPatches()[jj]->getType()->getType());

            if (jj < (road->getSpeciesRoadPatches()[ii]->getInitPops().size()
                    - 1)) {
                outputFile << ",";
            }
        }
        outputFile << std::endl;

        outputFile << "INITIAL POPULATIONS                           : ";
        for (int jj = 0; jj < road->getSpeciesRoadPatches()[ii]->getInitPops()
                .size(); jj++) {
            outputFile << road->getSpeciesRoadPatches()[ii]->getInitPops()(jj);

            if (jj < (road->getSpeciesRoadPatches()[ii]->getInitPops().size()
                    - 1)) {
                outputFile << ",";
            }
        }
        outputFile << std::endl;
        // Patch Cells
        // Centroid X : Centroid Y : Type : Cell_1, Cell_2, etc.

        outputFile << "PATCH CELLS" << std::endl;
        for (int jj = 0; jj < road->getSpeciesRoadPatches()[ii]->getInitPops()
                .size(); jj++) {
            outputFile << "PATCH_" << jj << " : ";

            for (int kk = 0; kk < road->getSpeciesRoadPatches()[ii]->
                    getHabPatches()[jj]->getCells().size(); kk++) {
                outputFile << road->getSpeciesRoadPatches()[ii]->
                        getHabPatches()[jj]->getCells()(kk);

                if (kk < (road->getSpeciesRoadPatches()[ii]->
                        getHabPatches()[jj]->getCells().size() - 1)) {
                    outputFile << ",";
                }
            }
            outputFile << std::endl;
        }

        outputFile << std::endl;
    }

    outputFile.close();

    // Simulation Results
    std::string simulationResults = runFolder + "/" + "simulation_results_MTE";

    // Simulate a single forward path
    std::vector<Eigen::MatrixXd> visualisePopulations(this->species.size());

    for (int ii = 0; ii < this->species.size(); ii++) {
        visualisePopulations[ii].resize(road->getSpeciesRoadPatches()[ii]->
                getInitPops().size(),this->getEconomic()->getYears() + 1);
    }

    road->getSimulator()->simulateMTE(visualisePopulations);

    // MTE:
    outputFile.open(simulationResults,std::ios::out);

    outputFile << "####################################################################################################" << std::endl;
    outputFile << "###################################### SAMPLE SIMULATION MTE #######################################" << std::endl;
    outputFile << "####################################################################################################" << std::endl;
    outputFile << "# Each path row (for each species) shows the population at succeeding time steps. See the original" << std::endl;
    outputFile << "# input file for details of the number of periods and their spacing." << std::endl;
    outputFile << std::endl;

    // Single Path Simulation (For Visualisation)
    for (int ii = 0; ii < this->species.size(); ii++) {
        outputFile << "# SPECIES " << ii << std::endl;
        for (int jj = 0; jj < road->getSpeciesRoadPatches()[ii]->getInitPops()
                .size(); jj++) {
            outputFile << "PATCH_" << jj << " : ";

            for (int kk = 0; kk < this->economic->getYears(); kk++) {
                outputFile << visualisePopulations[ii](jj,kk);
                if (kk < (this->economic->getYears() - 1)) {
                    outputFile << ",";
                }
            }
            outputFile << std::endl;
        }
        outputFile << std::endl << std::endl;
    }
    outputFile.close();

    if (this->type >= Optimiser::CONTROLLED) {
        // First, perform the simulation calculation
        Eigen::VectorXd visualiseUnitProfits((int)this->getEconomic()->
                getYears());
        Eigen::VectorXi visualiseFlows((int)this->getEconomic()->getYears());

        road->getSimulator()->simulateROVCR(visualisePopulations,
                visualiseFlows,visualiseUnitProfits);

        // Now save the results to the output file
        std::string simulationResults = runFolder + "/" +
                "simulation_results_ROV";
        // ROV:
        outputFile.open(simulationResults,std::ios::out);

        outputFile << "####################################################################################################" << std::endl;
        outputFile << "###################################### SAMPLE SIMULATION ROV #######################################" << std::endl;
        outputFile << "####################################################################################################" << std::endl;
        outputFile << "# Each path row (for each species) shows the population at succeeding time steps. See the original" << std::endl;
        outputFile << "# input file for details of the number of periods and their spacing." << std::endl;
        outputFile << std::endl;

        outputFile << "####################################################################################################" << std::endl;
        outputFile << "# POPULATION DATA" << std::endl;

        for (int ii = 0; ii < this->species.size(); ii++) {
            outputFile << "# SPECIES " << ii << std::endl;
            for (int jj = 0; jj < road->getSpeciesRoadPatches()[ii]->getInitPops()
                    .size(); jj++) {
                outputFile << "PATCH_" << jj << " : ";

                for (int kk = 0; kk < this->economic->getYears(); kk++) {
                    outputFile << visualisePopulations[ii](jj,kk);
                    if (kk < (this->economic->getYears() - 1)) {
                        outputFile << ",";
                    }
                }
                outputFile << std::endl;
            }
            outputFile << std::endl << std::endl;
        }

        outputFile << "####################################################################################################" << std::endl;
        outputFile << "# FLOWS AND UNIT PROFITS" << std::endl << std::endl;

        outputFile << "# FLOWS                  : ";
        for (int ii = 0; ii < visualiseFlows.size(); ii++) {
            outputFile << visualiseFlows(ii);
            if (ii < visualiseFlows.size()) {
                outputFile << ",";
            }
        }

        outputFile << std::endl;

        outputFile << "# UNIT PROFITS           : ";
        for (int ii = 0; ii < visualiseFlows.size(); ii++) {
            outputFile << visualiseUnitProfits(ii);
            if (ii < visualiseUnitProfits.size()) {
                outputFile << ",";
            }
        }

        outputFile.close();

        std::string rovData = runFolder + "/" +
                "policy_map_data";
        // ROV:
        outputFile.open(rovData,std::ios::out);

        // Now save the ROV model and raw data
        outputFile << "####################################################################################################" << std::endl;
        outputFile << "######################################## ROV POLICY MAP DATA #######################################" << std::endl;
        outputFile << "####################################################################################################" << std::endl;
        outputFile << std::endl;

        // Raw Simulation Data:
        outputFile << "####################################################################################################" << std::endl;
        outputFile << "# RAW SIMULATION DATA USED TO PRODUCE REGRESSIONS AND POLICY MAP ###################################" << std::endl;
        outputFile << "####################################################################################################" << std::endl;
        outputFile << std::endl;
        outputFile << "####################################################################################################" << std::endl;
        outputFile << "# STATES RAW DATA" << std::endl;
        outputFile << "# Data arranged by year->state->path" << std::endl;
        for (int ii = 0; ii < visualiseFlows.size(); ii++) {
            outputFile << "# YEAR " << ii << std::endl;
            // Species adjusted populations
            for (int jj = 0; jj < this->species.size(); jj++) {
                outputFile << "SPECIES " << std::setw(3) << jj << " ADJUSTED POPULATIONS            : ";

                for (int kk = 0; kk < this->otherInputs->getNoPaths(); kk++) {
                    outputFile << road->getPolicyMap()->getPolicyMapYear()[ii]
                            ->getStateLevels()(kk,jj);
                    if (ii < visualiseFlows.size()) {
                        outputFile << ",";
                    }
                }
                outputFile << std::endl;
            }

            // Unit profits
            outputFile << "UNIT PROFITS                                : ";
            for (int jj = 0; jj < this->otherInputs->getNoPaths(); jj++) {
                outputFile << road->getPolicyMap()->getPolicyMapYear()[ii]->
                        getStateLevels()(jj,this->species.size());
                if (ii < visualiseFlows.size()) {
                    outputFile << ",";
                }
            }
            outputFile << std::endl;

            // Corresponding conditional expectations
            outputFile << "CORRESPONDING CONDITIONAL EXPECTATIONS      : ";
            for (int jj = 0; jj < this->otherInputs->getNoPaths(); jj++) {
                outputFile << road->getPolicyMap()->getPolicyMapYear()[ii]->
                        getProfits()(jj);
                if (ii < visualiseFlows.size()) {
                    outputFile << ",";
                }
            }
            outputFile << std::endl;

            // Corresponding controls
            outputFile << "CORRESPONDING CONTROLS                      : ";
            for (int jj = 0; jj < this->otherInputs->getNoPaths(); jj++) {
                outputFile << road->getPolicyMap()->getPolicyMapYear()[ii]->
                        getOptConts()(jj);
                if (ii < visualiseFlows.size()) {
                    outputFile << ",";
                }
            }
            outputFile << std::endl;
        }

        outputFile.close();

        // Regressions (Not implemented yet)
        std::string rovRegressions = runFolder + "/" +
                "regressions";
        // ROV:
        outputFile.open(rovRegressions,std::ios::out);

        outputFile.close();
    }
}

void Optimiser::computeBestRoadResults() {

    this->bestRoads[this->scenario->getCurrentScenario()][this->
            scenario->getRun()]->designRoad();

    // First, compute the road using ROV
    Optimiser::Type tempType = this->type;

    // Temporarily alter the type to MTE to compute under MTE (if using original tests)
    this->type = Optimiser::CONTROLLED;

    this->bestRoads[this->scenario->getCurrentScenario()][this->
            scenario->getRun()]->evaluateRoad(true,true);

    // Now save all of the predictors and other details

    // PREDICTORS /////////////////////////////////////////////////////////////

    // INITIAL PERIOD UNIT PROFIT
    // For now, we use the initial unit profit. This is saved as a road
    // attribute.

    // Fixed cost per unit traffic
    double unitCost = this->bestRoads[this->scenario->getCurrentScenario()]
            [this->scenario->getRun()]->getAttributes()->getUnitVarCosts();
    // Fuel consumption per vehicle class per unit traffic (L)
    Eigen::VectorXd fuelCosts = this->bestRoads[this->scenario->
            getCurrentScenario()][this->scenario->getRun()]->getCosts()->
            getUnitFuelCost();
    Eigen::VectorXd currentFuelPrice(fuelCosts.size());

    for (int ii = 0; ii < fuelCosts.size(); ii++) {
        currentFuelPrice(ii) = (this->bestRoads[this->scenario->
                getCurrentScenario()][this->scenario->getRun()]->getOptimiser()
                ->getTraffic()->getVehicles())[ii]->getFuel()->getCurrent();
    }

    unitCost += fuelCosts.transpose()*currentFuelPrice;
    double rovalue = this->bestRoads[this->scenario->getCurrentScenario()]
            [this->scenario->getRun()]->getAttributes()->
            getTotalUtilisationROV();
    double rovalueSD = this->bestRoads[this->scenario->getCurrentScenario()]
            [this->scenario->getRun()]->getAttributes()->
            getTotalUtilisationROVSD();

//    // As the revenue per unit traffic is the same for each road, we leave it
//    // out for now.
//    // Load per unit traffic
//    // double unitRevenue = road->getCosts()->getUnitRevenue();

//    this->bestRoads[this->scenario->getCurrentScenario()][this->
//            scenario->getRun()]->getAttributes()->setInitialUnitCost(unitCost);

//    // Temporarily alter the type to CONTROLLED to compute under ROV
//    this->type = Optimiser::CONTROLLED;
//    this->bestRoads[this->scenario->getCurrentScenario()][this->
//            scenario->getRun()]->evaluateRoad(true,true);

    // We also need to compute the model using MTE to know what the full
    // traffic situation looks like
    this->type = Optimiser::MTE;
    this->bestRoads[this->scenario->getCurrentScenario()][this->
            scenario->getRun()]->evaluateRoad(true,true);

    // Assign the values that were overwritten
    this->bestRoads[this->scenario->getCurrentScenario()][this->
            scenario->getRun()]->getAttributes()->setInitialUnitCost(unitCost);
    this->bestRoads[this->scenario->getCurrentScenario()]
                [this->scenario->getRun()]->getAttributes()->
                setTotalUtilisationROV(rovalue);
    this->bestRoads[this->scenario->getCurrentScenario()]
                [this->scenario->getRun()]->getAttributes()->
                setTotalUtilisationROVSD(rovalueSD);

    // Finally, reset the type
    this->type = tempType;
}
