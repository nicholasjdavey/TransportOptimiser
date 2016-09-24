#include "../transportbase.h"

Optimiser::Optimiser(std::vector<TrafficProgramPtr>* programs, OtherInputsPtr oInputs,
        DesignParametersPtr desParams, EarthworkCostsPtr earthworks,
        UnitCostsPtr unitCosts, VariableParametersPtr varParams,
        std::vector<SpeciesPtr>* species, EconomicPtr economic, TrafficPtr traffic,
        RegionPtr region, double mr, unsigned long cf, unsigned long gens,
        unsigned long popSize, double stopTol, double confidence,
        unsigned long habGridRes, std::string solScheme, unsigned long noRuns,
        Optimiser::Type type) {

//	std::vector<RoadPtr>* crp(new std::vector<RoadPtr>());
    this->type = type;
	Eigen::MatrixXd currPop(popSize,3*(desParams->getIntersectionPoints()+1));
	this->currentRoadPopulation = currPop;

	unsigned long noTests = ((varParams->getPopulationLevels())->size())*
			((varParams->getHabPref())->size())*((varParams->getLambda())->
			size())*((varParams->getBeta())->size());

	std::vector< std::vector<RoadPtr> > br(noTests);

	for(unsigned int ii=0; ii<noTests;ii++) {
		std::vector<RoadPtr> brr(noRuns);
		br.push_back(brr);
	}
	this->bestRoads = br;

//	std::vector<ProgramPtr>* programs(new std::vector<ProgramPtr>());
	this->programs = *programs;

	this->otherInputs = oInputs;
	this->designParams = desParams;
	this->earthworks = earthworks;
	this->economic = economic;
	this->traffic = traffic;
	this->region = region;
	
	this->earthworks = earthworks;
	this->unitCosts = unitCosts;
	this->species = *species;
	this->mutationRate = mr;
	this->crossoverFrac = cf;
	this->variableParams = varParams;
	this->generations = gens;
	this->noRuns = noRuns;
	this->populationSizeGA = popSize;
	this->stoppingTol = stopTol;
	this->confidence = confidence;
	this->habGridRes = habGridRes;
	this->solutionScheme = solScheme;
}

Optimiser::~Optimiser() {
}

void Optimiser::optimiseRoad() {}
