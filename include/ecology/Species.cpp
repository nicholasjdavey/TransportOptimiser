#include "../transportbase.h"

Species::Species(std::string* nm, bool sex, double lm, double lsd, double rcm,
        double rcsd, double grm, double grsd, double lenm, double lensd,
        double spm, double spsd, double cpa, bool active,
        std::vector<HabitatTypePtr>* habitat) :
		Uncertainty() {

	// Initialise object values
	this->setName(*nm);
    this->sex = sex;
	this->lambdaMean = lm;
	this->lambdaSD = lsd;
	this->rangingCoeffMean = rcm;
	this->rangingCoeffSD = rcsd;
	this->growthRateMean = grm;
	this->growthRateSD = grsd;
	this->lengthMean = lenm;
	this->lengthSD = lensd;
	this->speedMean = spm;
	this->speedSD = spsd;
	this->costPerAnimal = cpa;
	this->setStatus(active);
	this->habitat = *habitat;
}

Species::Species(std::string* nm, bool sex, double lm, double lsd, double rcm,
        double rcsd, double grm, double grsd, double lenm, double lensd,
        double spm, double spsd, double cpa, bool active,
        std::vector<HabitatTypePtr>* habitat, double current,
        double meanP, double mean, double stdDev, double rev) :
		Uncertainty(*nm, meanP, stdDev, rev, active) {

	// Initialise object values
	this->setName(*nm);
    this->sex = sex;
	this->lambdaMean = lm;
	this->lambdaSD = lsd;
	this->rangingCoeffMean = rcm;
	this->rangingCoeffSD = rcsd;
	this->growthRateMean = grm;
	this->growthRateSD = grsd;
	this->lengthMean = lenm;
	this->lengthSD = lensd;
	this->speedMean = spm;
	this->speedSD = spsd;
	this->costPerAnimal = cpa;
	this->setStatus(active);
	this->habitat = *habitat;
}

Species::generateHabitatPatches(RoadPtr road) {

}
