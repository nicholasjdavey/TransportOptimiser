#include "../transportbase.h"

Species::Species(std::string* nm, bool sex, double lm, double lsd, double rcm,
        double rcsd, double grm, double grsd, double lenm, double lensd,
        double spm, double spsd, double cpa, bool active,
        std::vector<HabitatTypePtr> &habitat) {

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
    this->setActive(active);
    this->habitat = habitat;
}

Species::Species(std::string* nm, bool sex, double lm, double lsd, double rcm,
        double rcsd, double grm, double grsd, double lenm, double lensd,
        double spm, double spsd, double cpa, bool active,
        std::vector<HabitatTypePtr>& habitat, double current) {

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
    this->setActive(active);
    this->habitat = habitat;
}

SpeciesPtr Species::me() {
    return shared_from_this();
}

void Species::generateHabitatMap(OptimiserPtr optimiser) {
    const Eigen::MatrixXi& veg = optimiser->getRegion()->getVegetation();

    this->habitatMap = Eigen::MatrixXi::Zero(veg.rows(),veg.cols());
    const std::vector<HabitatTypePtr>& habTypes = this->getHabitatTypes();

    for (int ii = 0; ii < habTypes.size(); ii++) {
        const Eigen::VectorXi& vegNos = habTypes[ii]->getVegetations();
        for (int jj; jj < vegNos.size(); jj++) {
            this->habitatMap += ((veg.array() == vegNos(jj))*
                    (int)(habTypes[ii]->getType())).cast<int>().matrix();
        }
    }
}
