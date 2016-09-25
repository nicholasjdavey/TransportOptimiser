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

void Species::generateHabitatMap(OptimiserPtr optimiser) {
    Eigen::MatrixXi* veg = optimiser->getRegion()->getVegetation();

    this->habitatMap = Eigen::MatrixXi::Zero(veg->rows(),veg->cols());
    std::vector<HabitatTypePtr>* habTypes = this->getHabitatTypes();

    for (int ii = 0; ii < habTypes->size(); ii++) {
        Eigen::VectorXi* vegNos = (*habTypes)[ii]->getVegetations();
        for (int jj; jj < vegNos->size(); jj++) {
            this->habitatMap += ((veg->array() == (*vegNos)(jj))*
                    (int)((*habTypes)[ii]->getType())).cast<int>().matrix();
        }
    }
}

void Species::generateHabitatPatches(RoadPtr road) {
    RegionPtr region = road->getOptimiser()->getRegion();
    Eigen::MatrixXd* X = region->getX();
    Eigen::MatrixXd* Y = region->getY();

    Eigen::VectorXd xspacing = (X->block(1,0,X->rows()-1,1)
            - X->block(0,0,X->rows()-1,1)).transpose();
    Eigen::VectorXd yspacing = Y->block(0,1,1,Y->cols()-1)
            - Y->block(0,0,1,Y->cols()-1);

    // Grid will be evenly spaced upon call
    if ((xspacing.segment(1,xspacing.size()-1)
            - xspacing.segment(0,xspacing.size()-1)).sum() > 1e-4 ||
            (yspacing.segment(1,yspacing.size()-1)
            - yspacing.segment(0,yspacing.size()-1)).sum() > 1e-4) {
        throw std::invalid_argument("Grid must be evenly spaced in both X and Y");
    }

    Eigen::MatrixXi modHab = (*this->getHabitatMap());
    Eigen::MatrixXi tempHabVec = Eigen::MatrixXi::Constant(1,
            road->getRoadCells()->getUniqueCells()->size(),
            (int)(HabitatType::ROAD));
    igl::slice_into(tempHabVec,*road->getRoadCells()->getUniqueCells(),modHab);

    // We create bins for each habitat type into which we place the patches

}
