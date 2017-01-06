#include "../transportbase.h"

Species::Species(std::string nm, bool sex, double lm, double lsd, double rcm,
        double rcsd, double grm, double grsd, double lenm, double lensd,
        double spm, double spsd, double cpa, bool active, double initPop,
        std::vector<HabitatTypePtr> &habitat) {

	// Initialise object values
    this->setName(nm);
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
    this->initialPop = initPop;
}

Species::Species(std::string nm, bool sex, double lm, double lsd, double rcm,
        double rcsd, double grm, double grsd, double lenm, double lensd,
        double spm, double spsd, double cpa, bool active, double initPop,
        std::vector<HabitatTypePtr>& habitat, double current) {

    // Initialise object values
    this->setName(nm);
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
    this->initialPop = initPop;
}

Species::~Species() {

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
        for (int jj = 0; jj < vegNos.size(); jj++) {
            this->habitatMap += ((veg.array() == vegNos(jj)).cast<int>()*
                    (int)(habTypes[ii]->getType())).cast<int>().matrix();
        }
    }
}

void Species::initialisePopulationMap(OptimiserPtr optimiser) {
    this->populationMap.resize(this->habitatMap.rows(),
            this->habitatMap.cols());

    // For now, we can only deal with regular rectangular grids
    double xSpan = optimiser->getRegion()->getX()(1,0) -
            optimiser->getRegion()->getX()(0,0);
    double ySpan = optimiser->getRegion()->getY()(0,1) -
            optimiser->getRegion()->getY()(0,0);

    double cellArea = xSpan*ySpan;

    Eigen::VectorXd maxCaps(this->habitat.size());
    for (int ii = 0; ii < this->habitat.size(); ii++) {
        maxCaps(ii) = this->habitat[ii]->getMaxPop();
    }

    int maxCellPop = ceil(cellArea*maxCaps.maxCoeff());
    int maxElements = maxCellPop*this->habitatMap.rows()*
            this->habitatMap.cols();

    int iterator = 0;
    std::vector<unsigned long> events(maxElements);

    for (int ii = 0; ii < this->habitatMap.cols(); ii++) {
        for (int jj = 0; jj < this->habitatMap.rows(); jj++) {
            int cellMax = ceil(cellArea*this->habitat[this->habitatMap(jj,ii)]
                    ->getMaxPop());
            if (cellMax > 0) {
                for (int kk = 0; kk < cellMax; kk++) {
                    events[iterator] = jj + ii*this->habitatMap.rows();
                    iterator++;
                }
            }
        }
    }

    // Remove excess elements
    events.resize(iterator);

    // Randomly select locations to place the animals
    std::random_shuffle(events.begin(),events.end());

    // Use the first N entries to assign animals to the population map where N
    // is the number animals in the initial population
    double* popMapPtr = this->populationMap.data();

    for (int ii = 0; ii < this->initialPop; ii++) {
        popMapPtr[events[ii]] = popMapPtr[events[ii]] + 1.0;
    }
}
