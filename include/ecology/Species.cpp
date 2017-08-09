#include "../transportbase.h"

Species::Species() {
    // Default values
    this->setName("nm""");
    this->sex = true;
    this->lambdaMean = 0;
    this->lambdaSD = 0;
    this->rangingCoeffMean = 0;
    this->rangingCoeffSD = 0;
    this->localVariability = 0;
    this->lengthMean = 0;
    this->lengthSD = 0;
    this->speedMean = 0;
    this->speedSD = 0;
    this->costPerAnimal = 0;
    this->setActive(true);
    this->initialPop = 0;
    this->threshold = 0;
}

Species::Species(std::string nm, bool sex, double lm, double lsd, double rcm,
        double rcsd, UncertaintyPtr gr, double lv, double lenm, double lensd,
        double spm,double spsd, double cpa, bool active, double initPop, double
        t,std::vector<HabitatTypePtr> &habitat) {

    // Initialise object values
    this->setName(nm);
    this->sex = sex;
    this->lambdaMean = lm;
    this->lambdaSD = lsd;
    this->rangingCoeffMean = rcm;
    this->rangingCoeffSD = rcsd;
    this->growthRate = gr;
    this->localVariability = lv;
    this->lengthMean = lenm;
    this->lengthSD = lensd;
    this->speedMean = spm;
    this->speedSD = spsd;
    this->costPerAnimal = cpa;
    this->setActive(active);
    this->habitat = habitat;
    this->initialPop = initPop;
    this->threshold = t;
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
    this->populationMap = Eigen::MatrixXd::Constant(this->habitatMap.rows(),
            this->habitatMap.cols(),0);

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
    //double* popMapPtr = this->populationMap.data();

    for (int ii = 0; ii < this->initialPop; ii++) {
        this->populationMap(events[ii])++;
        //popMapPtr[events[ii]] = popMapPtr[events[ii]] + 1.0;
    }
}
