#include "../transportbase.h"

Uncertainty::Uncertainty(OptimiserPtr optimiser) {

    this->optimiser = optimiser;
    this->name = "";
    this->current = 0;
    this->meanP = 0;
    this->standardDev = 0;
    this->reversion = 0;
    this->poissonJump = 0;
    this->jumpProb = 0;
    this->active = false;
}

Uncertainty::Uncertainty(OptimiserPtr optimiser, std::string nm, double curr,
        double mp, double sd, double rev, double pj, double jp, bool active) {

    this->optimiser = optimiser;
    this->name = nm;
    this->current = curr;
    this->meanP = mp;
    this->standardDev = sd;
    this->reversion = rev;
    this->poissonJump = pj;
    this->jumpProb = jp;
    this->active = active;
}

Uncertainty::~Uncertainty() {
}

UncertaintyPtr Uncertainty::me() {
    return shared_from_this();
}

void Uncertainty::computeExpPV() {
    // Ornstein-Uhlenbeck process dx = N(m - x)dt + sdW
    OptimiserPtr optimiser = this->optimiser.lock();
    ThreadManagerPtr threader = optimiser->getThreadManager();
    unsigned int paths = optimiser->getOtherInputs()->getNoPaths();
    double total = 0;

    // Place to store simulation results
    Eigen::VectorXd finalResults(paths);

    //std::chrono::steady_clock::time_point beginTime = std::chrono::steady_clock::now();

    if (optimiser->getGPU()) {
        SimulateGPU::expPV(this->me());
    } else if (threader != nullptr) {
        // If multithreading is enabled
        std::vector<std::future<double>> results(paths);

        for (unsigned long ii = 0; ii < paths; ii++)  {
            // Push onto the pool with a lambda expression
//            results[ii] = p.push([this](int) {return
//                    this->singlePathValue();});
            results[ii] = threader->push([this](int){return
                    this->singlePathValue();});
        }

        for (unsigned long ii = 0; ii < paths; ii++) {
            finalResults(ii) = results[ii].get();
            total += finalResults(ii);
        }

    } else {
        for (unsigned long ii = 0; ii < paths; ii++) {
            finalResults(ii) = this->singlePathValue();
            total += finalResults(ii);
        }

        this->expPV = total/paths;
        this->expPVSD = sqrt(((finalResults.array() - expPV).square()).sum());
    }


    //std::chrono::steady_clock::time_point endTime = std::chrono::steady_clock::now();
    //std::cout << "Time difference = " << std::chrono::duration_cast<std::chrono::microseconds>(endTime - beginTime).count() <<std::endl;
}

double Uncertainty::singlePathValue() {
    // Compute a single path
    double curr = this->current;
    double value = 0;
    OptimiserPtr optimiser = this->optimiser.lock();
    EconomicPtr economic = this->getOptimiser()->getEconomic();
    double gr = optimiser->getTraffic()->getGR()*economic->getTimeStep();
    // Instantiate the default C++11 Mersenne twiser pseudo random number
    // generator
//    unsigned seed1 = std::chrono::system_clock::now().time_since_epoch().
//            count();
//    std::mt19937_64 generator(seed1);
    // Brownian motion uncertainty
    std::normal_distribution<double> brownian(0,this->standardDev);
    // Jump size uncertainty
    std::normal_distribution<double> jumpSize(-pow(this->poissonJump,2)/2,
            pow(this->poissonJump,2));
    // One Binomial distribution (one draw, probaility lambda) for a jump
    // occurring at any time step
    std::binomial_distribution<int> jump(1,this->jumpProb);

    for (int ii = 0; ii < economic->getYears(); ii++) {
        curr += this->reversion*(this->meanP - curr)*economic->getTimeStep()
                + curr*brownian(generator) + (exp(jumpSize(generator))-1)*
                curr*jump(generator);
        value += pow(1+gr,ii)*curr/pow((1+economic->getRRR()),ii);
    }

    return value;
}
