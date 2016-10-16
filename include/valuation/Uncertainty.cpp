#include "../transportbase.h"

Uncertainty::Uncertainty(OptimiserPtr optimiser) {

    this->optimiser = optimiser;
	this->name = "";
	this->current = 0;
	this->meanP = 0;
	this->standardDev = 0;
	this->reversion = 0;
	this->active = false;
}

Uncertainty::Uncertainty(OptimiserPtr optimiser, std::string nm, double mp,
        double sd, double rev, bool active) {

    this->optimiser = optimiser;
	this->name = nm;
	this->current = 0;
	this->meanP = mp;
	this->standardDev = sd;
	this->reversion = rev;
	this->active = active;
}

Uncertainty::~Uncertainty() {
}

void Uncertainty::computeExpPV() {
    // Ornstein-Uhlenbeck process dx = N(m - x)dt + sdW
    OptimiserPtr optimiser = this->optimiser.lock();
    ThreadManagerPtr threader = optimiser->getThreadManager();
    unsigned int paths = optimiser->getOtherInputs()->getNoPaths();
    double total = 0;

    // Place to store simulation results
    std::vector< std::future<double> > results(paths);

    if (threader != nullptr) {
        for (unsigned long ii = 0; ii < paths; ii++)  {
            // Push onto the pool with a lambda expression
            results[ii] = threader->push([this](int id){return
                    this->singlePathValue();});
        }

        for (unsigned long ii = 0; ii < paths; ii++) {
            total += results[ii].get();
        }

    } else {
        for (unsigned long ii = 0; ii < paths; ii++) {
            total += this->singlePathValue();
        }
    }

    this->expPV = total/paths;
}

double Uncertainty::singlePathValue() {
    // Compute a single path
    double curr = this->current;
    double value = 0;
    OptimiserPtr optimiser = this->optimiser.lock();
    double gr = optimiser->getTraffic()->getGR();
    // Instantiate the default C++11 Mersenne twiser pseudo random number
    // generator
    std::mt19937_64 generator;
    // Brownian motion uncertainty
    std::normal_distribution<double> brownian(0,this->standardDev);
    // Jump size uncertainty
    std::normal_distribution<double> jumpSize(-pow(this->poissonJump,2)/2,
            pow(this->poissonJump,2));
    // One Binomial distribution (one draw, probaility lambda) for a jump
    // occurring at any time step
    std::binomial_distribution<int> jump(1,this->jumpProb);

    EconomicPtr economic = this->getOptimiser()->getEconomic();

    for (int ii = 0; ii < economic->getYears(); ii++) {
        curr += this->reversion*(this->meanP - curr)*economic->getTimeStep()
                + curr*brownian(generator) + (exp(jumpSize(generator))-1)*
                curr*jump(generator);
        value += pow(1+gr,ii)*curr/pow((1+economic->getRRR()),ii);
    }
}
