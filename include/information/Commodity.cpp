#include "../transportbase.h"

Commodity::Commodity(OptimiserPtr optimiser) : Uncertainty(optimiser) {
}

Commodity::Commodity(OptimiserPtr optimiser, std::string nm, double mp, double
        sd, double rev, bool active, double oc, double ocsd) :
        Uncertainty(optimiser, nm, mp, sd, rev, active) {

    this->oreContent = oc;
    this->oreContentSD = ocsd;
}

Commodity::~Commodity() {}
