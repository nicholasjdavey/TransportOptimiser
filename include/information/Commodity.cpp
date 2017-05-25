#include "../transportbase.h"

Commodity::Commodity(OptimiserPtr optimiser) : Uncertainty(optimiser) {
}

Commodity::Commodity(OptimiserPtr optimiser, std::string nm, double curr,
        double mp, double sd, double rev, double pj, double jp, bool active,
        double oc, double ocsd) :
        Uncertainty(optimiser, nm, curr, mp, sd, rev, pj, jp, active) {

    this->oreContent = oc;
    this->oreContentSD = ocsd;
}

Commodity::~Commodity() {}
