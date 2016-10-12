#include "../transportbase.h"

Commodity::Commodity(OptimiserPtr optimiser) : Uncertainty(optimiser) {
}

Commodity::Commodity(OptimiserPtr optimiser, std::string nm, double mp, double
        sd, double rev, const std::vector<CommodityCovariancePtr>& covs, bool
        active) : Uncertainty(optimiser, nm, mp, sd, rev, active) {
    this->covariances = covs;
}

Commodity::~Commodity() {}
