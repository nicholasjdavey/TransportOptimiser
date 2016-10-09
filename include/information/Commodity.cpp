#include "../transportbase.h"

Commodity::Commodity() : Uncertainty() {
}

Commodity::Commodity(std::string nm, double mp, double sd, double rev,
        const std::vector<CommodityCovariancePtr>& covs, bool active) :
		Uncertainty(nm, mp, sd, rev, active) {
    this->covariances = covs;
}

Commodity::~Commodity() {}
