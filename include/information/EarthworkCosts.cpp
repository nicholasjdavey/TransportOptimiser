#include "../transportbase.h"

EarthworkCosts::EarthworkCosts(const Eigen::VectorXd& cd, const
        Eigen::VectorXd& cc, double fc) {

    this->cDepths = cd;
    this->cCosts = cc;
    this->fCost = fc;
}
