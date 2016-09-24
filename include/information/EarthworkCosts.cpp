#include "../transportbase.h"

EarthworkCosts::EarthworkCosts(Eigen::VectorXd* cd, Eigen::VectorXd* cc,
		double fc) {

	this->cDepths = *cd;
	this->cCosts = *cc;
	this->fCost = fc;
}
