#include "../transportbase.h"

Region::Region(Eigen::MatrixXd* X, Eigen::MatrixXd* Y, Eigen::MatrixXd* Z,
		Eigen::MatrixXd* acCost, Eigen::MatrixXd* ssc, Eigen::MatrixXd* cc,
        Eigen::MatrixXi* veg, std::string inputFile) {

	this->X = *X;
	this->Y = *Y;
	this->Z = *Z;
    this->veg = *veg;
	this->acCost = *acCost;
	this->soilStabCost = *ssc;
	this->clearCosts = *cc;
	this->inputFile = inputFile;
}

Region::Region(std::string input) {
}

Region::Region(std::string rawData, bool rd) {
}
