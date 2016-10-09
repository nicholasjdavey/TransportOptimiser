#include "../transportbase.h"

Region::Region(const Eigen::MatrixXd& X, const Eigen::MatrixXd& Y, const
        Eigen::MatrixXd& Z, const Eigen::MatrixXd& acCost, const
        Eigen::MatrixXd& ssc, const Eigen::MatrixXd& cc, const
        Eigen::MatrixXi& veg, std::string inputFile) {

    this->X = X;
    this->Y = Y;
    this->Z = Z;
    this->veg = veg;
    this->acCost = acCost;
    this->soilStabCost = ssc;
    this->clearCosts = cc;
	this->inputFile = inputFile;
}

Region::Region(std::string input) {
}

Region::Region(std::string rawData, bool rd) {
}
