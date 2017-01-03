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

Region::~Region() {

}

void Region::placeNetwork(const Eigen::VectorXd& x, const Eigen::VectorXd& y,
        Eigen::VectorXd& z) {

    unsigned int nxd = this->X.rows();
    unsigned int nyd = this->Y.cols();
    int ni = x.size();

    Eigen::MatrixXd xvals = this->X.block(0,0,nxd,1).transpose();
    Eigen::MatrixXd yvals = this->Y.block(0,0,1,nyd);
    Eigen::VectorXd tempx = x;
    double* xi;
    xi = tempx.data();
    Eigen::VectorXd tempy = y;
    double* yi;
    yi = tempy.data();
    double* pwl = z.data();
    pwl = pwl_interp_2d(nxd,nyd,xvals.data(),yvals.data(),this->Z.data(),ni,xi,yi);
    z = Eigen::Map<Eigen::MatrixXd>(pwl,z.rows(),z.cols());
    /*
     * old method with creating arrays
    double* xd;
    xd = new double[nxd];
    double* yd;
    yd = new double[nyd];
    double* zd;
    zd = new double[nxd*nyd];
    double* xi;
    xi = new double[ni];
    double* yi;
    yi = new double[ni];
    double* pwl;
    pwl = new double[ni];

    Eigen::MatrixXd xvals = this->X.block(0,0,nxd,1).transpose();
    Eigen::MatrixXd yvals = this->X.block(0,0,1,nyd);

    // Convert the Eigen matrices to standard C++ arrays for use
    Eigen::Map<Eigen::MatrixXd>(xd,xvals.rows(),xvals.cols()) = xvals;
    Eigen::Map<Eigen::MatrixXd>(yd,yvals.rows(),yvals.cols()) = yvals;
    Eigen::Map<Eigen::MatrixXd>(zd,Z.rows(),Z.cols()) = Z;
    Eigen::Map<Eigen::MatrixXd>(xi,x.rows(),x.cols()) = x;
    Eigen::Map<Eigen::MatrixXd>(yi,y.rows(),y.cols()) = y;

    pwl = pwl_interp_2d(nxd, nyd, xd, yd, zd, ni, xi, yi);
    z = Eigen::Map<Eigen::MatrixXd>(pwl,1,ni);

    delete[] xd;
    delete[] yd;
    delete[] zd;
    delete[] xi;
    delete[] yi;
    delete[] pwl;
    */
}

void Region::placeNetwork(double& x, double& y, double& z) {
    unsigned int nxd = this->X.rows();
    unsigned int nyd = this->Y.cols();

    double* xd;
    double* yd;
    double* zd;
    double* pwl = &z;

    Eigen::MatrixXd xvals = this->X.block(0,0,nxd,1).transpose();
    Eigen::MatrixXd yvals = this->X.block(0,0,1,nyd);

    // Convert the Eigen matrices to standard C++ arrays for use
    Eigen::Map<Eigen::MatrixXd>(xd,xvals.rows(),xvals.cols()) = xvals;
    Eigen::Map<Eigen::MatrixXd>(yd,yvals.rows(),yvals.cols()) = yvals;
    Eigen::Map<Eigen::MatrixXd>(zd,Z.rows(),Z.cols()) = Z;

    pwl = pwl_interp_2d(nxd,nyd,xd,yd,zd,1,&x,&y);
}
