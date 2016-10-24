#include "../transportbase.h"

Eigen::MatrixXi Utility::lineSegmentIntersect(const Eigen::MatrixXd& XY1,
        const Eigen::MatrixXd& XY2){

    Eigen::MatrixXd X1 = XY1.block(0,0,XY1.rows(),1).replicate(1,XY2.rows());
    Eigen::MatrixXd X2 = XY1.block(0,2,XY1.rows(),1).replicate(1,XY2.rows());
    Eigen::MatrixXd Y1 = XY1.block(0,1,XY1.rows(),1).replicate(1,XY2.rows());
    Eigen::MatrixXd Y2 = XY1.block(0,3,XY1.rows(),1).replicate(1,XY2.rows());

    Eigen::MatrixXd XY2T = XY2.transpose();

    Eigen::MatrixXd X3 = XY2.block(0,0,1,XY2.cols()).replicate(XY1.rows(),1);
    Eigen::MatrixXd X4 = XY2.block(0,2,1,XY2.cols()).replicate(XY1.rows(),1);
    Eigen::MatrixXd Y3 = XY2.block(0,1,1,XY2.cols()).replicate(XY1.rows(),1);
    Eigen::MatrixXd Y4 = XY2.block(0,3,1,XY2.cols()).replicate(XY1.rows(),1);

    Eigen::MatrixXd X4_X3 = (X4-X3);
    Eigen::MatrixXd Y1_Y3 = (Y1-Y3);
    Eigen::MatrixXd Y4_Y3 = (Y4-Y3);
    Eigen::MatrixXd X1_X3 = (X1-X3);
    Eigen::MatrixXd X2_X1 = (X2-X1);
    Eigen::MatrixXd Y2_Y1 = (Y2-Y1);

    Eigen::MatrixXd numerator_a = X4_X3.array() * Y1_Y3.array() -
            Y4_Y3.array() * X1_X3.array();
    Eigen::MatrixXd numerator_b = X2_X1.array() * Y1_Y3.array() -
            Y2_Y1.array() * X1_X3.array();
    Eigen::MatrixXd denominator = Y4_Y3.array() * X2_X1.array() -
            X4_X3.array() * Y2_Y1.array();

    Eigen::MatrixXd u_a = numerator_a.array() / denominator.array();
    Eigen::MatrixXd u_b = numerator_b.array() / denominator.array();

    // Find adjacency matrix
    Eigen::MatrixXi INT_B = ((u_a.array() >= 0) && (u_a.array() <= 1) &&
            (u_b.array() >= 0) && (u_b.array() <= 1)).cast<int>();

    // Return the number of times each of the input lines intersects the road
    return INT_B.rowwise().sum();
}

void Utility::lineSegmentIntersect(
        const Eigen::MatrixXd& XY1, const Eigen::MatrixXd& XY2,
        Eigen::Matrix<bool, Eigen::Dynamic,Eigen::Dynamic>& adjMat,
        Eigen::MatrixXd& intMatX, Eigen::MatrixXd& intMatY,
        Eigen::MatrixXd& normDist1to2, Eigen::MatrixXd& normDist2to1,
        Eigen::Matrix<bool, Eigen::Dynamic, Eigen::Dynamic>& parAdjMat,
        Eigen::Matrix<bool, Eigen::Dynamic, Eigen::Dynamic>& coincAdjMat) {

    Eigen::MatrixXd X1 = XY1.block(0,0,XY1.rows(),1).replicate(1,XY2.rows());
    Eigen::MatrixXd X2 = XY1.block(0,2,XY1.rows(),1).replicate(1,XY2.rows());
    Eigen::MatrixXd Y1 = XY1.block(0,1,XY1.rows(),1).replicate(1,XY2.rows());
    Eigen::MatrixXd Y2 = XY1.block(0,3,XY1.rows(),1).replicate(1,XY2.rows());

    Eigen::MatrixXd XY2T = XY2.transpose();

    Eigen::MatrixXd X3 = XY2.block(0,0,1,XY2.cols()).replicate(XY1.rows(),1);
    Eigen::MatrixXd X4 = XY2.block(0,2,1,XY2.cols()).replicate(XY1.rows(),1);
    Eigen::MatrixXd Y3 = XY2.block(0,1,1,XY2.cols()).replicate(XY1.rows(),1);
    Eigen::MatrixXd Y4 = XY2.block(0,3,1,XY2.cols()).replicate(XY1.rows(),1);

    Eigen::MatrixXd X4_X3 = (X4-X3);
    Eigen::MatrixXd Y1_Y3 = (Y1-Y3);
    Eigen::MatrixXd Y4_Y3 = (Y4-Y3);
    Eigen::MatrixXd X1_X3 = (X1-X3);
    Eigen::MatrixXd X2_X1 = (X2-X1);
    Eigen::MatrixXd Y2_Y1 = (Y2-Y1);

    Eigen::MatrixXd numerator_a = X4_X3.array() * Y1_Y3.array() -
            Y4_Y3.array() * X1_X3.array();
    Eigen::MatrixXd numerator_b = X2_X1.array() * Y1_Y3.array() -
            Y2_Y1.array() * X1_X3.array();
    Eigen::MatrixXd denominator = Y4_Y3.array() * X2_X1.array() -
            X4_X3.array() * Y2_Y1.array();

    Eigen::MatrixXd u_a = numerator_a.array() / denominator.array();
    Eigen::MatrixXd u_b = numerator_b.array() / denominator.array();

    // Find adjacency matrix
    Eigen::MatrixXd INT_X = X1.array() + X2_X1.array()*u_a.array();
    Eigen::MatrixXd INT_Y = Y1.array() + Y2_Y1.array()*u_a.array();
    Eigen::MatrixXi INT_B = ((u_a.array() >= 0) && (u_a.array() <= 1) &&
            (u_b.array() >= 0) && (u_b.array() <= 1)).cast<int>();
    adjMat = INT_B.cast<bool>();
    intMatX = INT_X.array()*INT_B.array().cast<double>();
    intMatY = INT_Y.array()*INT_B.array().cast<double>();
    normDist1to2 = u_a;
    normDist2to1 = u_b;
    parAdjMat = (denominator.array() == 0);
    coincAdjMat = ((numerator_a.array() == 0) && (numerator_b.array() == 0)
            && (parAdjMat.array()));
}

// The below function is a fast alternative to the Boost libraries, which are
// somewhat overkill for what we require.
double Utility::NormalCDFInverse(double p) {
    if (p <= 0.0 || p >= 1.0)
    {
        std::stringstream os;
        os << "Invalid input argument (" << p
        << "); must be larger than 0 but less than 1.";
        throw std::invalid_argument( os.str() );
    }

    if (p < 0.5)
    {
        // F^-1(p) = - G^-1(p)
        return -RationalApproximation( sqrt(-2.0*log(p)) );
    }
    else
    {
        // F^-1(p) = G^-1(1-p)
        return RationalApproximation( sqrt(-2.0*log(1-p)) );
    }
}

double Utility::RationalApproximation(double t) {
    // Abramowitz and Stegun formula 26.2.23.
    // The absolute value of the error should be less than 4.5 e-4.
    double c[] = {2.515517, 0.802853, 0.010328};
    double d[] = {1.432788, 0.189269, 0.001308};
    return t - ((c[2]*t + c[1])*t + c[0]) /
                (((d[2]*t + d[1])*t + d[0])*t + 1.0);
}

void Utility::cuttingPlanes(const double &xMin, const double &xMax, const
        double &yMin, const double &yMax, const double &xS, const double &yS,
        const double &zS, const double &xE, const double &yE, const double &zE,
        const long& n, Eigen::VectorXd &xO, Eigen::VectorXd &yO,
        Eigen::VectorXd &zO, Eigen::VectorXd &dU, Eigen::VectorXd &dL, double
        &theta) {

    Eigen::VectorXd ii = Eigen::VectorXd::LinSpaced(n,1,n);

    xO = xS + (ii.array()*(xE-xS))/(n+1);
    yO = yS + (ii.array()*(yE-yS))/(n+1);
    zO = zS + (ii.array()*(zE-zS))/(n+1);

    // Angles between cutting planes and x-axis
    theta = atan((yE-yS)/(xE-xS)) + M_PI/2;

    // Compute upper and lower bounds for cutting planes
    if(theta == 0 || theta == M_PI) {
        dU = xMax - xO.array();
        dL = xMin - xO.array();
    } else if (0 < theta < M_PI/2) {
        dU = ((xMax - xO.array())/cos(theta)).min(
                (yMax - yO.array())/sin(theta));
        dL = ((xMin - xO.array())/cos(theta)).max(
                (yMin - yO.array())/sin(theta));
    } else if (theta == M_PI/2) {
        dU = yMax - yO.array();
        dL = yMin - yO.array();
    } else if (M_PI/2 < theta < M_PI) {
        dU = ((xMin - xO.array())/cos(theta)).min(
                (yMax - yO.array())/sin(theta));
        dL = ((xMax - xO.array())/cos(theta)).max(
                (yMin - yO.array())/sin(theta));
    }
}
