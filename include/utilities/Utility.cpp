#include "../transportbase.h"

Eigen::MatrixXi Utility::lineSegmentIntersect(const Eigen::MatrixXd& XY1,
        const Eigen::MatrixXd& XY2,bool gpu){

    if (gpu) {
        Eigen::VectorXi crossings = Eigen::VectorXi::Zero(XY1.rows());
        SimulateGPU::lineSegmentIntersect(XY1,XY2,crossings);
        return crossings;
    } else {
        clock_t begin = clock();

        Eigen::MatrixXd X1 = XY1.block(0,0,XY1.rows(),1).replicate(1,XY2.
                rows());
        Eigen::MatrixXd X2 = XY1.block(0,2,XY1.rows(),1).replicate(1,XY2.
                rows());
        Eigen::MatrixXd Y1 = XY1.block(0,1,XY1.rows(),1).replicate(1,XY2.
                rows());
        Eigen::MatrixXd Y2 = XY1.block(0,3,XY1.rows(),1).replicate(1,XY2.
                rows());

        clock_t end = clock();
        double elapsed_secs = double(end - begin) / CLOCKS_PER_SEC;
        std::cout << "\t Block 1: " << elapsed_secs << " s" << std::endl;

        begin = clock();

        Eigen::MatrixXd XY2T = XY2.transpose();

        Eigen::MatrixXd X3 = XY2T.block(0,0,1,XY2T.cols()).replicate(XY1.
                rows(),1);
        Eigen::MatrixXd X4 = XY2T.block(2,0,1,XY2T.cols()).replicate(XY1.
                rows(),1);
        Eigen::MatrixXd Y3 = XY2T.block(1,0,1,XY2T.cols()).replicate(XY1.
                rows(),1);
        Eigen::MatrixXd Y4 = XY2T.block(3,0,1,XY2T.cols()).replicate(XY1.
                rows(),1);

        end = clock();
        elapsed_secs = double(end - begin) / CLOCKS_PER_SEC;
        std::cout << "\t Block 2: " << elapsed_secs << " s" << std::endl;

        begin = clock();

        Eigen::MatrixXd X4_X3 = (X4-X3);
        Eigen::MatrixXd Y1_Y3 = (Y1-Y3);
        Eigen::MatrixXd Y4_Y3 = (Y4-Y3);
        Eigen::MatrixXd X1_X3 = (X1-X3);
        Eigen::MatrixXd X2_X1 = (X2-X1);
        Eigen::MatrixXd Y2_Y1 = (Y2-Y1);

        end = clock();
        elapsed_secs = double(end - begin) / CLOCKS_PER_SEC;
        std::cout << "\t Block 3: " << elapsed_secs << " s" << std::endl;

        begin = clock();

        Eigen::MatrixXd numerator_a = X4_X3.array() * Y1_Y3.array() -
                Y4_Y3.array() * X1_X3.array();
        Eigen::MatrixXd numerator_b = X2_X1.array() * Y1_Y3.array() -
                Y2_Y1.array() * X1_X3.array();
        Eigen::MatrixXd denominator = Y4_Y3.array() * X2_X1.array() -
                X4_X3.array() * Y2_Y1.array();

        end = clock();
        elapsed_secs = double(end - begin) / CLOCKS_PER_SEC;
        std::cout << "\t Block 4: " << elapsed_secs << " s" << std::endl;

        begin = clock();

        Eigen::MatrixXd u_a = numerator_a.array() / denominator.array();
        Eigen::MatrixXd u_b = numerator_b.array() / denominator.array();

        end = clock();
        elapsed_secs = double(end - begin) / CLOCKS_PER_SEC;
        std::cout << "\t Block 5: " << elapsed_secs << " s" << std::endl;

        begin = clock();

        // Find adjacency matrix
        Eigen::MatrixXi INT_B = ((u_a.array() >= 0) && (u_a.array() <= 1) &&
                (u_b.array() >= 0) && (u_b.array() <= 1)).cast<int>();

        end = clock();
        elapsed_secs = double(end - begin) / CLOCKS_PER_SEC;
        std::cout << "\t Block 6: " << elapsed_secs << " s" << std::endl;

        // Return the number of times each of the input lines intersects the
        // road
        return INT_B.rowwise().sum();
    }
}

void Utility::lineSegmentIntersect(
        const Eigen::MatrixXd& XY1, const Eigen::MatrixXd& XY2,
        Eigen::Matrix<bool, Eigen::Dynamic,Eigen::Dynamic>& adjMat,
        Eigen::MatrixXd& intMatX, Eigen::MatrixXd& intMatY,
        Eigen::MatrixXd& normDist1to2, Eigen::MatrixXd& normDist2to1,
        Eigen::Matrix<bool, Eigen::Dynamic, Eigen::Dynamic>& parAdjMat,
        Eigen::Matrix<bool, Eigen::Dynamic, Eigen::Dynamic>& coincAdjMat) {

    clock_t begin = clock();

    Eigen::MatrixXd X1 = XY1.block(0,0,XY1.rows(),1).replicate(1,XY2.rows());
    Eigen::MatrixXd X2 = XY1.block(0,2,XY1.rows(),1).replicate(1,XY2.rows());
    Eigen::MatrixXd Y1 = XY1.block(0,1,XY1.rows(),1).replicate(1,XY2.rows());
    Eigen::MatrixXd Y2 = XY1.block(0,3,XY1.rows(),1).replicate(1,XY2.rows());

    Eigen::MatrixXd XY2T = XY2.transpose();

    clock_t end = clock();
    double elapsed_secs = double(end - begin) / CLOCKS_PER_SEC;
    std::cout << "\t Block 1: " << elapsed_secs << " s" << std::endl;

    begin = clock();

    Eigen::MatrixXd X3 = XY2T.block(0,0,1,XY2T.cols()).replicate(XY1.rows(),1);
    Eigen::MatrixXd X4 = XY2T.block(2,0,1,XY2T.cols()).replicate(XY1.rows(),1);
    Eigen::MatrixXd Y3 = XY2T.block(1,0,1,XY2T.cols()).replicate(XY1.rows(),1);
    Eigen::MatrixXd Y4 = XY2T.block(3,0,1,XY2T.cols()).replicate(XY1.rows(),1);

    Eigen::MatrixXd X4_X3 = (X4-X3);
    Eigen::MatrixXd Y1_Y3 = (Y1-Y3);
    Eigen::MatrixXd Y4_Y3 = (Y4-Y3);
    Eigen::MatrixXd X1_X3 = (X1-X3);
    Eigen::MatrixXd X2_X1 = (X2-X1);
    Eigen::MatrixXd Y2_Y1 = (Y2-Y1);

    end = clock();
    elapsed_secs = double(end - begin) / CLOCKS_PER_SEC;
    std::cout << "\t Block 2: " << elapsed_secs << " s" << std::endl;

    begin = clock();

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

    elapsed_secs = double(end - begin) / CLOCKS_PER_SEC;
    std::cout << "\t Block 3: " << elapsed_secs << " s" << std::endl;
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
    } else if ((0 < theta) && (theta < M_PI/2)) {
        dU = ((xMax - xO.array())/cos(theta)).min(
                (yMax - yO.array())/sin(theta));
        dL = ((xMin - xO.array())/cos(theta)).max(
                (yMin - yO.array())/sin(theta));
    } else if (theta == M_PI/2) {
        dU = yMax - yO.array();
        dL = yMin - yO.array();
    } else if ((M_PI/2 < theta) && (theta < M_PI)) {
        dU = ((xMin - xO.array())/cos(theta)).min(
                (yMax - yO.array())/sin(theta));
        dL = ((xMax - xO.array())/cos(theta)).max(
                (yMin - yO.array())/sin(theta));
    }
}

void Utility::ksrlin_vw(const Eigen::VectorXd& x, const Eigen::VectorXd& y,
        const int ww, const int n, Eigen::VectorXd& rx, Eigen::VectorXd& ry) {

    // Check input vectors are of the same size
    if (x.size() != y.size()) {
        throw std::invalid_argument("The input x and y vectors must be of type double and the same length.");
    }

    // Check output vectors are of the required length
    if ((rx.size() != n) || (ry.size() != n)) {
        throw std::invalid_argument("The output rx and ry vectors must be the same as the desired resolution (n)");
    }

    // Resolution must be specified

    // Gaussian kernel function
    auto kerf = [](double z){return exp(-z*z)/sqrt(2*M_PI);};

    rx = Eigen::VectorXd::LinSpaced(n,x.minCoeff(),x.maxCoeff());
    ry = Eigen::VectorXd::Zero(n);
}

double Utility::interpolateSurrogate(Eigen::VectorXd& surrogate,
        Eigen::VectorXd &predictors, int dimRes) {

    // First find global the upper and lower bounds in each dimension as well
    // as the index of the lower bound of the regressed value in each
    // dimension.
    Eigen::VectorXd lowerVal(predictors.size());
    Eigen::VectorXd upperVal(predictors.size());
    Eigen::VectorXd coeffs(pow(2,predictors.size()-1));
    Eigen::VectorXd lowerIdx(predictors.size());

    for (int ii = 0; ii < predictors.size(); ii++) {
        lowerVal(ii) = surrogate[ii*dimRes];
        upperVal(ii) = surrogate[(ii+1)*dimRes - 1];
        lowerIdx(ii) = (int)(dimRes*(predictors(ii) - lowerVal(ii))/(
                upperVal(ii) - lowerVal(ii)));
    }

    // Now that we have all the index requirements, let's interpolate.
    // Get the uppermost dimension x value
    double x0 = surrogate(lowerIdx(0));
    double x1 = surrogate(lowerIdx(0)+1);
    double xd = (predictors(0) - x0)/(x1 - x0);

    // First, assign the yvalues to the coefficients matrix
    for (int ii = 0; ii < (int)pow(2,predictors.size()-1); ii++) {
        // Get the indices of the yvalues of the lower and upper bounding
        // values on this dimension.
        int idxL = predictors.size()*dimRes;

        for (int jj = 0; jj < (predictors.size() - 1); jj++) {
            int rem = ((int)(ii/((int)pow(2,predictors.size()-1-jj))) + 1) -
                    2*(int)(((int)(ii/((int)pow(2,predictors.size()-1-jj))) +
                    1)/2);

            if (rem > 0) {
                idxL += lowerIdx[jj]*(int)pow(dimRes,predictors.size()-1-jj);
            } else {
                idxL += (lowerIdx[jj]+1)*(int)pow(dimRes,predictors.size()-1-
                        jj);
            }
        }

        int idxU = idxL + (lowerIdx[0]+1)*(int)pow(dimRes,predictors.size()-1);

        idxL += idxL + lowerIdx[0]*(int)pow(dimRes,predictors.size()-1);

        coeffs[ii] = surrogate(idxL)*(1 - xd) + surrogate(idxU)*xd;
    }

    // Now we work our way down the dimensions using our computed coefficients
    // to get the interpolated value.
    for (int ii = 1; ii < predictors.size(); ii++) {
        // Get the current dimensions x value
        x0 = surrogate(lowerIdx[ii] + dimRes*ii);
        x1 = surrogate(lowerIdx[0]+1 + dimRes*ii);
        xd = (predictors(0) - x0)/(x1 - x0);

        for (int jj = 0; jj < (int)pow(2,ii); jj++) {
            int jump = (int)pow(2,predictors.size()-ii-2);
            coeffs[jj] = coeffs[jj]*(1 - xd) + coeffs[jj+jump]*xd;
        }
    }

    return coeffs[0];
}
