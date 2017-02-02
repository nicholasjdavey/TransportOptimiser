#include "../transportbase.h"

HorizontalAlignment::HorizontalAlignment() {
}

HorizontalAlignment::HorizontalAlignment(RoadPtr road) {

    unsigned long ip = road->getOptimiser()->getDesignParameters()->
        getIntersectionPoints();

    this->road = road;
    this->deltas.resize(ip);
    this->radii.resize(ip);
    this->radiiReq.resize(ip);
    this->pocx.resize(ip);
    this->pocy.resize(ip);
    this->potx.resize(ip);
    this->poty.resize(ip);
    this->mx.resize(ip);
    this->my.resize(ip);
    this->delx.resize(ip);
    this->dely.resize(ip);
    this->vel.resize(ip);
}

HorizontalAlignment::~HorizontalAlignment() {
}

void HorizontalAlignment::computeAlignment() {
    int maxiter = 1;
    RoadPtr roadPtrShared = this->road.lock();

    // Create short names for input data
    const Eigen::VectorXd& xFull = roadPtrShared->getXCoords();
    const Eigen::VectorXd& yFull = roadPtrShared->getYCoords();
    std::vector<bool> duplicates(xFull.size(),false);
    int uniqueEntries = 1;

    // If we have duplicate entries in our list, remove them for now and record
    // where they occur for later use
    for (int ii = 1; ii < xFull.size(); ii++) {
        if ((xFull(ii) == xFull(ii-1)) &&
            (yFull(ii) == yFull(ii-1))) {
                duplicates[ii] = true;
        } else {
            uniqueEntries++;
        }
    }

    Eigen::VectorXd xCoords(uniqueEntries);
    Eigen::VectorXd yCoords(uniqueEntries);
    xCoords(0) = xFull(0);
    yCoords(0) = yFull(0);

    for (int ii = 1; ii < xCoords.size(); ii++) {
        if (!duplicates[ii]) {
            xCoords(ii) = xFull(ii);
            yCoords(ii) = yFull(ii);
        }
    }

    if (xCoords.size() != yCoords.size()) {
        std::cerr << "X and Y vectors must be of the same length" << std::endl;
    } else {
        double maxSE = roadPtrShared->getOptimiser()->getDesignParameters()->getMaxSE();
        double designVel = roadPtrShared->getOptimiser()->getDesignParameters()
                ->getDesignVelocity();
        bool spiral = roadPtrShared->getOptimiser()->getDesignParameters()
                ->getSpiral();

        unsigned long ip = xCoords.size() - 2;
        Eigen::VectorXd vmax(ip);
        for (int ii = 0; ii < vmax.size(); ii++) {
            vmax(ii) = designVel;
        }

        // Side friction from AASHTO2004
        Eigen::VectorXd fsmax(vmax.size());
        this->sideFriction(vmax,fsmax);

        // Minimum curvature radius for desired speed (superelevations < 10deg)
        this->radiiReq = (vmax.array()).pow(2) / (9.8*(fsmax.array() + maxSE/100));

        // Initialise class attributes variables using predefined size. As we
        // have removed duplicates, the following lists also do not contain
        // duplicates.

        this->deltas.resize(ip);
        this->radii.resize(ip);
        this->pocx.resize(ip);
        this->pocy.resize(ip);
        this->potx.resize(ip);
        this->poty.resize(ip);
        this->mx.resize(ip);
        this->my.resize(ip);
        this->delx.resize(ip);
        this->dely.resize(ip);
        this->vel.resize(ip);

        // If we are designing with spiral transition curves (for safety)
        // In the following code, the indices for the circular arc points
        // are such that the first circle has index 1.
        if (spiral) {
        } else {
            // Simple circular transitions

            // Iterate the code below until a continuous path is formed. If the
            // maximum number of iterations is reached without a plausible
            // solution, the program exits with a failure flag.
            for (int ii = 1; ii < (xCoords.size() - 1); ii++) {
                // First pass (initialise all curves)
                this->deltas(ii-1) = this->computeDelta(xCoords,yCoords,ii);
                this->potx(ii-1) = this->computePOTX(this->radiiReq,
                        this->deltas,xCoords,yCoords,ii);
                this->poty(ii-1) = this->computePOTY(this->radiiReq,
                        this->deltas,xCoords,yCoords,ii);
                this->pocx(ii-1) = this->computePOCX(this->radiiReq,
                        this->deltas,xCoords,yCoords,ii);
                this->pocy(ii-1) = this->computePOCY(this->radiiReq,
                        this->deltas,xCoords,yCoords,ii);

                this->radii(ii-1) = this->radiiReq(ii-1);
                this->vel(ii-1) = vmax(ii-1);
            }

            ///////////////////////////////////////////////////////////////////
            // For the next lines, we work on a curve basis, therefore, index
            // ii refers to the iith curve, not the iith intersection point.

            // First and last arcs must be constrained so as not to overshoot
            // the start and end points.
            double cp2 = sqrt(pow(this->pocx(0)-xCoords(1),2)
                    +pow(this->pocy(0)-yCoords(1),2));
            double tanLen = sqrt(pow(xCoords(1)-xCoords(0),2)
                    +pow(yCoords(1)-yCoords(0),2));

            if ((cp2 - tanLen) > DBL_PREC) {
                // We must set the poc of the first arc to the start point
                this->radii(0) = radNewPOC(xCoords, yCoords, this->deltas, 1,
                        tanLen, xCoords(0), yCoords(0));

                this->radiiReq(0) = this->radii(0);

                this->potx(0) = this->computePOTX(this->radiiReq,
                        this->deltas,xCoords,yCoords,1);
                this->poty(0) = this->computePOTY(this->radiiReq,
                        this->deltas,xCoords,yCoords,1);
                this->pocx(0) = xCoords(0);
                this->pocy(0) = yCoords(0);
//                this->pocx(0) = this->computePOCX(this->radiiReq,
//                        this->deltas,xCoords,yCoords,1);
//                this->pocy(0) = this->computePOCY(this->radiiReq,
//                        this->deltas,xCoords,yCoords,1);
            }

            double tpn1 = sqrt(pow(this->potx(ip-1)-xCoords(ip),2)
                    + pow(this->poty(ip-1)-yCoords(ip),2));
            tanLen = sqrt(pow(xCoords(ip+1)-xCoords(ip),2)
                    + pow(yCoords(ip+1)-yCoords(ip),2));

            if ((tpn1 - tanLen) > DBL_PREC) {
                // We must set the pot of the last arc to the end point
                this->radii(ip-1) = radNewPOT(xCoords, yCoords, this->deltas, ip,
                        tanLen, xCoords(ip+1), yCoords(ip+1));
                this->radiiReq(ip-1) = this->radii(ip-1);

//                this->potx(ip-1) = this->computePOTX(this->radiiReq,
//                        this->deltas,xCoords,yCoords,ip);
//                this->poty(ip-1) = this->computePOTY(this->radiiReq,
//                        this->deltas,xCoords,yCoords,ip);
                this->potx(ip-1) = xCoords(ip+1);
                this->poty(ip-1) = yCoords(ip+1);
                this->pocx(ip-1) = this->computePOCX(this->radiiReq,
                        this->deltas,xCoords,yCoords,ip);
                this->pocy(ip-1) = this->computePOCY(this->radiiReq,
                        this->deltas,xCoords,yCoords,ip);
            }

            /* Now check if the curves created above are continuous. If not, we
             * must relax the minimum radius at certain points and recduce the
             * corresponding speed accordingly. We repeat this process until a
             * continuous curve is found or we run out of iterations. The
             * penalty for having small radius curves is that the speed will be
             * lower and the accident risk increases. We run the loop once
             * because we need to check the path anyway, even if it is correct.
             */
            int counter = 0;
            bool continuous = false;

            // We start at the second curve and go to the last one
            while (counter <= maxiter && !continuous) {
                for (unsigned int ii = 1; ii < ip; ii++) {
                    // Find midpoint of tangent line
                    double midx = 0.5*(xCoords(ii)+xCoords(ii+1));
                    double midy = 0.5*(yCoords(ii)+yCoords(ii+1));

                    /* If ||T(i-1)->P(i-1)|| + ||C(i)->P(i)|| >
                     * ||P(i-1)->P(i)||, we must reduce one or both of the radii
                     * to accomodate the curves in a way that ensures a
                     * continuous function.
                     */
                    double pArcLen = sqrt(pow(this->potx(ii-1)-xCoords(ii),2)+
                            pow(this->poty(ii-1)-yCoords(ii),2));
                    double cArcLen = sqrt(pow(xCoords(ii+1)-this->pocx(ii),2)+
                            pow(yCoords(ii+1)-this->pocy(ii),2));
                    tanLen = sqrt(pow(xCoords(ii+1)-xCoords(ii),2)+
                            pow(yCoords(ii+1)-yCoords(ii),2));

                    if (((pArcLen + cArcLen) - tanLen) > DBL_PREC) {
                        // We need to adjust the radii

                        if (((pArcLen - 0.5*tanLen) > DBL_PREC) &&
                                ((cArcLen - 0.5*tanLen) > DBL_PREC)) {
                            // Make both meet at the midpoint as a compromise
                            this->radii(ii-1) = this->radNewPOT(xCoords,
                                    yCoords, this->deltas, ii, tanLen,
                                    midx, midy);
                            this->radii(ii) = this->radNewPOC(xCoords,
                                    yCoords, this->deltas, ii+1, tanLen, midx,
                                    midy);

                            // Now adjust all previous values to reflect this
                            // change

                            // Previous circle
//                            this->potx(ii-1) = this->computePOTX(
//                                    this->radii, this->deltas, xCoords,
//                                    yCoords,ii);
//                            this->poty(ii-1) = this->computePOTY(
//                                    this->radii, this->deltas, xCoords,
//                                    yCoords,ii);
                            this->potx(ii-1) = midx;
                            this->poty(ii-1) = midy;
                            this->pocx(ii-1) = this->computePOCX(
                                    this->radii, this->deltas, xCoords,
                                    yCoords,ii);
                            this->pocy(ii-1) = this->computePOCY(
                                    this->radii, this->deltas, xCoords,
                                    yCoords,ii);

                            // Current circle
                            this->potx(ii) = this->computePOTX(this->radii,
                                    this->deltas,xCoords,yCoords,ii+1);
                            this->poty(ii) = this->computePOTY(this->radii,
                                    this->deltas,xCoords,yCoords,ii+1);
                            this->pocx(ii) = midx;
                            this->pocy(ii) = midy;
//                            this->pocx(ii) = this->computePOCX(this->radii,
//                                    this->deltas,xCoords,yCoords,ii+1);
//                            this->pocy(ii) = this->computePOCY(this->radii,
//                                    this->deltas,xCoords,yCoords,ii+1);
						
                        } else if ((pArcLen - 0.5*tanLen) > DBL_PREC) {
                            // Reduce the radius of the circle at the previous
                            // PI to meet the start of the current circle.
                            this->radii(ii-1) = this->radNewPOT(xCoords,
                                    yCoords, this->deltas, ii, tanLen,
                                    this->pocx(ii), this->pocy(ii));

//                            this->potx(ii-1) = this->computePOTX(
//                                    this->radii, this->deltas, xCoords,
//                                    yCoords,ii);
//                            this->poty(ii-1) = this->computePOTY(
//                                    this->radii, this->deltas, xCoords,
//                                    yCoords,ii);
                            this->potx(ii-1) = this->pocx(ii);
                            this->poty(ii-1) = this->pocy(ii);
                            this->pocx(ii-1) = this->computePOCX(
                                    this->radii, this->deltas, xCoords,
                                    yCoords,ii);
                            this->pocy(ii-1) = this->computePOCY(
                                    this->radii, this->deltas, xCoords,
                                    yCoords,ii);

                        } else {
                            // Reduce the radius of the current circle to meet
                            // the end of the circle at the previous PI
                            this->radii(ii) = this->radNewPOC(xCoords,
                                    yCoords, this->deltas, ii+1, tanLen,
                                    this->potx(ii-1), this->poty(ii-1));

                            this->potx(ii) = this->computePOTX(this->radii,
                                    this->deltas,xCoords,yCoords,ii+1);
                            this->poty(ii) = this->computePOTY(this->radii,
                                    this->deltas,xCoords,yCoords,ii+1);
                            this->pocx(ii) = this->potx(ii-1);
                            this->pocy(ii) = this->poty(ii-1);
//                            this->pocx(ii) = this->computePOCX(this->radii,
//                                    this->deltas,xCoords,yCoords,ii+1);
//                            this->pocy(ii) = this->computePOCY(this->radii,
//                                    this->deltas,xCoords,yCoords,ii+1);
                        }
                    }
                }

                counter++;

                // Now run through each arc to increase the radius where
                // possible. We don't do the start and end curves yet.
                for (unsigned int ii = 1; ii < ip-1; ii++) {
                    if ((this->radiiReq(ii) - this->radii(ii)) > DBL_PREC) {
                        double tan1 = sqrt(pow(xCoords(ii+1)-xCoords(ii),2)
                                + pow(yCoords(ii+1)-yCoords(ii),2));
                        double tan2 = sqrt(pow(xCoords(ii+2)-xCoords(ii+1),2)
                                + pow(yCoords(ii+2)-yCoords(ii+1),2));
                        double pTLen = sqrt(pow(this->potx(ii-1)-xCoords(ii),2)
                                + pow(this->poty(ii-1)-yCoords(ii),2));
                        double cCLen = sqrt(pow(xCoords(ii+1)-this->pocx(ii),2)
                                + pow(yCoords(ii+1)-this->pocy(ii),2));
                        double cTLen = sqrt(pow(this->potx(ii)-xCoords(ii+1),2)
                                + pow(this->poty(ii)-yCoords(ii+1),2));
                        double nCLen = sqrt(pow(xCoords(ii+2)
                                -this->pocx(ii+1),2)+ pow(yCoords(ii+2)
                                -this->pocy(ii+1),2));

                        // If there is room to expand the radius
                        if (((tan1 - (pTLen+cCLen)) > DBL_PREC ) &&
                                ((tan2 - (cTLen+nCLen)) > DBL_PREC)) {

                            double R1 = this->radNewPOC(xCoords,
                                    yCoords, this->deltas, ii+1, tan1,
                                    this->potx(ii-1), this->poty(ii-1));
                            double R2 = this->radNewPOT(xCoords,
                                    yCoords, this->deltas, ii+1, tan2,
                                    this->pocx(ii+1), this->pocy(ii+1));
                            this->radii(ii) = (float)std::min(
                                    this->radiiReq(ii),std::min(R1,R2));

                            // Now adjust all previous values to reflect this
                            // change
                            this->potx(ii) = this->computePOTX(this->radii,
                                    this->deltas,xCoords,yCoords,ii+1);
                            this->poty(ii) = this->computePOTY(this->radii,
                                    this->deltas,xCoords,yCoords,ii+1);
                            this->pocx(ii) = this->computePOCX(this->radii,
                                    this->deltas,xCoords,yCoords,ii+1);
                            this->pocy(ii) = this->computePOCY(this->radii,
                                    this->deltas,xCoords,yCoords,ii+1);
                        }
                    }
                }

                // Adjust the start and end points if there is now room to do
                // so

                // Start curve
                if ((this->radiiReq(0) - this->radii(0)) > DBL_PREC) {
                    double tan1 = sqrt(pow(xCoords(1)-xCoords(0),2)
                            + pow(yCoords(1)-yCoords(0),2));
                    double tan2 = sqrt(pow(xCoords(2)-xCoords(1),2)
                            + pow(yCoords(2)-yCoords(1),2));
                    double cCLen = sqrt(pow(xCoords(1)-this->pocx(0),2)
                            + pow(yCoords(1)-this->pocy(0),2));
                    double cTLen = sqrt(pow(this->potx(0)-xCoords(1),2)
                            + pow(this->poty(0)-yCoords(1),2));
                    double nCLen = sqrt(pow(xCoords(2)-this->pocx(1),2)
                            + pow(yCoords(2)-this->pocy(1),2));

                    // If there is room to expand the radius
                    if (((tan1 - cCLen) > DBL_PREC) &&
                            ((tan2 - (cTLen+nCLen)) > DBL_PREC)) {
                        double R2 = this->radNewPOT(xCoords,
                                yCoords, this->deltas, 1, tan2,
                                this->pocx(1), this->pocy(1));
                        this->radii(0) = std::min(this->radiiReq(0),R2);

                        // Now adjust all the previous values to reflect this
                        // change
                        this->potx(0) = this->computePOTX(this->radii,
                                this->deltas,xCoords,yCoords,1);
                        this->poty(0) = this->computePOTY(this->radii,
                                this->deltas,xCoords,yCoords,1);
                        this->pocx(0) = this->computePOCX(this->radii,
                                this->deltas,xCoords,yCoords,1);
                        this->pocy(0) = this->computePOCY(this->radii,
                                this->deltas,xCoords,yCoords,1);
                    }
                }

                // End Curve
                if ((this->radiiReq(ip-1) - this->radii(ip-1)) > DBL_PREC) {
                    double tan1 = sqrt(pow(xCoords(ip)-xCoords(ip-1),2)
                            + pow(yCoords(ip)-yCoords(ip-1),2));
                    double tan2 = sqrt(pow(xCoords(ip+1)-xCoords(ip),2)
                            + pow(yCoords(ip+1)-yCoords(ip),2));
                    double pTLen = sqrt(pow(this->potx(ip-2)-xCoords(ip-1),2)
                            + pow(this->poty(ip-2)-yCoords(ip),2));
                    double cCLen = sqrt(pow(xCoords(ip)-this->pocx(ip-1),2)
                            + pow(yCoords(ip)-this->pocy(ip-1),2));
                    double cTLen = sqrt(pow(this->potx(ip-1)-xCoords(ip),2)
                            + pow(this->poty(ip-1)-yCoords(ip),2));

                    // If there is room to expand the radius
                    if (((tan1 - (pTLen+cCLen)) > DBL_PREC) &&
                            ((tan2 - (cTLen)) > DBL_PREC)) {
                        double R1 = this->radNewPOC(xCoords, yCoords,
                                this->deltas, ip, tan1, this->potx(ip-2),
                                this->poty(ip-2));
                        this->radii(ip-1) = std::min(R1,this->radiiReq(ip-1));

                        // Now adjust all the previous values to reflect this
                        // change
                        this->potx(ip-1) = this->computePOTX(this->radiiReq,
                                this->deltas,xCoords,yCoords,ip);
                        this->poty(ip-1) = this->computePOTY(this->radiiReq,
                                this->deltas,xCoords,yCoords,ip);
                        this->pocx(ip-1) = this->computePOCX(this->radiiReq,
                                this->deltas,xCoords,yCoords,ip);
                        this->pocy(ip-1) = this->computePOCY(this->radiiReq,
                                this->deltas,xCoords,yCoords,ip);
                    }
                }

                /* Check if the new curve is continuous. We first assume that
                 * the curve is continuous everywhere and then test this
                 * assumption. As soon as a discontinuity is found, we set the
                 * continuity to false.
                 */
                bool continuous = true;

                for (unsigned int ii = 1; ii < ip; ii++) {
                    double pArcLen = sqrt(pow(this->potx(ii-1)-xCoords(ii),2)+
                            pow(this->poty(ii-1)-yCoords(ii),2));
                    double cArcLen = sqrt(pow(xCoords(ii+1)-this->pocx(ii),2)+
                            pow(yCoords(ii+1)-this->pocy(ii),2));
                    double tanLen = sqrt(pow(xCoords(ii+1)-xCoords(ii),2)+
                            pow(yCoords(ii+1)-yCoords(ii),2));

                    if (((pArcLen + cArcLen) - tanLen) > DBL_PREC) {
                        continuous = false;
                    }
                }
            }

            // Compute the chord, midpoint and velocity
            for (unsigned int ii = 0; ii < ip; ii++) {
                this->mx(ii) = 0.5*(this->pocx(ii)+this->potx(ii));
                this->my(ii) = 0.5*(this->pocy(ii)+this->poty(ii));

                this->delx(ii) = curveCentreX(xCoords, yCoords,
                        this->radii, this->deltas, this->mx, this->my, ii+1);
                this->dely(ii) = curveCentreY(xCoords, yCoords,
                        this->radii, this->deltas, this->mx, this->my, ii+1);

                if ((this->radiiReq(ii) - this->radii(ii)) > DBL_PREC) {
                    // Iterate to calculate the new velocity

                    unsigned int velcount = 1;
                    Eigen::VectorXd velprev(1);
                    Eigen::VectorXd sideFric(1);
                    velprev(0) = this->vel(ii);
                    double veldiff = 1;

                    while (veldiff > 0.05 && velcount <=5) {
                    this->sideFriction(velprev,sideFric);
                        this->vel(ii) = sqrt(this->radii(ii)*15*
                                (maxSE+sideFric(0)));
                        velprev(0) = this->vel(ii);
                        velcount = velcount + 1;
                        veldiff = std::abs(this->vel(ii)-velprev(0))/velprev(0);
                    }
                }
            }

            /* Compute vector of distances along the path where the distances
             * represent the distance travelled along the road from the
             * starting position.
             */

        }
    }
}

void HorizontalAlignment::sideFriction(const Eigen::VectorXd &vels,
        Eigen::VectorXd& fric) {

    for (int ii = 0; ii < fric.size(); ii++ ){
        if (vels(ii) < 0) {
            std::cerr << "Speed must be positive" << std::endl;
        } else if (vels(ii) <= 40/3.6) {
            fric(ii) = 0.21;
        } else if (vels(ii) <= 50/3.6) {
            fric(ii) = 0.21 - (vels(ii) - 40/3.6)*0.0108;
        } else if (vels(ii) <= 55/3.6) {
            fric(ii) = 0.18 - (vels(ii) - 50/3.6)*0.0216;
        } else if (vels(ii) <= 80/3.6) {
            fric(ii) = 0.15;
        } else if (vels(ii) <= 110/3.6) {
            fric(ii) = 0.15 - (vels(ii) - 80/3.6)*0.006;
        } else {
            fric(ii) = 0.1;
        }
    }
}

double HorizontalAlignment::computeDelta(const Eigen::VectorXd &xCoords,
        const Eigen::VectorXd &yCoords, int ii) {

    double val = ((xCoords(ii) - xCoords(ii-1)) *
                 (xCoords(ii+1) - xCoords(ii)) + (yCoords(ii)
                 - yCoords(ii-1)) * (yCoords(ii+1) - yCoords(ii))) /
                 (sqrt(pow((xCoords(ii) - xCoords(ii-1)), 2)
                 + pow((yCoords(ii) - yCoords(ii-1)), 2))
                 * sqrt(pow((xCoords(ii+1) - xCoords(ii)), 2)
                 + pow((yCoords(ii+1) - yCoords(ii)), 2)));

    val = (val > 1.0 ? 1.0 : (val < -1.0 ? -1.0 : val));

    double delta = acos(val);

    if (std::abs(delta) < (DBL_PREC)) {
        delta = 0;
    }

    return delta;
}

double HorizontalAlignment::computePOTX(const Eigen::VectorXd &rad,
    const Eigen::VectorXd &delta, const Eigen::VectorXd &xCoords,
    const Eigen::VectorXd &yCoords, int ii) {

    return xCoords(ii) + (rad(ii-1) * tan(delta(ii-1)/2)) *
            ((xCoords(ii+1) - xCoords(ii))/sqrt(pow(xCoords(ii+1)
            - xCoords(ii), 2) + pow(yCoords(ii+1) - yCoords(ii), 2)));
}

double HorizontalAlignment::computePOTY(const Eigen::VectorXd& rad,
    const Eigen::VectorXd& delta, const Eigen::VectorXd& xCoords,
    const Eigen::VectorXd& yCoords, int ii) {

    return yCoords(ii) + (rad(ii-1) * tan(delta(ii-1)/2)) *
            ((yCoords(ii+1) - yCoords(ii))/sqrt(pow(xCoords(ii+1)
            - xCoords(ii), 2) + pow(yCoords(ii+1) - yCoords(ii), 2)));
}

double HorizontalAlignment::computePOCX(const Eigen::VectorXd& rad,
    const Eigen::VectorXd& delta, const Eigen::VectorXd& xCoords,
    const Eigen::VectorXd& yCoords, int ii) {

    return xCoords(ii) + (rad(ii-1) * tan(delta(ii-1)/2)) *
            ((xCoords(ii-1) - xCoords(ii))/sqrt(pow(xCoords(ii-1)
            - xCoords(ii), 2) + pow(yCoords(ii-1) - yCoords(ii), 2)));
}

double HorizontalAlignment::computePOCY(const Eigen::VectorXd& rad,
    const Eigen::VectorXd& delta, const Eigen::VectorXd& xCoords,
    const Eigen::VectorXd& yCoords, int ii) {

    return yCoords(ii) + (rad(ii-1) * tan(delta(ii-1)/2)) *
            ((yCoords(ii-1) - yCoords(ii))/sqrt(pow(xCoords(ii-1)
            - xCoords(ii), 2) + pow(yCoords(ii-1) - yCoords(ii), 2)));
}

double HorizontalAlignment::poc2pi(const Eigen::VectorXd& xCoords,
    const Eigen::VectorXd& yCoords, const Eigen::VectorXd& pocx,
    const Eigen::VectorXd& pocy, int ii) {

    return sqrt(pow(pocx(ii)-xCoords(ii+1),2) +
            pow(pocy(ii)-yCoords(ii+1),2));
}

double HorizontalAlignment::pot2pi(const Eigen::VectorXd &xCoords,
    const Eigen::VectorXd &yCoords, const Eigen::VectorXd &potx,
    const Eigen::VectorXd &poty, int ii) {

    return sqrt(pow(potx(ii)-xCoords(ii+1),2) +
            pow(poty(ii)-yCoords(ii+1),2));
}

double HorizontalAlignment::pi2prev(const Eigen::VectorXd& xCoords,
    const Eigen::VectorXd& yCoords, int ii) {

    return sqrt(pow(xCoords(ii)-xCoords(ii+1),2) +
            pow(yCoords(ii)-yCoords(ii+1),2));
}

double HorizontalAlignment::pi2next(const Eigen::VectorXd& xCoords,
    const Eigen::VectorXd& yCoords, int ii) {

    return sqrt(pow(xCoords(ii+1)-xCoords(ii+2),2) +
            pow(yCoords(ii+1)-yCoords(ii+2),2));
}

double HorizontalAlignment::radNewPOC(const Eigen::VectorXd &xCoords,
    const Eigen::VectorXd &yCoords, const Eigen::VectorXd &delta, int ii,
    double tanlen, double revPOCX, double revPOCY) {

    if (xCoords(ii-1)!=xCoords(ii)) {
        return (revPOCX - xCoords(ii)) * tanlen / (tan(delta(ii-1)/2) *
                (xCoords(ii-1)-xCoords(ii)));
    } else {
        return (revPOCY - yCoords(ii)) * tanlen / (tan(delta(ii-1)/2) *
                (yCoords(ii-1)-yCoords(ii)));
    }
}

double HorizontalAlignment::radNewPOT(const Eigen::VectorXd& xCoords,
        const Eigen::VectorXd& yCoords, const Eigen::VectorXd& delta, int ii,
        double tanlen, double revPOTX, double revPOTY) {

    if (xCoords(ii)!=xCoords(ii+1)) {
        return (revPOTX - xCoords(ii)) * tanlen / (tan(delta(ii-1)/2) *
                (xCoords(ii+1)-xCoords(ii)));
    } else {
        return (revPOTY - yCoords(ii)) * tanlen / (tan(delta(ii-1)/2) *
                (yCoords(ii+1)-yCoords(ii)));
    }
}

double HorizontalAlignment::curveCentreX(const Eigen::VectorXd &xCoords,
    const Eigen::VectorXd &yCoords, const Eigen::VectorXd &rad,
    const Eigen::VectorXd &delta, const Eigen::VectorXd &mx,
    const Eigen::VectorXd &my, int ii) {

    return xCoords(ii) + (rad(ii-1))*(1/cos((delta(ii-1))/2))*(mx(ii-1)
            -xCoords(ii))/sqrt(pow((mx(ii-1)-xCoords(ii)),2)
            +pow((my(ii-1)-yCoords(ii)),2));
}

double HorizontalAlignment::curveCentreY(const Eigen::VectorXd& xCoords,
    const Eigen::VectorXd& yCoords, const Eigen::VectorXd& rad,
    const Eigen::VectorXd& delta, const Eigen::VectorXd& mx,
    const Eigen::VectorXd& my, int ii) {

    return yCoords(ii) + (rad(ii-1))*(1/cos((delta(ii-1))/2))*(my(ii-1)
            -yCoords(ii))/sqrt(pow((mx(ii-1)-xCoords(ii)),2)
            +pow((my(ii-1)-yCoords(ii)),2));
}
