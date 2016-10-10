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
		this->radiiReq = fsmax.array().pow(2) / (fsmax.array() + 15*maxSE);

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

				this->radii(ii) = this->radiiReq(ii);
				this->vel(ii) = vmax(ii);
			}

			///////////////////////////////////////////////////////////////////
			// For the next lines, we work on a curve basis, therefore, index
			// 0 refers to the first curve, not the first intersection point.

			// First and last arcs must be constrained so as not to overshoot
			// the start and end points.
			double cp2 = sqrt(pow(this->pocx(0)-xCoords(1),2)
					+pow(this->pocy(0)-yCoords(1),2));
			double tanlen = sqrt(pow(xCoords(1)-xCoords(0),2)
					+pow(yCoords(1)-yCoords(0),2));

			if (cp2 > tanlen) {
				// We must set the poc of the first arc to the start point
                this->radii(0) = radNewPOC(xCoords, yCoords, this->deltas, 0,
					tanlen, xCoords(0), yCoords(0));

				this->radiiReq(0) = this->radii(0);

                this->potx(0) = this->computePOTX(this->radiiReq,
                        this->deltas,xCoords,yCoords,0);
                this->poty(0) = this->computePOTY(this->radiiReq,
                        this->deltas,xCoords,yCoords,0);
                this->pocx(0) = this->computePOCX(this->radiiReq,
                        this->deltas,xCoords,yCoords,0);
                this->pocy(0) = this->computePOCY(this->radiiReq,
                        this->deltas,xCoords,yCoords,0);
			}

			double tpn1 = sqrt(pow(this->potx(ip-1)-xCoords(ip+1),2) 
					+ pow(this->poty(ip-1)-yCoords(ip+1),2));
			tanlen = sqrt(pow(xCoords(ip+2)-xCoords(ip+1),2)
					+ pow(yCoords(ip+2)-yCoords(ip+1),2));

			if (tpn1 > tanlen) {
				// We must set the pot of the last arc to the end point
                this->radii(ip-1) = radNewPOT(xCoords, yCoords, this->deltas, ip-1,
					tanlen, xCoords(ip+1), yCoords(ip+1));
				this->radiiReq(ip-1) = this->radii(ip-1);

                this->potx(ip-1) = this->computePOTX(this->radiiReq,
                        this->deltas,xCoords,yCoords,ip-1);
                this->poty(ip-1) = this->computePOTY(this->radiiReq,
                        this->deltas,xCoords,yCoords,ip-1);
                this->pocx(ip-1) = this->computePOCX(this->radiiReq,
                        this->deltas,xCoords,yCoords,ip-1);
                this->pocy(ip-1) = this->computePOCY(this->radiiReq,
                        this->deltas,xCoords,yCoords,ip-1);
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

			while (counter <= maxiter && !continuous) {
				// Find midpoint of tangent line
				for (unsigned int ii = 1; ii < ip-1; ii++) {
					double midx = 0.5*(xCoords(ii)+xCoords(ii+1));
					double midy = 0.5*(yCoords(ii)+xCoords(ii+1));

					/* If ||T(i-1)->P(i-1)|| + ||C(i)->P(i)|| >
					 * ||P(i-1)->P(i)||, we must reduce one or both of the radii
					 * to accomodate the curves in a way that ensures a
					 * continuous function.
					 */
					double pArcLen = sqrt(pow(this->potx(ii-1)-xCoords(ii),2)+
							pow(this->poty(ii-1)-yCoords(ii),2));
					double cArcLen = sqrt(pow(xCoords(ii+1)-this->pocx(ii),2)+
							pow(yCoords(ii+1)-this->pocy(ii),2));
					double tanLen = sqrt(pow(xCoords(ii+1)-xCoords(ii),2)+
							pow(yCoords(ii+1)-yCoords(ii),2));

					if (tanLen < (pArcLen + cArcLen)) {
						// We need to adjust the radii

						if (pArcLen > 0.5*tanLen && cArcLen > 0.5*tanLen) {
							// Make both meet at the midpoint as a compromise
                            this->radii(ii-1) = this->radNewPOT(xCoords,
                                    yCoords, this->deltas, ii-1, tanlen,
									midx, midy);
                            this->radii(ii) = this->radNewPOC(xCoords,
                                    yCoords, this->deltas, ii, tanlen, midx,
									midy);

							// Now adjust all previous values to reflect this
							// change

							// Previous circle
							this->potx(ii-1) = this->computePOTX(
                                    this->radiiReq, this->deltas, xCoords,
                                    yCoords,ii-1);
							this->poty(ii-1) = this->computePOTY(
                                    this->radiiReq, this->deltas, xCoords,
                                    yCoords,ii-1);
							this->pocx(ii-1) = this->computePOCX(
                                    this->radiiReq, this->deltas, xCoords,
                                    yCoords,ii-1);
							this->pocy(ii-1) = this->computePOCY(
                                    this->radiiReq, this->deltas, xCoords,
                                    yCoords,ii-1);

							// Current circle
                            this->potx(ii) = this->computePOTX(this->radiiReq,
                                    this->deltas,xCoords,yCoords,ii);
                            this->poty(ii) = this->computePOTY(this->radiiReq,
                                    this->deltas,xCoords,yCoords,ii);
                            this->pocx(ii) = this->computePOCX(this->radiiReq,
                                    this->deltas,xCoords,yCoords,ii);
                            this->pocy(ii) = this->computePOCY(this->radiiReq,
                                    this->deltas,xCoords,yCoords,ii);
						
						} else if (pArcLen > 0.5*tanLen) {
							// Reduce the radius of the circle at the previous
							// PI to meet the start of the current circle.
                            this->radii(ii-1) = this->radNewPOT(xCoords,
                                    yCoords, this->deltas, ii-1, tanlen,
									this->pocx(ii), this->pocy(ii));

							this->potx(ii-1) = this->computePOTX(
                                    this->radiiReq, this->deltas, xCoords,
                                    yCoords,ii-1);
							this->poty(ii-1) = this->computePOTY(
                                    this->radiiReq, this->deltas, xCoords,
                                    yCoords,ii-1);
							this->pocx(ii-1) = this->computePOCX(
                                    this->radiiReq, this->deltas, xCoords,
                                    yCoords,ii-1);
							this->pocy(ii-1) = this->computePOCY(
                                    this->radiiReq, this->deltas, xCoords,
                                    yCoords,ii-1);

						} else {
							// Reduce the radius of the current circle to meet
							// the end of the circle at the previous PI
                            this->radii(ii) = this->radNewPOC(xCoords,
                                    yCoords, this->deltas, ii, tanlen,
									this->potx(ii-1), this->poty(ii-1));

                            this->potx(ii) = this->computePOTX(this->radiiReq,
                                    this->deltas,xCoords,yCoords,ii);
                            this->poty(ii) = this->computePOTY(this->radiiReq,
                                    this->deltas,xCoords,yCoords,ii);
                            this->pocx(ii) = this->computePOCX(this->radiiReq,
                                    this->deltas,xCoords,yCoords,ii);
                            this->pocy(ii) = this->computePOCY(this->radiiReq,
                                    this->deltas,xCoords,yCoords,ii);
						}
					}
				}

				counter++;

				// Now run through each arc to increase the radius where
				// possible
				for (unsigned int ii = 1; ii < ip-1; ii++) {
					if (this->radii(ii) < this->radiiReq(ii)) {
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
						if ((pTLen+cCLen)<tan1 && (cTLen+nCLen)<tan2) {

                            double R1 = this->radNewPOC(xCoords,
                                    yCoords, this->deltas, ii, tan1,
									this->potx(ii-1), this->poty(ii-1));
                            double R2 = this->radNewPOT(xCoords,
                                    yCoords, this->deltas, ii, tan2,
									this->pocx(ii+1), this->pocy(ii+1));
							this->radii(ii) = std::min(this->radiiReq(ii),
									std::min(R1,R2));

							// Now adjust all previous values to reflect this
							// change
                            this->potx(ii) = this->computePOTX(this->radiiReq,
                                    this->deltas,xCoords,yCoords,ii);
                            this->poty(ii) = this->computePOTY(this->radiiReq,
                                    this->deltas,xCoords,yCoords,ii);
                            this->pocx(ii) = this->computePOCX(this->radiiReq,
                                    this->deltas,xCoords,yCoords,ii);
                            this->pocy(ii) = this->computePOCY(this->radiiReq,
                                    this->deltas,xCoords,yCoords,ii);
						}
					}
				}

				// Adjust the start and end points if there is now room to do
				// so

				// Start curve
				if (this->radii(0) < this->radiiReq(0)) {
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
					if (cCLen<tan1 && (cTLen+nCLen)<tan2) {
                        double R2 = this->radNewPOT(xCoords,
                                yCoords, this->deltas, 0, tan2,
								this->pocx(1), this->pocy(1));
						this->radii(0) = std::min(this->radiiReq(0),R2);

						// Now adjust all the previous values to reflect this
						// change
                        this->potx(0) = this->computePOTX(this->radiiReq,
                                this->deltas,xCoords,yCoords,0);
                        this->poty(0) = this->computePOTY(this->radiiReq,
                                this->deltas,xCoords,yCoords,0);
                        this->pocx(0) = this->computePOCX(this->radiiReq,
                                this->deltas,xCoords,yCoords,0);
                        this->pocy(0) = this->computePOCY(this->radiiReq,
                                this->deltas,xCoords,yCoords,0);
					}
				}

				// End Curve
				if (this->radii(ip-1) < this->radiiReq(ip-1)) {
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
					if ((pTLen+cCLen)<tan1 && (cTLen)<tan2) {
                        double R1 = this->radNewPOC(xCoords, yCoords,
                                this->deltas, ip-1, tan1, this->potx(ip-2),
								this->poty(ip-2));
						this->radii(ip-1) = std::min(R1,this->radii(ip-1));

						// Now adjust all the previous values to reflect this
						// change
                        this->potx(ip-1) = this->computePOTX(this->radiiReq,
                                this->deltas,xCoords,yCoords,ip-1);
                        this->poty(ip-1) = this->computePOTY(this->radiiReq,
                                this->deltas,xCoords,yCoords,ip-1);
                        this->pocx(ip-1) = this->computePOCX(this->radiiReq,
                                this->deltas,xCoords,yCoords,ip-1);
                        this->pocy(ip-1) = this->computePOCY(this->radiiReq,
                                this->deltas,xCoords,yCoords,ip-1);
					}
				}

				/* Check if the new curve is continuous. We first assume that
				 * the curve is continuous everywhere and then test this
				 * assumption. As soon as a discontinuity is found, we set the
				 * continuity to false.
				 */
				bool continuous = false;

				for (unsigned int ii = 1; ii < ip; ii++) {
					double pArcLen = sqrt(pow(this->potx(ii-1)-xCoords(ii),2)+
							pow(this->poty(ii-1)-yCoords(ii),2));
					double cArcLen = sqrt(pow(xCoords(ii+1)-this->pocx(ii),2)+
							pow(yCoords(ii+1)-this->pocy(ii),2));
					double tanLen = sqrt(pow(xCoords(ii+1)-xCoords(ii),2)+
							pow(yCoords(ii+1)-yCoords(ii),2));

					if (tanLen*1.000001 < (pArcLen + cArcLen)) {
						continuous = false;
					}
				}
			}

			// Compute the chord, midpoint and velocity
			for (unsigned int ii = 0; ii < ip; ii++) {
				this->mx(ii) = 0.5*(this->pocx(ii)+this->potx(ii));
				this->my(ii) = 0.5*(this->pocy(ii)+this->poty(ii));

                this->delx(ii) = curveCentreX(xCoords, yCoords,
                        this->radii, this->deltas, this->mx, this->my, ii);
                this->dely(ii) = curveCentreY(xCoords, yCoords,
                        this->radii, this->deltas, this->mx, this->my, ii);

				if (this->radii(ii) < this->radiiReq(ii)) {
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

    double delta = acos(((xCoords(ii) - xCoords(ii-1)) *
            (xCoords(ii+1) - xCoords(ii)) + (yCoords(ii)
            - yCoords(ii-1)) * (yCoords(ii+1)	- yCoords(ii))) /
            (sqrt(pow((xCoords(ii) - xCoords(ii-1)), 2)
            + pow((yCoords(ii) - yCoords(ii-1)), 2))
            * sqrt(pow((xCoords(ii+1) - xCoords(ii)), 2)
            + pow((yCoords(ii+1) - yCoords(ii)), 2))));

	if (abs(delta) < 1e-4) {
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

    if (xCoords(ii)==xCoords(ii+1)) {
        return (revPOCX - xCoords(ii+1)) * tanlen / (tan(delta(ii)/2) *
                (xCoords(ii)-xCoords(ii+1)));
	} else {
        return (revPOCY - yCoords(ii+1)) * tanlen / (tan(delta(ii)/2) *
                (yCoords(ii)-yCoords(ii+1)));
	}
}

double HorizontalAlignment::radNewPOT(const Eigen::VectorXd& xCoords,
        const Eigen::VectorXd& yCoords, const Eigen::VectorXd& delta, int ii,
		double tanlen, double revPOTX, double revPOTY) {

    if (xCoords(ii)==xCoords(ii+1)) {
        return (revPOTX - xCoords(ii+1)) * tanlen / (tan(delta(ii)/2) *
                (xCoords(ii+2)-xCoords(ii+1)));
	} else {
        return (revPOTY - yCoords(ii+1)) * tanlen / (tan(delta(ii)/2) *
                (yCoords(ii+2)-yCoords(ii+1)));
	}
}

double HorizontalAlignment::curveCentreX(const Eigen::VectorXd &xCoords,
        const Eigen::VectorXd &yCoords, const Eigen::VectorXd &rad,
        const Eigen::VectorXd &delta, const Eigen::VectorXd &mx,
        const Eigen::VectorXd &my, int ii) {

    return xCoords(ii+1) + (rad(ii))*(1/cos((delta(ii))/2))*(mx(ii)
            -xCoords(ii+1))/sqrt(pow((mx(ii)-xCoords(ii+1)),2)
            +pow((my(ii)-yCoords(ii+1)),2));
}

double HorizontalAlignment::curveCentreY(const Eigen::VectorXd& xCoords,
        const Eigen::VectorXd& yCoords, const Eigen::VectorXd& rad,
        const Eigen::VectorXd& delta, const Eigen::VectorXd& mx,
        const Eigen::VectorXd& my, int ii) {

    return yCoords(ii+1) + (rad(ii))*(1/cos((delta(ii))/2))*(my(ii)
            -yCoords(ii+1))/sqrt(pow((mx(ii)-xCoords(ii+1)),2)
            +pow((my(ii)-yCoords(ii+1)),2));
}
