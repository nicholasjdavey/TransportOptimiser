#include "../transportbase.h"

VerticalAlignment::VerticalAlignment() {
}

VerticalAlignment::VerticalAlignment(RoadPtr road) {

	unsigned long ip = road->getOptimiser()->getDesignParameters()->
			getIntersectionPoints();

	this->road = road;
	this->s.resize(ip+2);
	this->pvc.resize(ip);
	this->pvt.resize(ip);
	this->epvc.resize(ip);
	this->epvt.resize(ip);
	this->a.resize(ip,3);
	this->v.resize(ip);
	this->Ls.resize(ip);
	this->gr.resize(ip+1);
	this->ssd.resize(ip);
	Eigen::VectorXd theta(ip+1);

}

VerticalAlignment::~VerticalAlignment() {
}

void VerticalAlignment::computeAlignment() {

	// Create short names for input data
	Eigen::VectorXd* xFull = this->road->getXCoords();
	Eigen::VectorXd* yFull = this->road->getYCoords();
	Eigen::VectorXd* zFull = this->road->getZCoords();
	std::vector<bool> duplicates(xFull->size(),false);
	int uniqueEntries = 1;

	// If we have duplicate entries in our list, remove them for now and record
	// where they occur for later use
	for (int ii = 1; ii < xFull->size(); ii++) {
		if (((*xFull)(ii) == (*xFull)(ii-1)) &&
				((*yFull)(ii) == (*yFull)(ii-1))) {
			if ((*zFull)(ii) != (*zFull)(ii-1)) {
				std::cerr << "Discontinuous vertical alignment" << std::endl;
			}
			duplicates[ii] = true;
		} else {
			uniqueEntries++;
		}
	}

	Eigen::VectorXd xCoords(uniqueEntries);
	Eigen::VectorXd yCoords(uniqueEntries);
	Eigen::VectorXd zCoords(uniqueEntries);
	xCoords(0) = (*xFull)(0);
	yCoords(0) = (*yFull)(0);
	zCoords(0) = (*zFull)(0);

	for (int ii = 1; ii < xCoords.size(); ii++) {
		if (!duplicates[ii]) {
			xCoords(ii) = (*xFull)(ii);
			yCoords(ii) = (*yFull)(ii);
			zCoords(ii) = (*zFull)(ii);
		}
	}

	if (xCoords.size() != yCoords.size()) {
		std::cerr << "X and Y vectors must be of the same length" << std::endl;
	} else {
		// Short names for parameters
		double tr = this->road->getOptimiser()->getDesignParameters()
				->getReactionTime();
		double acc = this->road->getOptimiser()->getDesignParameters()
				->getDeccelRate();

		unsigned long ip = xCoords.size() - 2;
		Eigen::VectorXd* pocx = this->road->getHorizontalAlignment()
				->getPOCX();
		Eigen::VectorXd* pocy = this->road->getHorizontalAlignment()
				->getPOCX();
		Eigen::VectorXd* potx = this->road->getHorizontalAlignment()
				->getPOCX();
		Eigen::VectorXd* poty = this->road->getHorizontalAlignment()
				->getPOCX();
		Eigen::VectorXd* radii = this->road->getHorizontalAlignment()
				->getPOCX();
		Eigen::VectorXd* delta = this->road->getHorizontalAlignment()
				->getPOCX();

		this->s.resize(ip+2);

		this->s(0) = 0.0;
		this->s(1) = sqrt((pow((*pocx)(0)-xCoords(0),2) 
				+ pow((*pocy)(0)-yCoords(0),2))	+ (*delta)(0)*(*radii)(0)/2);

		for (unsigned long ii = 2; ii < ip+1; ii++) {
			this->s(ii) = this->s(ii-1) + (*delta)(ii-2)* (*radii)(ii-1)/2
					+ (*delta)(ii-1)* (*radii)(ii-1)/2 + sqrt(pow((*pocx)(ii-1)
					-(*potx)(ii-2),2) + pow((*pocy)(ii-1)-(*poty)(ii-2),2));
		}

		this->s(ip+1) = this->s(ip) + sqrt(pow(xCoords(ip+1)-(*potx)(ip-1),2) 
				+ pow(yCoords(ip+1)-(*poty)(ip-1),2))
				+ (*delta)(ip-1)*(*radii)(ip-1)/2;

		// this->s.resize(ip+2);
		this->pvc.resize(ip);
		this->pvt.resize(ip);
		this->epvc.resize(ip);
		this->epvt.resize(ip);
		this->a.resize(ip,3);
		this->v.resize(ip);
		this->Ls.resize(ip);
		this->gr.resize(ip+1);
		this->ssd.resize(ip);
		Eigen::VectorXd theta(ip+1);

		// Compute the grades for each segment
		for (unsigned int ii = 0; ii < ip; ii++) {
			this->gr(ii) = 100*(zCoords(ii+1)-zCoords(ii))/
					(this->s(ii+1)-this->s(ii));
			theta(ii) = atan(this->gr(ii)/100);
		}

		// Compute the corresponding required sight distance for each s other
		// than the start and end points (where we do not fit a parabolic curve)
		Eigen::VectorXd* vel = this->road->getHorizontalAlignment()
				->getVelocities();

		this->ssd = (vel->array()) * tr + 0.5*(vel->array().pow(2))/acc;
		this->v = *vel;

		// For each intersection point, compute the length if there is a
		// gradient change
		for (unsigned int ii = 1; ii < ip+1; ii++) {
			double A = std::abs(this->gr(ii) - this->gr(ii-1));
			
			if (theta(ii) > theta(ii-1)) {
				// Sag curve
				this->Ls(ii-1) = std::max(A*pow(this->ssd(ii-1),2)
						/(120+3.5*this->ssd(ii-1)),this->v(ii-1)*2.16);

			} else if (theta(ii) < theta(ii-1)) {
				// Crest curve
				this->Ls(ii-1) = std::max(A*pow(this->ssd(ii-1),2)/658,
						this->v(ii-1)*2.16);

			} else {
				// No curve
				this->Ls(ii-1) = 0;
			}
		}

		// First ensure that the end arcs fit

		// Start
		if (this->s(0) > (this->s(1) - this->Ls(0)/2)) {
			this->Ls(0) = 2*(this->s(1) - this->s(0));
		}

		// End
		if (this->s(ip+1) < (this->s(ip) + this->Ls(ip-1)/2)) {
			this->Ls(ip-1) = 2*(this->s(ip+1) - this->s(ip));
		}

		Eigen::VectorXd Lreq(ip);
		Lreq = this->Ls;

		// Now go through each computed arc and determine if it fits in the
		// tangents. If not, we must reduce the length to fit.
		for (unsigned int ii = 1; ii < ip; ii++) {
			double pLen = this->Ls(ii-1)/2;
			double cLen = this->Ls(ii)/2;
			double interLen = this->s(ii+1) - this->s(ii);

			// Adjust the length
			if (interLen < (pLen + cLen)) {
				if ((pLen > 0.5*interLen) && (cLen > 0.5*interLen)) {
					// Make both ends meet at the midpoint as a compromise
					this->Ls(ii-1) = interLen;
					this->Ls(ii) = Ls(ii-1);
				} else if (pLen > 0.5*interLen) {
					// Reduce the length of the previous curve to meet the
					// start of the current one
					this->Ls(ii-1) = 2*(interLen-this->Ls(ii)/2);
				} else {
					// Reduce the length of the current curve to meet the end
					// of the previous one
					this->Ls(ii) = 2*(interLen-this->Ls(ii-1)/2);
				}
			}
		}

		// See if we can increase the curvature lengths for any PIs where
		// needed
		for (unsigned int ii = 1; ii < ip-1; ii++) {
			if (this->Ls(ii) < Lreq(ii)) {
				double tan1 = this->s(ii+1) - this->s(ii);
				double tan2 = this->s(ii+2) - this->s(ii+1);

				if ((tan1 > (this->Ls(ii-1)/2+this->Ls(ii)/2)) &&
						(tan2 > (this->Ls(ii)/2+this->Ls(ii+1)/2))) {
					// We have room to expand the length of curvature
					double L1 = 2*(tan1 - this->Ls(ii-1)/2);
					double L2 = 2*(tan2 - this->Ls(ii+1)/2);
					this->Ls(ii) = std::min(std::min(L1,L2),Lreq(ii));
				}
			}
		}

		// Do the same for the start and end curves

		// Start curve
		if (this->Ls(0) < Lreq(1)) {
			double tan1 = this->s(1) - this->s(0);
			double tan2 = this->s(2) - this->s(1);

			if ((tan1 > this->Ls(0)/2) && 
					(tan2 > (this->Ls(0)/2+this->Ls(1)/2))) {
				// We can expand the length of curvature
				double L1 = tan1*2;
				double L2 = 2*(tan2 - this->Ls(1)/2);
				this->Ls(0) = std::min(std::min(L1,L2),Lreq(0));
			}
		}

		// End curve
		if (this->Ls(ip-1) < Lreq(ip-1)) {
			double tan1 = this->s(ip) - this->s(ip-1);
			double tan2 = this->s(ip+1) - this->s(ip);

			if ((tan2 > this->Ls(ip-1)/2) &&
					(tan1 > (this->Ls(ip-2)/2 + this->Ls(ip-1)/2))) {
				// We can expand the length of curvature
				double L1 = 2*(tan1 - this->Ls(ip-2));
				double L2 = 2*tan2;
				this->Ls(ip-1) = std::min(std::min(L1,L2),Lreq(ip-1));
			}
		}

		// Finally, we must compute the points of curvature, tangency and the
		// equations of the parabolas as well as the corresponding velocity
		for (unsigned int ii = 0; ii < ip; ii++) {
			double A = std::abs(this->gr(ii+1) - this->gr(ii));
			double theta1 = theta(ii);
			double theta2 = theta(ii+1);

			if (A > 0) {
				// Curvature and tangent points
				this->pvc(ii) = this->s(ii+1) - this->Ls(ii)/2;
				this->pvt(ii) = this->s(ii+1) + this->Ls(ii)/2;
				this->epvc(ii) = zCoords(ii+1) 
						- this->gr(ii) * this->Ls(ii)/200;
				this->epvt(ii) = zCoords(ii+1)
						+ this->gr(ii+1) * this->Ls(ii)/200;

				// Parabola coefficients
				this->a(ii,0) = this->epvc(ii);
				this->a(ii,1) = this->gr(ii)/100;
				this->a(ii,2) = (this->gr(ii+1) - this->gr(ii)) /
						(200*this->Ls(ii));

			} else {
				this->pvc(ii) = this->s(ii+1);
				this->pvt(ii) = this->s(ii+1);
				this->epvc(ii) = zCoords(ii+1);
				this->epvt(ii) = zCoords(ii+1);

				// Parabola coefficients
				this->a(ii,0) = this->epvc(ii);
				this->a(ii,1) = this->gr(ii);
				this->a(ii,2) = 0.0;
			}

			// Velocities adjustment
			double v1 = this->Ls(ii)/2.16;
			double sd = v1*tr + 0.5*pow(v1,2)/acc;

			if (theta2 > theta1) {
				double L1 = A*pow(sd,2)/(120+3.5*sd);

				if (L1 < this->Ls(ii)) {
					this->ssd(ii) = sd;
					this->v(ii) = v1;
				} else {
					this->ssd(ii) = (3.5*this->Ls(ii)
							+ sqrt(pow(3.5*this->Ls(ii),2)
							+ 480*A*this->Ls(ii)))/(2*A);
					this->v(ii) = (-tr + sqrt(pow(tr,2) 
							+ 2*this->ssd(ii)/acc))*acc;
				}

			} else if (theta2 < theta1) {
				double L1 = A*pow(sd,2)/658;

				if (L1 < this->Ls(ii)) {
					this->ssd(ii) = sd;
					this->v(ii) = v1;
				} else {
					this->ssd(ii) = sqrt(this->Ls(ii)*658/A);
					this->v(ii) = (-tr+sqrt(pow(tr,2) 
							+ 2*this->ssd(ii)/acc))*acc;
				}

			} else {
				this->v(ii) = (*vel)(ii);
			}
		}
	}
}
