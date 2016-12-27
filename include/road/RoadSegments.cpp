#include "../transportbase.h"
#include <gnuplot-iostream-master/gnuplot-iostream.h>

RoadSegments::RoadSegments(RoadPtr road) {
	this->road = road;
}

RoadSegments::~RoadSegments() {
}

void RoadSegments::computeSegments() {
    unsigned int noPoints = 1;
    RoadPtr roadPtrShared = this->road.lock();

	// Set up short names
    unsigned int ip = roadPtrShared->getHorizontalAlignment()->getPOTX().size();
    double segLen = roadPtrShared->getOptimiser()->getDesignParameters()
            ->getSegmentLength();
    double vMax = roadPtrShared->getOptimiser()->getDesignParameters()
            ->getDesignVelocity();

    double startX = roadPtrShared->getOptimiser()->getDesignParameters()
            ->getStartX();
    double startY = roadPtrShared->getOptimiser()->getDesignParameters()
            ->getStartY();
    double endX = roadPtrShared->getOptimiser()->getDesignParameters()
            ->getEndX();
    double endY = roadPtrShared->getOptimiser()->getDesignParameters()
            ->getEndY();
    double endZ = roadPtrShared->getOptimiser()->getDesignParameters()
            ->getEndZ();
    double width = roadPtrShared->getOptimiser()->getDesignParameters()
            ->getRoadWidth();
    const Eigen::VectorXd& pocx = (roadPtrShared->getHorizontalAlignment()->getPOCX());
    const Eigen::VectorXd& pocy = (roadPtrShared->getHorizontalAlignment()->getPOCY());
    const Eigen::VectorXd& potx = (roadPtrShared->getHorizontalAlignment()->getPOTX());
    const Eigen::VectorXd& poty = (roadPtrShared->getHorizontalAlignment()->getPOTY());
    const Eigen::VectorXd& radii = (roadPtrShared->getHorizontalAlignment()->getRadii());
    const Eigen::VectorXd& delta = (roadPtrShared->getHorizontalAlignment()->getDeltas());
    const Eigen::VectorXd& delx = (roadPtrShared->getHorizontalAlignment()->getDelX());
    const Eigen::VectorXd& dely = (roadPtrShared->getHorizontalAlignment()->getDelY());
    const Eigen::VectorXd& pvc = (roadPtrShared->getVerticalAlignment()->getPVCs());
    const Eigen::VectorXd& pvt = (roadPtrShared->getVerticalAlignment()->getPVTs());
    const Eigen::VectorXd& epvt = (roadPtrShared->getVerticalAlignment()->getEPVTs());
    const Eigen::VectorXd& gr = (roadPtrShared->getVerticalAlignment()->getGrades());
    const Eigen::MatrixXd& a = (roadPtrShared->getVerticalAlignment()->getPolyCoeffs());

    Eigen::VectorXd thetaS(ip);
    Eigen::VectorXd thetaE(ip);
    std::vector<unsigned int> curveN(ip);
    std::vector<unsigned int> straightN(ip+1);

    straightN[0] = (unsigned int)std::ceil(sqrt(pow(pocx(0)-startX,2)
            + pow(pocy(0)-startY,2))/segLen);
    noPoints += straightN[0];

    // Compute the number of segments
    for (unsigned int ii = 0; ii < ip; ii++) {
        thetaS(ii) = atan((pocy(ii) - dely(ii))/(pocx(ii)
                - delx(ii)));

        if ((((pocy(ii) - dely(ii)) >= 0) && (thetaS(ii) < 0)) ||
                (((pocy(ii) - dely(ii)) <= 0) && (thetaS(ii) > 0))) {
            thetaS(ii) += M_PI;
        } else if (thetaS(ii) < 0) {
            thetaS(ii) += 2*M_PI;
        }

        thetaE(ii) = atan((poty(ii) - dely(ii))/(potx(ii)
                - delx(ii)));
        if ((((poty(ii) - dely(ii)) >= 0) && (thetaE(ii) < 0)) ||
                (((poty(ii) - dely(ii)) <= 0) && (thetaE(ii) > 0))) {
            thetaE(ii) += M_PI;
        } else if (thetaE(ii) < 0) {
            thetaE(ii) += 2*M_PI;
        }

        curveN[ii] = (unsigned int)std::max(ceil(delta(ii)*180/M_PI),
                ceil(delta(ii)*radii(ii)/segLen));

        if (ii > 0) {
            straightN[ii] = (unsigned int)ceil(sqrt(pow(pocx(ii)
                -potx(ii-1),2) + pow(pocy(ii)-poty(ii-1),2))/segLen);
            noPoints += (curveN[ii] + straightN[ii]);
        } else {
            noPoints += curveN[ii];
        }
    }

    // Now add the final straight segment
    straightN[ip] = (unsigned int)floor(sqrt(pow(endX - potx(ip-1),2)
            + pow(endY - poty(ip-1),2))/segLen);
    noPoints += straightN[ip];

    // Resize parametrised vectors
    this->x.resize(noPoints);
    this->y.resize(noPoints);
    this->z.resize(noPoints);
    this->s.resize(noPoints);
    this->w.resize(noPoints-1);
    this->spc.resize(2*(ip+1));
    this->v.resize(noPoints);
    this->v = Eigen::VectorXd::Constant(noPoints,vMax);
    this->typ = Eigen::VectorXi::Constant(noPoints-1,(int)(RoadSegments::ROAD));

    this->x(0) = startX;
    this->y(0) = startY;
    this->s(0) = 0;
    this->spc(0) = 0;

    Eigen::VectorXd span = Eigen::VectorXd::LinSpaced(straightN[0],
            1,straightN[0]);

    this->x.segment(1,straightN[0]) = this->x(0)
            + (pocx(0)-this->x(0))*span.array()/straightN[0];
    this->y.segment(1,straightN[0]) = this->y(0)
            + (pocy(0)-this->y(0))*span.array()/straightN[0];
    this->s.segment(1,straightN[0]) = this->s(0)
            + sqrt(pow(pocx(0)-this->x(0),2)
            + pow(pocy(0)-this->y(0),2))*span.array()/straightN[0];

    unsigned int counter = straightN[0] + 1;
    this->spc(1) = s(counter-1);

    for (unsigned int ii = 0; ii < ip; ii++) {
        // Straight segments first
        if (ii > 0) {
            span = Eigen::VectorXd::LinSpaced(straightN[ii],1,straightN[ii]);
            this->x.segment(counter,straightN[ii]) = potx(ii-1)
                    + span.array() * (pocx(ii)-potx(ii-1))/((double)straightN[ii]);
            this->y.segment(counter,straightN[ii]) = poty(ii-1)
                    + span.array() * (pocy(ii)-poty(ii-1))/((double)straightN[ii]);
            this->s.segment(counter,straightN[ii]) = this->s(counter-1)
                    + span.array() * sqrt(pow(pocx(ii)-potx(ii-1),2)
                    + pow(pocy(ii)-poty(ii-1),2))/((double)straightN[ii]);

            counter += straightN[ii];
            this->spc((ii+1)*2-1) = s(counter-1);
        }

        // Next do the curved segments
        // If the angle between tangents is not zero
        if (delta(ii) != 0.0) {
            if ((thetaE(ii) - thetaS(ii)) > M_PI) {
                thetaE(ii) = thetaE(ii) - 2*M_PI;
            } else if (thetaS(ii) - thetaE(ii) > M_PI) {
                thetaS(ii) = thetaS(ii) - 2*M_PI;
            }

            span = Eigen::VectorXd::LinSpaced(curveN[ii],1,curveN[ii]);
            double thetaDiff = (thetaE(ii) - thetaS(ii))/((double)curveN[ii]);
            Eigen::VectorXd theta = span.array() * thetaDiff + thetaS(ii);

            this->x.segment(counter,curveN[ii]) = delx(ii) + radii(ii) *
                    theta.array().cos();
            this->y.segment(counter,curveN[ii]) = dely(ii) + radii(ii) *
                    theta.array().sin();
            this->s.segment(counter,curveN[ii]) = this->s(counter-1)
                    + radii(ii)*(theta.array()-thetaS(ii)).abs();
        } else {
            this->x(counter) = pocx(ii);
            this->y(counter) = pocy(ii);
            this->s(counter) = this->s(counter-1) + sqrt(pow(pocx(ii)
                -this->x(counter-1),2) + pow(pocy(ii)
                -this->y(counter-1),2));
        }

        counter += curveN[ii];
        this->spc((ii+1)*2) = s(counter-1);
    }
	
    // Add final straight as well as the final point
    // Final segment
    span = Eigen::VectorXd::LinSpaced(straightN[ip],1,straightN[ip]);

    this->x.segment(counter,straightN[ip]) = potx(ip-1)
            + span.array() * (endX-potx(ip-1))/((double)straightN[ip]);
    this->y.segment(counter,straightN[ip]) = poty(ip-1)
            + span.array() * (endY-poty(ip-1))/((double)straightN[ip]);
    this->s.segment(counter,straightN[ip]) = this->s(counter-1)
            + span.array() * sqrt(pow(endX - potx(ip-1),2)
            + pow(endY - poty(ip-1),2))/((double)straightN[ip]);
    counter += straightN[ip];
    this->spc(ip*2+1) = s(counter-1);

    // Final point
    this->x(this->x.size()-1) = endX;
    this->y(this->y.size()-1) = endY;
    this->s(this->s.size()-1) = (roadPtrShared->getVerticalAlignment()->
            getSDistances()(roadPtrShared->getVerticalAlignment()->
            getSDistances().size()-1));
    // Compute the velocities of each of the segments based on the horizontal
    // alignment maximum speeds
    const Eigen::VectorXd& velv = roadPtrShared->getVerticalAlignment()->
            getVelocities();

    // Ranges counter
    unsigned int r1 = 0;
    unsigned int r2 = 1;

    for (unsigned int ii = 0; ii < noPoints; ii++) {
        bool inSegment = false;

        // Find the horizontal curve segment to which this value belongs. We
        // do not consider prior segments as s is monotonically increasing.
        while (!inSegment && (r2 <= ip*2+1)) {
            if ((this->s(ii) >= this->spc(r1)) &&
                    (this->s(ii) <= this->spc(r2))) {
                inSegment = true;

                // If r1 is 0 or even, we are in a tangent segment. Otherwise
                // we need to adjust the velocity
                if ((r1 % 2) != 0) {
                    this->v(ii) = velv((r1-1)/2);
                }

            } else {
                r1++;
                r2++;
            }
        }
    }

    // Now compute the elevations of each of the x and y values based on their
    // corresponding s values and adjust for the velocity during these
    // transitions if required

    Eigen::VectorXd ranges = Eigen::VectorXd::Zero(2*(pvc.size())+2);
    ranges(0) = 0;

    for (int ii = 0; ii < pvc.size(); ii++) {
        ranges(2*(ii+1)-1) = pvc(ii);
        ranges(2*(ii+1)) = pvt(ii);
    }

    ranges(ranges.size()-1) = this->s(noPoints-1);

    // Ranges counter
    r1 = 0;
    r2 = 1;

    for (unsigned int ii = 0; ii < noPoints; ii++) {
        bool inSegment = false;

        // We initially set a uniform road width for the entire length
        if (ii < (noPoints-1)) {
            this->w(ii) = width;
        }

        // Find the curve segment to which this value belongs. We do not
        // consider prior segments as s is monotonically increasing.
        while(!inSegment && (r2 <= ip*2+1)) {
            if (this->s(ii) >= ranges(r1) &&
                    (this->s(ii) <= ranges(r2))) {
                inSegment = true;

                // If r1 is 0 or even, we are in a tangent segment
                if ((r1 % 2) == 0) {
                    if (r1 == 0) {
                        this->z(ii) = z(0) + (this->s(ii) - this->s(0))
                                * gr(0) / 100.0;
                    } else {
                        this->z(ii) = epvt(r1/2-1) + (this->s(ii)
                                - pvt(r1/2-1))*gr(r1/2)/100.0;
                    }

                } else {
                    this->z(ii) = a(r2/2-1,0) + a(r2/2-1,1)*(
                            this->s(ii) - pvc(r2/2-1)) + a(r2/2-1,2) *
                            pow(this->s(ii)-pvc(r2/2-1),2);
                }

            } else {
                r1++;
                r2++;
            }
        }
    }

    this->z(noPoints-1) = endZ;
    this->computeRoadLength();
}

void RoadSegments::placeNetwork() {

    RoadPtr roadPtrShared = this->road.lock();

    RegionPtr region = this->road.lock()->getOptimiser()->getRegion();

    region->placeNetwork(this->x, this->y, this->e);
}

void RoadSegments::computeRoadLength() {
    Eigen::VectorXd xe = this->x.segment(1,this->x.size() - 1);
    Eigen::VectorXd xs = this->x.segment(0,this->x.size() - 1);
    Eigen::VectorXd ye = this->y.segment(1,this->y.size() - 1);
    Eigen::VectorXd ys = this->y.segment(0,this->y.size() - 1);
    Eigen::VectorXd ze = this->z.segment(1,this->z.size() - 1);
    Eigen::VectorXd zs = this->z.segment(0,this->z.size() - 1);

    double length = (((xe-xs).array().pow(2) + (ye-ys).array().pow(2)
            + (ze-zs).array().pow(2)).sqrt()).sum();
    this->getRoad()->getAttributes()->setLength(length);
}
