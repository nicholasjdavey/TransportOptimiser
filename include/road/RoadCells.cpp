#include "../transportbase.h"

RoadCells::RoadCells(RoadPtr road) {
	this->road = road;
}

RoadCells::~RoadCells() {
}

void RoadCells::computeRoadCells() {

	// Create short names for all important inputs
    const Eigen::MatrixXd& X = this->road->getOptimiser()->getRegion()->getX();
    const Eigen::MatrixXd& Y = this->road->getOptimiser()->getRegion()->getY();
    const Eigen::VectorXd& x = this->road->getRoadSegments()->getX();
    const Eigen::VectorXd& y = this->road->getRoadSegments()->getY();
    const Eigen::VectorXd& z = this->road->getRoadSegments()->getZ();
    const Eigen::VectorXd& w = this->road->getRoadSegments()->getWidths();
    const Eigen::VectorXi& typ = this->road->getRoadSegments()->getType();

	// First need to see if the grid is evenly-spaced
    unsigned int rx = X.rows();
    unsigned int cy = Y.cols();

    Eigen::VectorXd xspacing = (X.block(1,0,rx-1,1)-X.block(0,0,rx-1,1));
    Eigen::VectorXd yspacing = (Y.block(0,1,0,cy-1)-Y.block(0,0,1,cy-1));

	if (((xspacing.block(1,0,rx-2,1)-xspacing.block(0,0,rx-2,1)).sum()
			> 1e-4) || ((yspacing.block(0,1,1,cy-2)-
			yspacing.block(0,0,1,cy-2)).sum()) > 1e-4) {
		std::cerr << "The input terrain must be an evenly-spaced grid."
				<< std::endl;
	}

	// X and Y must be monotonically increasing. That is, they must be ordered
	// such that they can directly plot the region
    double xmin = X(0,0);
    double ymin = Y(0,0);

	// Compute road cells
    unsigned int norg = x.size() - 1;

	// Compute the span of each grid cell. We assume all cells have the same
	// span.
    double xspan = X(1,0) - X(0,0);
    double yspan = Y(0,1) - Y(0,0);

	// Compute the number of horizontal crosses between each consecutive pair
	// of road points
    Eigen::VectorXi horCross = ((x.segment(1,norg)/xspan)
            - (x.segment(0,norg)/xspan)).cast<int>();
    Eigen::VectorXi verCross = ((y.segment(1,norg)/yspan)
            - (y.segment(0,norg)/yspan)).cast<int>();

	// Number of intersection points for reparametrised road. These
	// intersection points all lie on grid lines of the terrain grid.
	unsigned long nnew = horCross.array().abs().sum() + 
            verCross.array().abs().sum() + x.size();

	// Straight line equation in x,y for each segment
	Eigen::MatrixXd sleq = Eigen::MatrixXd::Zero(4,norg);

	// Where the difference in x is non-zero (and hence we have a function
	// mapping x to a unique and defined y)
    sleq.block(0,0,1,norg) = (((y.segment(1,norg)
            - y.segment(0,norg)).array())/((x.segment(1,norg)
            - x.segment(0,norg)).array())).matrix();
    sleq.block(1,0,1,norg) = (y.segment(0,norg)
            - x.segment(0,norg)).array()*sleq.block(0,0,1,norg).array();
	// Otherwise
    sleq.block(2,0,1,norg) = (((x.segment(1,norg)
            - x.segment(0,norg)).array())/((y.segment(1,norg)
            - y.segment(0,norg)).array())).matrix();
    sleq.block(3,0,1,norg) = (x.segment(0,norg)
            - y.segment(0,norg)).array()*sleq.block(0,0,1,norg).array();
	
	// Straight line equation of variation in z for each segment
	Eigen::MatrixXd sleq2 = Eigen::MatrixXd::Zero(2,norg);
    sleq2.block(0,0,1,norg) = (((z.segment(1,norg)
            - z.segment(0,norg)).array())/(((x.segment(1,norg)
            - x.segment(0,norg)).array().pow(2) + (y.segment(1,norg)
            - y.segment(0,norg)).array().pow(2)).sqrt()));
    sleq2.block(1,0,1,norg) = z.segment(0,norg);

	// Note, here if the distance from (x1,y1) to (x2,y2) is zero, then we just
	// say that we are at the same z value because it is the same point.
	// Therefore, we do not need another set of linear equations here.

	// For each original span, we find the intervening points and add these to
	// the list of points for the road.
    this->x = Eigen::VectorXd::Zero(nnew);
    this->y = Eigen::VectorXd::Zero(nnew);
    this->z = Eigen::VectorXd::Zero(nnew);
    this->w = Eigen::VectorXd::Zero(nnew);
    this->type = Eigen::VectorXi::Zero(nnew);

	unsigned long counter = 0;

	for (unsigned int ii = 0; ii < norg; ii++) {
		// First make sure that the current x,y and z values are finite
        if (std::isfinite(x(ii)) && std::isfinite(y(ii)) && std::isfinite(z(ii))) {

			// Put the existing points in the matrix
            this->x(counter) = x(ii);
            this->y(counter) = y(ii);
            this->z(counter) = z(ii);
            this->w(counter) = w(ii);
            this->type(counter) = typ(ii);
			unsigned int nump = 0;

			// Find the x and y coordinates of every intervening intersection
			// point
			if ((abs(horCross(ii)) + abs(verCross(ii))) > 0) {
				nump = abs(horCross(ii)) + abs(verCross(ii));
				Eigen::MatrixXd pairs = Eigen::MatrixXd::Zero(nump,3);

				if (horCross(ii) != 0.0) {
					// Compute the x values
					if (horCross(ii) > 0) {
						pairs.block(0,0,abs(horCross(ii)),1) = 
                                xspan*(floor(x(ii)/xspan)
								+ Eigen::VectorXd::LinSpaced(
								abs(horCross(ii)),1,abs(horCross(ii)))
								.array());
					} else if (horCross(ii) < 0) {
                        if (fmod(x(ii), xspan) == 0) {
							pairs.block(0,0,abs(horCross(ii)),1) =
                                    xspan*(ceil(x(ii)/xspan)
									- Eigen::VectorXd::LinSpaced(
									abs(horCross(ii)),0,abs(horCross(ii))-1)
									.array());
						} else {
							pairs.block(0,0,abs(horCross(ii)),1) =
                                    xspan*(ceil(x(ii)/xspan)
									- Eigen::VectorXd::LinSpaced(
									abs(horCross(ii)),1,abs(horCross(ii)))
									.array());
						}
					}

					// Compute the corresponding y values
					if (std::isfinite(sleq(0,ii))) {
						pairs.block(0,1,abs(horCross(ii)),1) = sleq(0,ii)*
								pairs.block(0,0,abs(horCross(ii)),1).array()
								+ sleq(2,ii);
					} else if (std::isfinite(sleq(2,ii))) {
						pairs.block(0,1,abs(horCross(ii)),1) =
								((pairs.block(0,0,abs(horCross(ii)),1)).
								array())/sleq(2,ii);
					} else {
						// It appears that the two points of (x1,y1) and
						// (x2,y2) are the same point. Therefore, the 
						// intervening point is also the same. This is rare but
						// we get around it by using this approach carefully.
						pairs.block(0,1,abs(horCross(ii)),1) =
								Eigen::VectorXd::Constant(abs(horCross(ii)),1,
                                y(ii));
					}

					// Compute corresponding z values
					if (std::isfinite(sleq2(0,ii))) {
						pairs.block(0,2,abs(horCross(ii)),1) = sleq2(1,ii) + 
								sleq2(0,ii)*(((pairs.block(0,0,abs(horCross(ii
                                )),1).array() - x(ii)).pow(2)
								+ (pairs.block(0,1,abs(horCross(ii)),1).array()
                                - y(ii)).pow(2)).array().sqrt());
					} else {
						pairs.block(0,2,abs(horCross(ii)),1) = 
								Eigen::VectorXd::Constant(abs(horCross(ii)),1,
                                z(ii));
					}
				}

				if (verCross(ii) != 0.0) {
					// Compute the y values
					if (verCross(ii) > 0) {
						pairs.block(abs(horCross(ii)),1,abs(verCross(ii)),1) =
                                yspan*(floor(y(ii)/yspan)
								+ Eigen::VectorXd::LinSpaced(
								abs(verCross(ii)),1,abs(horCross(ii)))
								.array());
					} else if (verCross(ii) < 0) {
                        if (fmod(y(ii), yspan) == 0) {
							pairs.block(abs(horCross(ii)),1,abs(verCross(ii)),
                                    1) = yspan*(ceil(y(ii)/yspan)
									- Eigen::VectorXd::LinSpaced(abs(verCross(
									ii)),0,abs(horCross(ii))-1).array());
						} else {
							pairs.block(abs(horCross(ii)),1,abs(verCross(ii)),
                                    1) = yspan*(ceil(y(ii)/yspan)
									- Eigen::VectorXd::LinSpaced(abs(verCross(
									ii)),1,abs(verCross(ii))).array());
						}
					}

					// Compute the corresponding x values
					if (std::isfinite(sleq(0,ii))) {
						pairs.block(abs(horCross(ii)),0,abs(verCross(ii)),1) =
								((pairs.block(abs(horCross(ii)),1,abs(
								verCross(ii)),1)).array())/sleq(0,ii);
					} else if (std::isfinite(sleq(2,ii))) {
						pairs.block(abs(horCross(ii)),0,abs(verCross(ii)),1) =
								sleq(3,ii) + sleq(2,ii)*pairs.block(abs(
								horCross(ii)),1,abs(verCross(ii)),1).array();
					} else {
						// It appears that the two points of (x1,y1) and
						// (x2,y2) are the same point. Therefore, the
						// intervening point is also the same. This is rare but
						// we get around it by using this approach carefully.
						pairs.block(abs(horCross(ii)),0,abs(verCross(ii)),1) =
								Eigen::VectorXd::Constant(verCross(ii),1,
                                x(ii));
					}

					// Compute the corresponding z values
					if (std::isfinite(sleq2(0,ii))) {
						pairs.block(abs(horCross(ii)),2,abs(verCross(ii)),1) =
								sleq2(0,ii)*((pairs.block(abs(horCross(ii)),0,
                                abs(verCross(ii)),1).array() - x(ii)).pow(2)
								+ (pairs.block(abs(horCross(ii)),1,
                                abs(verCross(ii)),1).array() - y(ii)).pow(2)
								).sqrt() + sleq(1,ii);
					} else {
						pairs.block(abs(horCross(ii)),2,abs(verCross(ii)),1) =
								Eigen::VectorXd::Constant(abs(verCross(ii)),1,
                                z(ii));
					}
				}

                if (x(ii+1) >= x(ii)) {
                    Eigen::MatrixXd temppairs = Eigen::MatrixXd::Zero(nump,3);
                    Eigen::VectorXi indices = Eigen::VectorXi::Zero(nump);
                    igl::sortrows(pairs,true,temppairs,indices);
                    pairs = temppairs;
				} else {
                    Eigen::MatrixXd temppairs = Eigen::MatrixXd::Zero(nump,3);
                    Eigen::VectorXi indices = Eigen::VectorXi::Zero(nump);
                    igl::sortrows(pairs,false,temppairs,indices);
                    pairs = temppairs;
				}

                this->x.segment(counter+1,nump) = (pairs.block(0,0,pairs.
						cols(),1)).transpose();
                this->y.segment(counter+1,nump) = (pairs.block(0,1,pairs.
						cols(),1)).transpose();
                this->z.segment(counter+1,nump) = (pairs.block(0,2,pairs.
						cols(),1)).transpose();
                this->w.segment(counter+1,nump) = Eigen::VectorXd::Constant(
                            nump,w(ii));
                this->type.segment(counter+1,nump) = Eigen::VectorXi::Constant(
                            nump,typ(ii));
			}

			counter += nump + 1;
		}
	}

	// Add the very last point if finite
    if (std::isfinite(x(x.size())) && std::isfinite(y(y.size()))
            && std::isfinite(z(z.size()))) {
        this->x(counter) = x(x.size() - 1);
        this->y(counter) = y(y.size() - 1);
        this->z(counter) = z(z.size() - 1);
        this->w(counter) = w(w.size() - 1);
        this->type(counter) = typ(x.size() - 1);
	}

	// Finally, remove any extra rows
    Eigen::MatrixXd interMat = Eigen::MatrixXd::Zero(nnew,5);
    Eigen::MatrixXd uniqueMat;
    Eigen::VectorXi uniqueRows;
    Eigen::VectorXi fullRows;
    interMat.block(0,0,nnew,1) = this->x.transpose();
    interMat.block(0,1,nnew,1) = this->y.transpose();
    interMat.block(0,2,nnew,1) = this->z.transpose();
    interMat.block(0,3,nnew,1) = this->w.transpose();
    interMat.block(0,4,nnew,1) = (this->type.cast<double>()).transpose();
    igl::unique_rows(interMat,uniqueMat,uniqueRows,fullRows);
    nnew = uniqueMat.rows();
    this->x = uniqueMat.block(0,0,nnew,1);
    this->y = uniqueMat.block(0,1,nnew,1);
    this->z = uniqueMat.block(0,2,nnew,1);
    this->w = uniqueMat.block(0,3,nnew,1);
    this->type = (uniqueMat.block(0,4,nnew,1)).cast<int>();

    // Find the grid references to which the segments belong in x,y pairs
    Eigen::MatrixXd gridrefsdouble(nnew,2);
    Eigen::MatrixXi gridRefs;
    gridrefsdouble.block(0,0,nnew,1) = (0.5*(((this->x.segment(0,nnew-1)
            + (this->x.segment(1,nnew-1))).array() - xmin)/xspan));
    gridrefsdouble.block(0,1,nnew,1) = (0.5*(((this->y.segment(0,nnew-1)
            + (this->y.segment(1,nnew-1))).array() - ymin)/yspan));
    igl::ceil(gridrefsdouble,gridRefs);

    // We crudely compute the area of cells occupied by each segement in the
    // newly parametrised road.
    this->len = ((((this->x.segment(1,nnew-1)).array()
            - (this->x.segment(0,nnew-1)).array()).pow(2)
            + ((this->y.segment(1,nnew-1)).array()
            - (this->y.segment(0,nnew-1)).array()).pow(2)).sqrt());
    this->areas = (this->len).array()*this->w.array();

    // Assign the corresponding habitat to each cell reference
    const Eigen::MatrixXi& vegPtr = this->road->getOptimiser()->getRegion()
            ->getVegetation();
    Eigen::VectorXi rows = gridRefs.block(0,0,nnew,1);
    Eigen::VectorXi cols = gridRefs.block(0,1,nnew,1);
    Utility::sub2ind(vegPtr,rows,cols,this->cellRefs);

    igl::slice(vegPtr,this->cellRefs,this->veg);

    // Finally, determine the unique cells occupied by road. This is the same
    // as this->cellRefs but with unique values removed.
    igl::unique_rows(this->cellRefs,this->uniqueCells,uniqueRows,fullRows);
}
