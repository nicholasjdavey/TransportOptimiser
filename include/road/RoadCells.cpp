#include "../transportbase.h"

RoadCells::RoadCells(RoadPtr road) {
	this->road = road;
}

RoadCells::~RoadCells() {
}

void RoadCells::computeRoadCells() {

    RoadPtr roadPtrShared = this->road.lock();

	// Create short names for all important inputs
    const Eigen::MatrixXd& X = roadPtrShared->getOptimiser()->getRegion()->getX();
    const Eigen::MatrixXd& Y = roadPtrShared->getOptimiser()->getRegion()->getY();
    const Eigen::VectorXd& x = roadPtrShared->getRoadSegments()->getX();
    const Eigen::VectorXd& y = roadPtrShared->getRoadSegments()->getY();
    const Eigen::VectorXd& z = roadPtrShared->getRoadSegments()->getZ();
    const Eigen::VectorXd& w = roadPtrShared->getRoadSegments()->getWidths();
    const Eigen::VectorXi& typ = roadPtrShared->getRoadSegments()->getType();

	// First need to see if the grid is evenly-spaced
    unsigned int rx = X.rows();
    unsigned int cy = Y.cols();

    Eigen::VectorXd xspacing = (X.block(1,0,rx-1,1)-X.block(0,0,rx-1,1));
    Eigen::RowVectorXd yspacing = (Y.block(0,1,1,cy-1)-Y.block(0,0,1,cy-1));

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
    Eigen::VectorXi horCross = ((x.segment(1,norg)/xspan).cast<int>()
            - (x.segment(0,norg)/xspan).cast<int>());
    Eigen::VectorXi verCross = ((y.segment(1,norg)/yspan).cast<int>()
            - (y.segment(0,norg)/yspan).cast<int>());

    // Number of intersection points for reparametrised road. These
    // intersection points all lie on grid lines of the terrain grid.
    unsigned long nnew = horCross.array().abs().sum() +
        verCross.array().abs().sum() + x.size();

    // Straight line equation in x,y for each segment
    Eigen::MatrixXd sleq = Eigen::MatrixXd::Zero(norg,4);

    // Where the difference in x is non-zero (and hence we have a function
    // mapping x to a unique and defined y)
    sleq.block(0,0,norg,1) = (((y.segment(1,norg)
            - y.segment(0,norg)).array())/((x.segment(1,norg)
            - x.segment(0,norg)).array())).matrix();
    sleq.block(0,1,norg,1) = y.segment(0,norg).array()
            - x.segment(0,norg).array()*sleq.block(0,0,norg,1).array();
    // Otherwise
    sleq.block(0,2,norg,1) = (((x.segment(1,norg)
            - x.segment(0,norg)).array())/((y.segment(1,norg)
            - y.segment(0,norg)).array())).matrix();
    sleq.block(0,3,norg,1) = x.segment(0,norg).array()
            - y.segment(0,norg).array()*sleq.block(0,2,norg,1).array();
	
    // Straight line equation of variation in z for each segment
    Eigen::MatrixXd sleq2 = Eigen::MatrixXd::Zero(norg,2);
    sleq2.block(0,0,norg,1) = (((z.segment(1,norg)
            - z.segment(0,norg)).array())/(((x.segment(1,norg)
            - x.segment(0,norg)).array().pow(2) + (y.segment(1,norg)
            - y.segment(0,norg)).array().pow(2)).sqrt()));
    sleq2.block(0,1,norg,1) = z.segment(0,norg);

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

                if (horCross(ii) != 0) {
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
                    if (std::isfinite(sleq(ii,0))) {
                        pairs.block(0,1,abs(horCross(ii)),1) = sleq(ii,0)*
                                pairs.block(0,0,abs(horCross(ii)),1).array()
                                + sleq(ii,1);
                    } else if (std::isfinite(sleq(ii,2))) {
                        pairs.block(0,1,abs(horCross(ii)),1) =
                                ((pairs.block(0,0,abs(horCross(ii)),1)).
                                array())/sleq(ii,2);
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
                    if (std::isfinite(sleq2(ii,0))) {
                        pairs.block(0,2,abs(horCross(ii)),1) = sleq2(ii,1) +
                                sleq2(ii,0)*(((pairs.block(0,0,abs(horCross(ii
                                )),1).array() - x(ii)).pow(2)
                                + (pairs.block(0,1,abs(horCross(ii)),1).array()
                                - y(ii)).pow(2)).array().sqrt());
                    } else {
                        pairs.block(0,2,abs(horCross(ii)),1) =
                                Eigen::VectorXd::Constant(abs(horCross(ii)),1,
                                z(ii));
                    }
                }

                if (verCross(ii) != 0) {
                    // Compute the y values
                    if (verCross(ii) > 0) {
                        pairs.block(abs(horCross(ii)),1,abs(verCross(ii)),1) =
                                yspan*(floor(y(ii)/yspan)
                                + Eigen::VectorXd::LinSpaced(
                                abs(verCross(ii)),1,abs(verCross(ii)))
                                .array());
                    } else if (verCross(ii) < 0) {
                        if (fmod(y(ii), yspan) == 0) {
                            pairs.block(abs(horCross(ii)),1,abs(verCross(ii)),
                                    1) = yspan*(ceil(y(ii)/yspan)
                                    - Eigen::VectorXd::LinSpaced(abs(verCross(
                                    ii)),0,abs(verCross(ii))-1).array());
                        } else {
                            pairs.block(abs(horCross(ii)),1,abs(verCross(ii)),
                                    1) = yspan*(ceil(y(ii)/yspan)
                                    - Eigen::VectorXd::LinSpaced(abs(verCross(
                                    ii)),1,abs(verCross(ii))).array());
                        }
                    }

                    // Compute the corresponding x values
                    if (std::isfinite(sleq(ii,0))) {
                        pairs.block(abs(horCross(ii)),0,abs(verCross(ii)),1) =
                                ((pairs.block(0,1,abs(verCross(ii)),1)).array()
                                 -sleq(ii,1))/sleq(ii,0);
                    } else if (std::isfinite(sleq(ii,2))) {
                        pairs.block(abs(horCross(ii)),0,abs(verCross(ii)),1) =
                                sleq(ii,3) + sleq(ii,2)*pairs.block(0,1,
                                abs(verCross(ii)),1).array();
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
                    if (std::isfinite(sleq2(ii,0))) {
                        pairs.block(abs(horCross(ii)),2,abs(verCross(ii)),1) =
                                sleq2(ii,0)*((pairs.block(0,0,
                                abs(verCross(ii)),1).array() - x(ii)).pow(2)
                                + (pairs.block(0,1,abs(verCross(ii)),1).array()
                                - y(ii)).pow(2)).sqrt() + sleq2(ii,1);
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

                this->x.segment(counter+1,nump) = (pairs.block(0,0,nump,1));
                this->y.segment(counter+1,nump) = (pairs.block(0,1,nump,1));
                this->z.segment(counter+1,nump) = (pairs.block(0,2,nump,1));
                this->w.segment(counter+1,nump) = Eigen::VectorXd::Constant(
                        nump,w(ii));
                this->type.segment(counter+1,nump) = Eigen::VectorXi::Constant(
                        nump,typ(ii));
            }

            counter += nump + 1;
        }
    }

    // Add the very last point if finite
    if (std::isfinite(x(x.size()-1)) && std::isfinite(y(y.size()-1))
            && std::isfinite(z(z.size()-1))) {
        this->x(counter) = x(x.size() - 1);
        this->y(counter) = y(y.size() - 1);
        this->z(counter) = z(z.size() - 1);
        this->w(counter) = w(w.size() - 1);
        this->type(counter) = typ(typ.size() - 1);
    }

    // Finally, remove any extra rows
    Eigen::MatrixXd interMat = Eigen::MatrixXd::Zero(nnew,5);
    Eigen::MatrixXd tempMat;
    Eigen::MatrixXd uniqueMat;
    Eigen::VectorXi uniqueRows;
    Eigen::VectorXi fullRows;
    interMat.block(0,0,nnew,1) = this->x;
    interMat.block(0,1,nnew,1) = this->y;
    interMat.block(0,2,nnew,1) = this->z;
    interMat.block(0,3,nnew,1) = this->w;
    interMat.block(0,4,nnew,1) = (this->type.cast<double>());
    igl::unique_rows(interMat,tempMat,uniqueRows,fullRows);
    Eigen::VectorXi originalOrder;
    Eigen::VectorXi originalOrderIDX;
    igl::sort(uniqueRows,1,true,originalOrder,originalOrderIDX);
    Eigen::RowVectorXi colIdx = Eigen::VectorXi::LinSpaced(5,0,4);
    igl::slice(interMat,originalOrder,colIdx,uniqueMat);
    nnew = uniqueMat.rows();
    this->x = uniqueMat.block(0,0,nnew,1);
    this->y = uniqueMat.block(0,1,nnew,1);
    this->z = uniqueMat.block(0,2,nnew,1);
    this->w = uniqueMat.block(0,3,nnew,1);
    this->type = (uniqueMat.block(0,4,nnew,1)).cast<int>();

    // Find the grid references to which the segments belong in x,y pairs
    Eigen::MatrixXd gridrefsdouble(nnew-1,2);
    Eigen::MatrixXi gridRefs;
    gridrefsdouble.block(0,0,nnew-1,1) = (0.5*(((this->x.segment(0,nnew-1)
            .array() + (this->x.segment(1,nnew-1)).array()) - xmin)/xspan));
    gridrefsdouble.block(0,1,nnew-1,1) = (0.5*(((this->y.segment(0,nnew-1)
            + (this->y.segment(1,nnew-1))).array() - ymin)/yspan));
    igl::ceil(gridrefsdouble,gridRefs);

    // We crudely compute the area of cells occupied by each segement in the
    // newly parametrised road.
    this->len = ((((this->x.segment(1,nnew-1)).array()
            - (this->x.segment(0,nnew-1)).array()).pow(2)
            + ((this->y.segment(1,nnew-1)).array()
            - (this->y.segment(0,nnew-1)).array()).pow(2)).sqrt());
    this->areas = (this->len).array()*this->w.segment(0,this->len.size())
            .array();

    // Assign the corresponding habitat to each cell reference
    const Eigen::MatrixXi& vegPtr = roadPtrShared->getOptimiser()->getRegion()
            ->getVegetation();
    Eigen::VectorXi rows = gridRefs.block(0,0,nnew-1,1);
    Eigen::VectorXi cols = gridRefs.block(0,1,nnew-1,1);
    Utility::sub2ind(vegPtr,rows,cols,this->cellRefs);

    this->veg.resize(this->areas.size());

    Utility::sliceIdx(vegPtr,this->cellRefs,this->veg);

    // Finally, determine the unique cells occupied by road. This is the same
    // as this->cellRefs but with unique values removed.
    igl::unique_rows(this->cellRefs,this->uniqueCells,uniqueRows,fullRows);
}
