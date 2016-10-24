#include "../transportbase.h"

RoadGA::RoadGA() : Optimiser() {

}

RoadGA::RoadGA(const std::vector<TrafficProgramPtr>& programs, OtherInputsPtr
    oInputs, DesignParametersPtr desParams, EarthworkCostsPtr earthworks,
    UnitCostsPtr unitCosts, VariableParametersPtr varParams, const
    std::vector<SpeciesPtr>& species, EconomicPtr economic, TrafficPtr traffic,
    RegionPtr region, double mr, unsigned long cf, unsigned long gens, unsigned
    long popSize, double stopTol, double confInt, double confLvl, unsigned long
    habGridRes, std::string solScheme, unsigned long noRuns,
    Optimiser::Type type) :
    Optimiser(programs, oInputs, desParams, earthworks, unitCosts, varParams,
            species, economic, traffic, region, mr, cf, gens, popSize, stopTol,
            confInt, confLvl, habGridRes, solScheme, noRuns, type) {

}

void RoadGA::creation() {

    unsigned long individualsPerPop = this->populationSizeGA/5;

    double sx = this->designParams->getStartX();
    double sy = this->designParams->getStartY();
    double ex = this->designParams->getEndX();
    double ey = this->designParams->getEndY();

    double gmax = this->designParams->getMaxGrade();
    double eHigh = this->designParams->getMaxSE()/100;
    double velDes = this->designParams->getDesignVelocity();

    // Region limits. We ignore the two outermost cell boundaries
    double minLon = (this->getRegion()->getX())(3,1);
    double maxLon = (this->getRegion()->getX())(this->getRegion()->getX().
            rows()-2,1);
    double minLat = (this->getRegion()->getY())(1,3);
    double maxLat = (this->getRegion()->getY())(1,this->getRegion()->getY().
            cols()-2);

    if (sx > maxLon || sx < minLon || ex > maxLon || ex < minLon ||
            sy > maxLat || sy < minLat || ey > maxLat || ey < minLat) {
        std::cerr << "One or more of the end points lie outside the region of interest"
                  << std::endl;
    }

    // The genome length is all of the design points, expressed in 3
    // dimensions. As we also include the start and end points (which are also
    // in 3 dimensions)
    unsigned long intersectPts = this->designParams->getIntersectionPoints();
    unsigned long genomeLength = 3*(intersectPts+2);

    // Create population using the initial population routine devised by
    // Jong et al. (2003)

    // First compute the elevations of the start and end points and place them
    // as the start and end points of the intersection points vectors that
    // represent the population roads.
    this->currentRoadPopulation = Eigen::MatrixXd::Zero(this->populationSizeGA,
            genomeLength);

    double sz;
    double ez;

    region->placeNetwork(sx,sy,sz);
    region->placeNetwork(ex,ey,ez);

    Eigen::RowVectorXd starting(3);
    Eigen::RowVectorXd ending(3);
    starting << sx, sy, sz;
    ending << ex, ey, ez;

    this->currentRoadPopulation.block(0,0,this->populationSizeGA,3).rowwise() =
            starting;
    this->currentRoadPopulation.block(0,genomeLength-3,
            this->populationSizeGA,3).rowwise() = ending;

    Utility::cuttingPlanes(minLon, maxLon, minLat, maxLat, sx, sy, sz, ex, ey,
            ez, intersectPts, this->xO, this->yO, this->zO, this->dU, this->dL,
            this->theta);

    // Compute the five different initial populations for road design

    // POPULATION 1 ///////////////////////////////////////////////////////////
    // Intersection points lie on the straight line connecting the start and
    // end points at the origin points
    {
        Eigen::VectorXd s1Ind = Eigen::VectorXd::Zero(intersectPts*3);
        Eigen::VectorXi idx = Eigen::VectorXi::LinSpaced(intersectPts,0,
                intersectPts - 1);
        igl::slice_into(this->xO,(3*idx.array()).matrix(), s1Ind);
        igl::slice_into(this->yO,(3*idx.array() + 1).matrix(), s1Ind);
        igl::slice_into(this->zO,(3*idx.array() + 2).matrix(), s1Ind);

        this->currentRoadPopulation.block(0,3,individualsPerPop,
                3*intersectPts).rowwise() = s1Ind.transpose();
    }

    // POPULATION 2 ///////////////////////////////////////////////////////////
    // Intersection points lie randomly on the perpendicular planes with random
    // elevations.
    {
        // X and Y coordinates
        this->randomXYOnPlanes(individualsPerPop, intersectPts,
                individualsPerPop);

        // Z coordinates
        this->randomZWithinRange(individualsPerPop, intersectPts,
                individualsPerPop);
    }

    // POPULATION 3 ///////////////////////////////////////////////////////////
    // Intersection points lie randomly on the perpendicular planes with
    // elevations as close as possible to the existing ground elevations.
    {
        // X and Y coordinates
        this->randomXYOnPlanes(individualsPerPop, intersectPts,
                2*individualsPerPop);

        // Z coordinates
        this->zOnTerrain(individualsPerPop, intersectPts,
                2*individualsPerPop);
    }

    // POPULATION 4 ///////////////////////////////////////////////////////////
    // Intersection points scatter randomly within the study region with random
    // elevations.
    {
        // X and Y coordinates
        this->randomXYinRegion(individualsPerPop, intersectPts,
                3*individualsPerPop);

        // Z coordinates
        this->randomZWithinRange(individualsPerPop, intersectPts,
                3*individualsPerPop);
    }

    // POPULATION 5 ///////////////////////////////////////////////////////////
    // Intersection points scatter randomly within the study region with
    // elevations as close as possible to the existing ground elevations.
    {
        // As the number of roads in the population may not be a multiple of 5,
        // we assign all remaining roads to this last category.
        unsigned long remainingIndividuals = this->populationSizeGA -
                individualsPerPop*4;

        // X and Y coordinates
        this->randomXYinRegion(remainingIndividuals, intersectPts,
                4*individualsPerPop);

        // Z coordinates
        this->zOnTerrain(remainingIndividuals, intersectPts,
                4*individualsPerPop);
    }

    // Now check for invalid rows (NaN or Inf elements)
    Eigen::MatrixXi invalid(this->populationSizeGA,3*(intersectPts+2));
    Eigen::VectorXi rows(this->populationSizeGA);
    Eigen::VectorXi rowsg(this->populationSizeGA);

    invalid = (this->currentRoadPopulation.unaryExpr([](double v){ return
            std::isfinite(v); }).cast<int>());
    rows = ((invalid.rowwise().sum()).array() > 0).cast<int>();
    rowsg = (1 - rows.array()).matrix();
    long noBadRows = rows.count();

    // Replace bad members with good members
    Eigen::VectorXi badRows(noBadRows);
    Eigen::VectorXi goodRows(this->populationSizeGA - noBadRows);
    igl::find(rows,badRows);
    igl::find(rowsg,goodRows);

    std::random_shuffle(rowsg.data(),rowsg.data() + this->populationSizeGA -
            noBadRows);

    std::cout << "Population contains " << std::to_string(noBadRows) << " individual(s) with NaN or Inf entries. Replacing with good individuals." << std::endl;

    for (int ii = 0; ii < noBadRows; ii++) {
        this->currentRoadPopulation.row(ii) = this->currentRoadPopulation .row(
                rowsg(ii));
    }
}

void RoadGA::crossover() {

}

void RoadGA::mutation() {}

void RoadGA::optimise() {}

void RoadGA::output() {}

void RoadGA::computeSurrogate() {}

void RoadGA::randomXYOnPlanes(const long &individuals, const long&
        intersectPts, const long& startRow) {

    Eigen::MatrixXd randomMatrixX = Eigen::MatrixXd::Random(individuals,
            intersectPts);
    Eigen::MatrixXd randomMatrixY = Eigen::MatrixXd::Random(individuals,
            intersectPts);
    Eigen::MatrixXd xvals = Eigen::MatrixXd::Zero(individuals,intersectPts);
    Eigen::MatrixXd yvals = Eigen::MatrixXd::Zero(individuals,intersectPts);

    xvals = ((this->xO.array() + this->dL.array()*cos(this->theta)).rowwise().
            replicate(individuals).transpose() + (randomMatrixX.array()*
            (((this->dU - this->dL)*cos(theta)).rowwise().replicate(
            individuals).transpose().array()))).matrix();

    yvals = ((this->yO.array() + this->dL.array()*sin(this->theta)).rowwise().
            replicate(individuals).transpose() + (randomMatrixY.array()*
            (((this->dU - this->dL)*sin(theta)).rowwise().replicate(
            individuals).transpose().array()))).matrix();

    // Place the genome values in the initial population matrix
    Eigen::VectorXi rowIdx = Eigen::VectorXi::LinSpaced(intersectPts,
            startRow,startRow + intersectPts - 1);
    Eigen::RowVectorXi colIdx = Eigen::RowVectorXi::LinSpaced(intersectPts,0,
            intersectPts - 1);

    igl::slice_into(xvals,rowIdx.rowwise().replicate(intersectPts),
            colIdx.colwise().replicate(individuals),
            this->currentRoadPopulation);
    igl::slice_into(yvals,rowIdx.rowwise().replicate(intersectPts),
            (colIdx.array() + 1).colwise().replicate(individuals),
            this->currentRoadPopulation);
}

void RoadGA::randomXYinRegion(const long &individuals, const long
        &intersectPts, const long &startRow) {

    double minLon = (this->getRegion()->getX())(3,1);
    double maxLon = (this->getRegion()->getX())(this->getRegion()->getX().
            rows()-2,1);
    double minLat = (this->getRegion()->getY())(1,3);
    double maxLat = (this->getRegion()->getY())(1,this->getRegion()->getY().
            cols()-2);

    Eigen::MatrixXd randomMatrixX = Eigen::MatrixXd::Random(individuals,
            intersectPts);
    Eigen::MatrixXd randomMatrixY = Eigen::MatrixXd::Random(individuals,
            intersectPts);
    Eigen::MatrixXd xvals = Eigen::MatrixXd::Zero(individuals,intersectPts);
    Eigen::MatrixXd yvals = Eigen::MatrixXd::Zero(individuals,intersectPts);

    xvals = randomMatrixX.array()*(maxLon - minLon) + minLon;
    yvals = randomMatrixY.array()*(maxLat - minLat) + minLat;

    // Place the genome values in the initial population matrix
    Eigen::VectorXi rowIdx = Eigen::VectorXi::LinSpaced(intersectPts,
            startRow,startRow + intersectPts - 1);
    Eigen::RowVectorXi colIdx = Eigen::RowVectorXi::LinSpaced(intersectPts,0,
            intersectPts - 1);

    igl::slice_into(xvals,rowIdx.rowwise().replicate(intersectPts),
            colIdx.colwise().replicate(individuals),
            this->currentRoadPopulation);
    igl::slice_into(yvals,rowIdx.rowwise().replicate(intersectPts),
            (colIdx.array() + 1).colwise().replicate(individuals),
            this->currentRoadPopulation);
}

void RoadGA::randomZWithinRange(const long &individuals, const long&
            intersectPts, const long& startRow) {

    double gmax = this->designParams->getMaxGrade();

    Eigen::MatrixXd sTemp = Eigen::MatrixXd::Zero(individuals,
            intersectPts + 2);

    for (long ii = 0; ii < individuals; ii++) {
        Eigen::VectorXd xtemp(intersectPts + 2);
        Eigen::VectorXd ytemp(intersectPts + 2);

        Eigen::VectorXi row = Eigen::VectorXi::Constant(intersectPts
                + 2, ii);
        Eigen::VectorXi colIdx = Eigen::VectorXi::LinSpaced(intersectPts,0,
                intersectPts - 1);
        Eigen::VectorXi col = (3*colIdx.array() + 1).transpose();

        igl::slice(this->currentRoadPopulation,row,col,xtemp);
        igl::slice(this->currentRoadPopulation,row,col,ytemp);

        RoadPtr road(new Road(this->me(),xtemp,ytemp,
                Eigen::VectorXd::Zero(intersectPts)));
        road->computeAlignment();

        sTemp.row(ii) = road->getVerticalAlignment()->getSDistances();
    }

    for (long ii = 0; ii < intersectPts; ii++) {
        Eigen::VectorXd randVec = Eigen::VectorXd::Random(individuals);

        Eigen::VectorXd zL = (this->currentRoadPopulation.block(startRow,
                3*ii+2,individuals,1) - (sTemp.col(ii+1) - sTemp.col(ii))*
                gmax/100).array().max((this->currentRoadPopulation.block(
                startRow,3*(intersectPts + 2) - 1,individuals,1) -
                (sTemp.col(intersectPts + 2) - sTemp.col(ii+1))*gmax/100).
                array()).matrix();

        Eigen::VectorXd zU = (this->currentRoadPopulation.block(startRow,
                3*ii+2,individuals,1) - (sTemp.col(ii+1) + sTemp.col(ii))*
                gmax/100).array().min((this->currentRoadPopulation.block(
                startRow,3*(intersectPts + 2) - 1,individuals,1) - (sTemp.
                col(intersectPts + 2) + sTemp.col(ii+1))*gmax/100).array()).
                matrix();

        this->currentRoadPopulation.block(startRow,3*(ii+1)+2,individuals,1) =
                ((randVec.array())*(zU-zL).array() + zL.array()).matrix();
    }
}

void RoadGA::zOnTerrain(const long &individuals, const long&
            intersectPts, const long& startRow) {

    double gmax = this->designParams->getMaxGrade();

    Eigen::MatrixXd sTemp = Eigen::MatrixXd::Zero(individuals,
            intersectPts + 2);

    for (long ii = 0; ii < individuals; ii++) {
        Eigen::VectorXd xtemp(intersectPts + 2);
        Eigen::VectorXd ytemp(intersectPts + 2);

        Eigen::VectorXi row = Eigen::VectorXi::Constant(intersectPts
                + 2, ii);
        Eigen::VectorXi colIdx = Eigen::VectorXi::LinSpaced(intersectPts,0,
                intersectPts - 1);
        Eigen::VectorXi col = (3*colIdx.array() + 1).transpose();

        igl::slice(this->currentRoadPopulation,row,col,xtemp);
        igl::slice(this->currentRoadPopulation,row,col,ytemp);

        RoadPtr road(new Road(this->me(),xtemp,ytemp,
                Eigen::VectorXd::Zero(intersectPts)));
        road->computeAlignment();

        sTemp.row(ii) = road->getVerticalAlignment()->getSDistances();
    }

    for (long ii = 0; ii < intersectPts; ii++) {
        Eigen::VectorXd randVec = Eigen::VectorXd::Random(individuals);

        Eigen::VectorXd zL = (this->currentRoadPopulation.block(startRow,
                3*ii+2,individuals,1) - (sTemp.col(ii+1) - sTemp.col(ii))*
                gmax/100).array().max((this->currentRoadPopulation.block(
                startRow,3*(intersectPts + 2) - 1,individuals,1) -
                (sTemp.col(intersectPts + 2) - sTemp.col(ii+1))*gmax/100).
                array()).matrix();

        Eigen::VectorXd zU = (this->currentRoadPopulation.block(startRow,
                3*ii+2,individuals,1) - (sTemp.col(ii+1) + sTemp.col(ii))*
                gmax/100).array().min((this->currentRoadPopulation.block(
                startRow,3*(intersectPts + 2) - 1,individuals,1) - (sTemp.
                col(intersectPts + 2) + sTemp.col(ii+1))*gmax/100).array()).
                matrix();

        Eigen::VectorXd zE(individuals);

        this->region->placeNetwork(this->currentRoadPopulation.block(
                startRow,3*(ii+1), individuals,1), this->currentRoadPopulation.
                block(startRow,3*(ii+1)+1, individuals,1), zE);

        Eigen::VectorXd selectGround = ((zE.array() <=
                zU.array()) && (zE.array() >= zL.array())).cast<double>();
        Eigen::VectorXd selectLower = (zE.array() < zL.array()).cast<double>();
        Eigen::VectorXd selectUpper = (zE.array() > zU.array()).cast<double>();

        this->currentRoadPopulation.block(startRow,3*(ii+1)+2,individuals,1) =
                (zE.array() * selectGround.array() + zL.array() * selectLower
                .array() + zU.array() * selectUpper.array()).matrix();
    }
}
