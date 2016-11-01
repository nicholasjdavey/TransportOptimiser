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
    Optimiser::Type type, double elite, double scale, unsigned long learnPeriod,
    double surrThresh, unsigned long maxLearnNo, unsigned long minLearnNo) :
    Optimiser(programs, oInputs, desParams, earthworks, unitCosts, varParams,
            species, economic, traffic, region, mr, cf, gens, popSize, stopTol,
            confInt, confLvl, habGridRes, solScheme, noRuns, type, elite) {

    this->theta = 0;
    this->generation = 0;
    this->scale = scale;
    this->xO = Eigen::VectorXd::Zero(this->designParams->getIntersectionPoints());
    this->yO = Eigen::VectorXd::Zero(this->designParams->getIntersectionPoints());
    this->zO = Eigen::VectorXd::Zero(this->designParams->getIntersectionPoints());
    this->dU = Eigen::VectorXd::Zero(this->designParams->getIntersectionPoints());
    this->dL = Eigen::VectorXd::Zero(this->designParams->getIntersectionPoints());

    int noSpecies = this->species.size();
    this->costs = Eigen::VectorXd::Zero(this->populationSizeGA);
    this->profits = Eigen::VectorXd::Zero(this->populationSizeGA);
    this->iarsCurr = Eigen::MatrixXd::Zero(this->populationSizeGA,noSpecies);
    this->popsCurr = Eigen::MatrixXd::Zero(this->populationSizeGA,noSpecies);
    this->useCurr = Eigen::VectorXd::Zero(this->populationSizeGA);

    this->best = Eigen::VectorXd::Zero(this->generations);
    this->av = Eigen::VectorXd::Zero(this->generations);

    this->iars = Eigen::MatrixXd::Zero(this->generations*this->populationSizeGA*
            this->maxSampleRate,noSpecies);
    this->pops = Eigen::MatrixXd::Zero(this->generations*this->populationSizeGA*
            this->maxSampleRate,noSpecies);
    this->use = Eigen::VectorXd::Zero(this->generations*this->populationSizeGA*
            this->maxSampleRate);

    this->maxLearnNo = maxLearnNo;
    this->minLearnNo = minLearnNo;
    this->learnPeriod = learnPeriod;
    this->surrThresh = surrThresh;
    this->surrErr = 1;
}

void RoadGA::creation() {

    unsigned long individualsPerPop = this->populationSizeGA/5;

    double sx = this->designParams->getStartX();
    double sy = this->designParams->getStartY();
    double ex = this->designParams->getEndX();
    double ey = this->designParams->getEndY();

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
                individualsPerPop,this->currentRoadPopulation);
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
                2*individualsPerPop,this->currentRoadPopulation);
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
                3*individualsPerPop,this->currentRoadPopulation);
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
                4*individualsPerPop,this->currentRoadPopulation);
    }

    // Now check for invalid rows (NaN or Inf elements)
    Eigen::VectorXd dummyCosts = Eigen::VectorXd::Zero(this->populationSizeGA);
    this->replaceInvalidRoads(this->currentRoadPopulation, dummyCosts);
}

void RoadGA::crossover(const Eigen::VectorXi& parentsIdx, Eigen::MatrixXd&
        children) {
    // Prior to calling this routine, make sure that the population of roads is
    // valid.

    Eigen::MatrixXd& parents = this->currentRoadPopulation;

    try {
        assert(parentsIdx.size() == children.rows());
    } catch (int err) {
        std::cerr << "Number of children must be equal to the number of parents in GA"
                 << std::endl;
    }

    unsigned long intersectPts = this->designParams->getIntersectionPoints();
    unsigned long len = parents.cols();

    // First ensure that the parents provided are valid
    //this->replaceInvalidRoads(parents, costs);

    std::default_random_engine generator;
    std::uniform_int_distribution<int> crossMethod(1,4);
    std::uniform_int_distribution<int> cross1(1,intersectPts);
    std::uniform_real_distribution<double> cross3(0,1);

    for (unsigned long ii = 0; ii < 0.5 * parentsIdx.size(); ii++) {

        int routine = crossMethod(generator);

        int r1 = parentsIdx(2*ii);
        int r2 = parentsIdx(2*ii+1);

        if (routine == 1) {
            // CROSSOVER 1: Simple Crossover //////////////////////////////////
            // Point from which to switch intersection points
            int kk = cross1(generator);

            children.block(2*ii,0,1,kk*3) = parents.block(r1,0,1,kk*3);
            children.block(2*ii,kk*3,1,len-kk*3) = parents.block(r2,kk*3,
                    1,len-kk*3);

            children.block(2*ii+1,0,1,kk*3) = parents.block(r2,0,1,kk*3);
            children.block(2*ii+1,kk*3,1,len-kk*3) = parents.block(r1,kk*3,
                    1,len-kk*3);

        } else if (routine == 2) {
            // CROSSOVER 2: Two-Point Crossover ///////////////////////////////
            Eigen::VectorXi pts = Eigen::VectorXi::LinSpaced(
                    intersectPts,1,intersectPts);
            std::random_shuffle(pts.data(),pts.data()+intersectPts);

            Eigen::VectorXi twoPts = pts.segment(0,2);
            std::sort(twoPts.data(),twoPts.data()+2);

            children.block(2*ii,0,1,3*twoPts(0)) = parents.block(r1,0,1,3*
                    twoPts(0));
            children.block(2*ii,3*twoPts(0),1,3*(twoPts(1)-twoPts(0))) =
                    parents.block(r2,3*twoPts(0),1,3*(twoPts(1)-
                    twoPts(0)));
            children.block(2*ii,3*twoPts(1),1,len-3*twoPts(1)) = parents.block(
                    r1,3*twoPts(1),1,len-3*twoPts(1));

            children.block(2*ii+1,0,1,3*twoPts(0)) = parents.block(r2,0,1,
                    3*twoPts(0));
            children.block(2*ii+1,3*twoPts(0),1,3*(twoPts(1)-twoPts(0))) =
                    parents.block(r1,3*twoPts(0),1,3*(twoPts(1)-
                    twoPts(0)));
            children.block(2*ii+1,3*twoPts(1),1,len-3*twoPts(1)) = parents.block(
                    r2,3*twoPts(1),1,len-3*twoPts(1));

        } else if (routine == 3) {
            // CROSSOVER 3: Arithmetic Crossover //////////////////////////////
            double omega = cross3(generator);

            children.row(2*ii) = parents.row(r1)*omega + parents.row(r2)*(1 -
                    omega);
            children.row(2*ii+1) = parents.row(r2)*omega + parents.row(r1)*(1 -
                    omega);

        } else {
            // CROSSOVER 4: Heuristic Crossover ///////////////////////////////
            // For some reason, the original paper by Jong and Schonfeld (2003)
            // has r1 and r2 swapped in the algorithm (i.e. move the better
            // road towards the worse one).
            // More importantly, the authors suggest that the generated road
            // may not satisfy the boundary conditions (of the rectangular
            // region). This does not make sense as the generated road's line
            // segments lie on the straight line segments connecting the
            // intersection points of the parent roads. As the rectangular
            // region is a convex set and we are placing the new road on
            // straight line segments joining points that already lie within
            // this convex set, then the resulting road must also lie within
            // the region. Hence, iteration is not required.
            //
            // Finally, we generate two roads with two different omegas for
            // consistency with the other crossover methods.
            double omega1 = cross3(generator);
            double omega2 = cross3(generator);

            if (costs(r1) > costs(r2)) {
                // Switch the pointers for the parents
                r2 = parentsIdx(2*ii);
                r1 = parentsIdx(2*ii+1);
            }

            children.row(2*ii) = omega1*(parents.row(r1) - parents.row(r2)) +
                    parents.row(r2);
            children.row(2*ii+1) = omega2*(parents.row(r1) - parents.row(r2)) +
                    parents.row(r2);
        }
    }

    // Now check for invalid rows (NaN or Inf elements)
    Eigen::VectorXd dummyCosts = Eigen::VectorXd::Zero(children.rows());
    this->replaceInvalidRoads(children, dummyCosts);
}

void RoadGA::mutation(const Eigen::VectorXi &parentsIdx,
        Eigen::MatrixXd& children) {

    Eigen::MatrixXd& parents = this->currentRoadPopulation;

    // Region limits. We ignore the two outermost cell boundaries
    double minLon = (this->getRegion()->getX())(3,1);
    double maxLon = (this->getRegion()->getX())(this->getRegion()->getX().
            rows()-2,1);
    double minLat = (this->getRegion()->getY())(1,3);
    double maxLat = (this->getRegion()->getY())(1,this->getRegion()->getY().
            cols()-2);

    try {
        assert(parentsIdx.size() == children.rows());
    } catch (int err) {
        std::cerr << "Number of children must be equal to the number of parents in GA"
                 << std::endl;
    }


    unsigned long intersectPts = this->designParams->getIntersectionPoints();

    std::default_random_engine generator;
    std::uniform_int_distribution<int> mutateMethod(1,4);
    std::uniform_int_distribution<int> cross1(1,intersectPts);
    std::uniform_real_distribution<double> mutate1(0,1);
    std::uniform_int_distribution<int> mutate3(0,1);

    for (unsigned long ii = 0; ii < parentsIdx.size(); ii++) {

        children.row(ii) = parents.row(parentsIdx(ii));

        int routine = mutateMethod(generator);

        int r1 = parentsIdx(ii);

        if (routine == 1) {
            // MUTATION 1: Uniform Mutation ///////////////////////////////////
            // Randomly select the point of intersection to mutate
            int kk = cross1(generator);
            children(ii,3*kk) = minLon + mutate1(generator)*(maxLon - minLon);
            children(ii,3*kk+1) = minLat + mutate1(generator)*(maxLat -
                    minLat);

            // Randomly select two points, one on either side (l and m), to
            // act as the two other independent locii for the CURVE ELIMINATION
            // PROCEDURE and modify all intervening points.
            std::uniform_int_distribution<int> cross1a(0,kk - 1);
            std::uniform_int_distribution<int> cross1b(kk+1,intersectPts+1);

            int jj = cross1a(generator);
            int ll = cross1b(generator);

            // Curve elimination
            this->curveEliminationProcedure(ii,jj,kk,ll,children);

            // Compute the elevations of all the points in the offspring
            this->zOnTerrain(1,this->designParams->getIntersectionPoints(),ii,
                    children);


        } else if (routine == 2) {
            // MUTATION 2: Straight Mutation //////////////////////////////////
            Eigen::VectorXi pts = Eigen::VectorXi::LinSpaced(
                    intersectPts,1,intersectPts);
            std::random_shuffle(pts.data(),pts.data()+intersectPts);

            Eigen::VectorXi twoPts = pts.segment(0,2);
            std::sort(twoPts.data(),twoPts.data()+2);

            // A and B will not be the same thanks to the random shuffle.
            // This ensures that we do actually get mutation.
            if ((twoPts(1) - twoPts(0)) >= 2) {
                int jj = twoPts(1);
                int kk = twoPts(1);

                // X values
                Eigen::VectorXd level = Eigen::VectorXd::LinSpaced(kk-jj-1,1,
                        kk-jj-1);
                Eigen::MatrixXd xvals = (children(ii,3*jj) +
                        (level*(children(ii,3*kk) - children(ii,3*jj))/
                        ((double)(kk - jj))).array()).matrix();
                Eigen::VectorXi xIdxI = Eigen::VectorXi::LinSpaced(kk-jj-1,
                        3*(jj+1),3*(kk-1));
                Eigen::VectorXi IdxJ = Eigen::VectorXi::Constant(kk-jj-1,ii);
                igl::slice_into(xvals,xIdxI,IdxJ,children);

                // Y values
                Eigen::MatrixXd yvals = (children(ii,3*jj+1) +
                        (level*(children(ii,3*kk+1) - children(ii,3*jj+1))/
                        ((double)(kk - jj))).array()).matrix();
                Eigen::VectorXi yIdxI = Eigen::VectorXi::LinSpaced(kk-jj-1,
                        3*(jj+1)+1,3*(kk-1)+1);
                igl::slice_into(yvals,yIdxI,IdxJ,children);

                // Z values
                Eigen::MatrixXd zvals = (children(ii,3*jj+2) +
                        (level*(children(ii,3*kk+2) - children(ii,3*jj+2))/
                        ((double)(kk - jj))).array()).matrix();
                Eigen::VectorXi zIdxI = Eigen::VectorXi::LinSpaced(kk-jj-1,
                        3*(jj+1)+2,3*(kk-1)+2);
                igl::slice_into(zvals,zIdxI,IdxJ,children);
            }


        } else if (routine == 3) {
            // MUTATION 3: Non-Uniform Mutation ///////////////////////////////
            int kk = cross1(generator);
            double rd1 = mutate3(generator);
            double rd2 = mutate3(generator);

            double scaling = pow((1 - this->generation/this->generations),
                    this->scale);

            if (rd1 == 0) {
                children(ii,3*kk) = children(ii,3*kk) - (children(ii,3*kk) -
                        minLon)*mutate1(generator)*scaling;
            } else {
                children(ii,3*kk) = children(ii,3*kk) + (maxLon - children(
                        ii,3*kk))*mutate1(generator)*scaling;
            }

            if (rd2 == 0) {
                children(ii,3*kk+1) = children(ii,3*kk+1) - (children(ii,
                        3*kk+1) - minLat)*mutate1(generator)*scaling;
            } else {
                children(ii,3*kk+1) = children(ii,3*kk+1) + (maxLat - children(
                        ii,3*kk+1))*mutate1(generator)*scaling;
            }

            // Randomly select two points, one on either side (l and m), to
            // act as the two other independent locii for the CURVE ELIMINATION
            // PROCEDURE and modify all intervening points.
            std::uniform_int_distribution<int> cross1a(0,kk - 1);
            std::uniform_int_distribution<int> cross1b(kk+1,intersectPts+1);

            int jj = cross1a(generator);
            int ll = cross1b(generator);

            // Curve elimination
            this->curveEliminationProcedure(ii,jj,kk,ll,children);

            // Compute the elevations of all the points in the offspring
            this->zOnTerrain(1,this->designParams->getIntersectionPoints(),ii,
                    children);

        } else {
            // MUTATION 4: Whole Non-Uniform Mutation /////////////////////////
            // Generate random sequence and successively perform the non-
            // uniform mutation on the points in the sequence

            double scaling = pow((1 - this->generation/this->generations),
                    this->scale);

            Eigen::VectorXi pts = Eigen::VectorXi::LinSpaced(
                    intersectPts,1,intersectPts);
            std::random_shuffle(pts.data(),pts.data()+intersectPts);

            for (int mm = 0; mm < pts.size(); mm++) {
                int kk = pts(mm);

                double rd1 = mutate3(generator);
                double rd2 = mutate3(generator);

                if (rd1 == 0) {
                    children(ii,3*kk) = children(ii,3*kk) -
                            (children(ii,3*kk) - minLon)*
                            mutate1(generator)*scaling;
                } else {
                    children(ii,3*kk) = children(ii,3*kk) + (maxLon - children(
                            ii,3*kk))*mutate1(generator)*scaling;
                }

                if (rd2 == 0) {
                    children(ii,3*kk+1) = children(ii,3*kk+1) - (children(ii,
                            3*kk+1) - minLat)*mutate1(generator)*scaling;
                } else {
                    children(ii,3*kk+1) = children(ii,3*kk+1) + (maxLat -
                            children(ii,3*kk+1))*mutate1(generator)*scaling;
                }

                // Randomly select two points, one on either side (l and m), to
                // act as the two other independent locii for the CURVE
                // ELIMINATION PROCEDURE and modify all intervening points.
                std::uniform_int_distribution<int> cross1a(0,kk - 1);
                std::uniform_int_distribution<int> cross1b(kk+1,intersectPts+1);

                int jj = cross1a(generator);
                int ll = cross1b(generator);

                // Curve elimination
                this->curveEliminationProcedure(ii,jj,kk,ll,children);
            }

            // Compute the elevations of all the points in the offspring
            this->zOnTerrain(1,this->designParams->getIntersectionPoints(),ii,
                    children);
        }
    }

    // Now check for invalid rows (NaN or Inf elements)
    Eigen::VectorXd dummyCosts = Eigen::VectorXd::Zero(children.rows());
    this->replaceInvalidRoads(children, dummyCosts);
}

void RoadGA::elite(const Eigen::VectorXi& parentsIdx,
        Eigen::MatrixXd& children) {

    int cols = this->currentRoadPopulation.cols();
    Eigen::RowVectorXi colIdx = Eigen::RowVectorXi::LinSpaced(cols,0,
            cols-1);
    int rows = parentsIdx.size();

    Eigen::MatrixXi rowRefs = parentsIdx.rowwise().replicate(cols);
    Eigen::MatrixXi colRefs = colIdx.colwise().replicate(rows);

    igl::slice(this->currentRoadPopulation,rowRefs,colRefs,children);
}

void RoadGA::optimise() {
    this->creation();

    int status = 0;

    // Initially, the surrogate model is simply a straight line for all
    // AAR values (100% survival)
    this->defaultSurrogate();

    while ((status <= 0) && (this->generation <= this->generations)) {
        // Evaluate current generation
        this->evaluateGeneration();
        this->output();
        this->assignBestRoad();
        status = this->stopCheck();

        if (status != 0) {
            continue;
        }

        // Prepare for next generation
        this->computeSurrogate();

        int pc = (floor(this->populationSizeGA * this->crossoverFrac))*2;
        int pm = floor(this->populationSizeGA * this->mutationRate);
        int pe = this->populationSizeGA - pc - pm;
        int cols = this->currentRoadPopulation.cols();

        Eigen::VectorXi parentsCrossover(pc);
        Eigen::VectorXi parentsMutation(pm);
        Eigen::VectorXi parentsElite(pe);
        Eigen::MatrixXd crossoverChildren(pc,cols);
        Eigen::MatrixXd mutationChildren(pm,cols);
        Eigen::MatrixXd eliteChildren(pe,cols);

        // Select the parents for crossover, mutation and elite children
        this->selection(parentsCrossover, parentsMutation, parentsElite,
                RoadGA::TOURNAMENT);

        // Perform crossover, mutation and elite subpopulation creation
        this->crossover(parentsCrossover,crossoverChildren);
        this->mutation(parentsMutation,mutationChildren);
        this->elite(parentsElite,eliteChildren);

        // Assign the new population
        this->currentRoadPopulation.block(0,0,pc/2,cols) = crossoverChildren;
        this->currentRoadPopulation.block(pc/2,0,pm,cols) = mutationChildren;
        this->currentRoadPopulation.block(pc/2+pm,0,pe,cols) = eliteChildren;

        this->generation++;
    }
}

void RoadGA::assignBestRoad() {

    // For some reason, igl does not allow passing a vector as the template
    // does not state the first input as a derived type
    Eigen::Matrix<double,Eigen::Dynamic,1> Y(1);
    Eigen::Matrix<int,Eigen::Dynamic,1> I(1);
    Eigen::Matrix<double,Eigen::Dynamic,Eigen::Dynamic> costs(
            this->populationSizeGA,1);
    costs = this->costs;

    igl::mat_max(costs,1,Y,I);

    RoadPtr road(new Road(this->me(),this->currentRoadPopulation.row(I(1))));

    this->bestRoads[this->scenario->getCurrentScenario()][this->scenario->
            getRun()] = road;
}

void RoadGA::output() {

    this->best(this->generation) = this->currentRoadPopulation.minCoeff();
    this->av(this->generation) = this->currentRoadPopulation.mean();
    this->surrFit(this->generation) = this->surrErr;
    // Include code to plot using gnuplot utilising POSIX pipes to plot the
    // best road
}

void RoadGA::defaultSurrogate() {

    int scenario = this->getScenario()->getCurrentScenario();
    int run = this->getScenario()->getRun();


}

void RoadGA::computeSurrogate() {
    // Select individuals for computing learning for the surrogate model
    // First select the proportion of individuals to compute for each
    // surrogate model.
    // We always wish to select some samples at each iteration to keep
    // testing the effectiveness of the algorithm.
    double pFull;

    if (this->type == Optimiser::MTE) {
        // MTE can support around 50 roads per generation at maximum learning,
        // 10 at minimum learning (during learning period) or 3 (after learning
        // period).
        if (this->generation < this->learnPeriod) {
            pFull = std::max((this->maxLearnNo/this->populationSizeGA)*
                    std::min((this->surrErr - this->surrThresh)*((double)
                    (this->surrErr > this->surrThresh))*this->maxLearnNo,1.0),
                    10.0/this->populationSizeGA);
        } else {
            pFull = std::max((this->maxLearnNo/this->populationSizeGA)*
                    std::min((this->surrErr - this->surrThresh)*((double)
                    (this->surrErr > this->surrThresh))*this->maxLearnNo,1.0),
                    3.0/this->populationSizeGA);
        }
    } else if (this->type == Optimiser::CONTROLLED) {
        // We treat ROV the same as MTE for now even though it should be much
        // much slower (we may have to resort to ignoring forward path
        // recomputation to improve speed at the expense of accuracy.
        if (this->generation < this->learnPeriod) {
            pFull = std::max((this->maxLearnNo/this->populationSizeGA)*
                    std::min((this->surrErr - this->surrThresh)*((double)
                    (this->surrErr > this->surrThresh))*this->maxLearnNo,1.0),
                    10.0/this->populationSizeGA);
        } else {
            pFull = std::max((this->maxLearnNo/this->populationSizeGA)*
                    std::min((this->surrErr - this->surrThresh)*((double)
                    (this->surrErr > this->surrThresh))*this->maxLearnNo,1.0),
                    3.0/this->populationSizeGA);
        }
    }

    // Actual number of roads to sample
    this->noSamples = ceil(pFull * this->populationSizeGA);
    Eigen::VectorXi sampleRoads(this->noSamples);

    // Select individuals for computing learning from the full model
    // We first sort the roads by cost
    Eigen::VectorXi sortedIdx(this->populationSizeGA);
    Eigen::VectorXd sorted(this->populationSizeGA);
    igl::sort(this->costs,1,true,sorted,sortedIdx);

    // We will always test the three best roads
    sampleRoads.segment(0,3) = sortedIdx.segment(0,3);

    // Now fill up the test road indices
    std::random_shuffle(sortedIdx.data()+3,sortedIdx.data() +
            this->populationSizeGA);
    sampleRoads.segment(3,this->noSamples-1) = sortedIdx.segment(3,
            this->noSamples-1);

    // Call the thread pool. The computed function and form of the surrogate
    // models are different under each scenario (MTE vs CONTROLLED).
    if (this->type = Optimiser::MTE) {

        std::vector< std::future< Eigen::MatrixXd > >
                results(this->noSamples);


        if (this->threader != nullptr) {
            for (unsigned long ii = 0; ii < this->noSamples; ii++) {

                // Lambda function to be passed to threadpool
                // This lambda function is MIMO. That is, we pass in the AAR of
                // each species in the region at time t=0 and the respective
                // resulting end populations to build f_i for each species,
                // where
                //      endPop_i = f_i(aar_i)
                results[ii] = this->threader->push([this](int id) {
                    RoadPtr road(new Road(this->me(),
                            this->currentRoadPopulation.row(id)));
                    return this->surrogateResultsMTE(road);
                });
            }

            for (unsigned long ii = 0; ii < this->noSamples; ii++) {
                results[ii].get();
            }
        }

        // Now that we have the results, let's build the surrogate model!!!

    } else if (this->type = Optimiser::CONTROLLED) {

        std::vector< std::future< Eigen::MatrixXd > >
                results(this->noSamples);

        if (this->threader != nullptr) {
            for (unsigned long ii = 0; ii < this->noSamples; ii++) {

                // Lambda function to be passed to threadpool
                // This lambda function is MISO. That is, we pass in the AAR of
                // each species in the region at time t=0 and the resulting
                // overall road utilisation to build f,
                // where
                //      road_utilisation = f(aar_0,...,aar_n)
                //
                // where n is the number of animal species
                results[ii] = this->threader->push([this](int id) {
                    RoadPtr road(new Road(this->me(),
                            this->currentRoadPopulation.row(id)));
                    return this->surrogateResultsMTE(road);
                });
            }

            for (unsigned long ii = 0; ii < this->noSamples; ii++) {
                results[ii].get();
            }
        }

        // Now that we have the results, let's build the surrogate model!!!
    }
}

void RoadGA::evaluateGeneration() {}

void RoadGA::selection(Eigen::VectorXi& pc, Eigen::VectorXi& pm,
        Eigen::VectorXi& pe, RoadGA::Selection selector) {

    switch (selector) {

    case TOURNAMENT:
    {
        break;
    }
    case 2:
    {
        break;
    }
    case 3:
    {
        break;
    }
    default:
    {
        break;
    }
    }
}

int RoadGA::stopCheck() {}

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
    Eigen::VectorXi rowIdx = Eigen::VectorXi::LinSpaced(individuals,
            startRow,startRow + individuals - 1);
    Eigen::RowVectorXi colIdx = Eigen::RowVectorXi::LinSpaced(intersectPts,3,
            3*intersectPts);

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
    Eigen::RowVectorXi colIdx = Eigen::RowVectorXi::LinSpaced(intersectPts,3,
            3*intersectPts);

    igl::slice_into(xvals,rowIdx.rowwise().replicate(intersectPts),
            colIdx.colwise().replicate(individuals),
            this->currentRoadPopulation);
    igl::slice_into(yvals,rowIdx.rowwise().replicate(intersectPts),
            (colIdx.array() + 1).colwise().replicate(individuals),
            this->currentRoadPopulation);
}

void RoadGA::randomZWithinRange(const long &individuals, const long&
            intersectPts, const long& startRow, Eigen::MatrixXd& population) {

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

        igl::slice(population,row,col,xtemp);
        igl::slice(population,row,col,ytemp);

        RoadPtr road(new Road(this->me(),xtemp,ytemp,
                Eigen::VectorXd::Zero(intersectPts)));
        road->computeAlignment();

        sTemp.row(ii) = road->getVerticalAlignment()->getSDistances();
    }

    for (long ii = 0; ii < intersectPts; ii++) {
        Eigen::VectorXd randVec = Eigen::VectorXd::Random(individuals);

        Eigen::VectorXd zL = (population.block(startRow,
                3*ii+2,individuals,1) - (sTemp.col(ii+1) - sTemp.col(ii))*
                gmax/100).array().max((population.block(
                startRow,3*(intersectPts + 2) - 1,individuals,1) -
                (sTemp.col(intersectPts + 2) - sTemp.col(ii+1))*gmax/100).
                array()).matrix();

        Eigen::VectorXd zU = (population.block(startRow,
                3*ii+2,individuals,1) - (sTemp.col(ii+1) + sTemp.col(ii))*
                gmax/100).array().min((population.block(
                startRow,3*(intersectPts + 2) - 1,individuals,1) - (sTemp.
                col(intersectPts + 2) + sTemp.col(ii+1))*gmax/100).array()).
                matrix();

        population.block(startRow,3*(ii+1)+2,individuals,1) =
                ((randVec.array())*(zU-zL).array() + zL.array()).matrix();
    }
}

void RoadGA::zOnTerrain(const long &individuals, const long&
            intersectPts, const long& startRow, Eigen::MatrixXd& population) {

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

        igl::slice(population,row,col,xtemp);
        igl::slice(population,row,col,ytemp);

        RoadPtr road(new Road(this->me(),xtemp,ytemp,
                Eigen::VectorXd::Zero(intersectPts)));
        road->computeAlignment();

        sTemp.row(ii) = road->getVerticalAlignment()->getSDistances();
    }

    for (long ii = 0; ii < intersectPts; ii++) {
        Eigen::VectorXd randVec = Eigen::VectorXd::Random(individuals);

        Eigen::VectorXd zL = (population.block(startRow,
                3*ii+2,individuals,1) - (sTemp.col(ii+1) - sTemp.col(ii))*
                gmax/100).array().max((population.block(
                startRow,3*(intersectPts + 2) - 1,individuals,1) -
                (sTemp.col(intersectPts + 2) - sTemp.col(ii+1))*gmax/100).
                array()).matrix();

        Eigen::VectorXd zU = (population.block(startRow,
                3*ii+2,individuals,1) - (sTemp.col(ii+1) + sTemp.col(ii))*
                gmax/100).array().min((population.block(
                startRow,3*(intersectPts + 2) - 1,individuals,1) - (sTemp.
                col(intersectPts + 2) + sTemp.col(ii+1))*gmax/100).array()).
                matrix();

        Eigen::VectorXd zE(individuals);

        this->region->placeNetwork(population.block(
                startRow,3*(ii+1), individuals,1), population.
                block(startRow,3*(ii+1)+1, individuals,1), zE);

        Eigen::VectorXd selectGround = ((zE.array() <=
                zU.array()) && (zE.array() >= zL.array())).cast<double>();
        Eigen::VectorXd selectLower = (zE.array() < zL.array()).cast<double>();
        Eigen::VectorXd selectUpper = (zE.array() > zU.array()).cast<double>();

        population.block(startRow,3*(ii+1)+2,individuals,1) =
                (zE.array() * selectGround.array() + zL.array() * selectLower
                .array() + zU.array() * selectUpper.array()).matrix();
    }
}

void RoadGA::replaceInvalidRoads(Eigen::MatrixXd& roads, Eigen::VectorXd&
        costs) {
    Eigen::MatrixXi invalid(roads.rows(),roads.cols()+1);
    Eigen::MatrixXd input(roads.rows(),roads.cols()+1);
    input << roads, costs;
    Eigen::VectorXi rows(roads.rows());

    invalid = (input.unaryExpr([](double v){ return
            std::isfinite(v); }).cast<int>());
    rows = ((invalid.rowwise().sum()).array() > 0).cast<int>();
    long noBadRows = rows.count();

    if (noBadRows > 0) {
        Eigen::VectorXi rowsg(roads.rows());
        rowsg = (1 - rows.array()).matrix();

        // Replace bad members with good members
        Eigen::VectorXi badRows(noBadRows);
        Eigen::VectorXi goodRows(roads.rows() - noBadRows);
        igl::find(rows,badRows);
        igl::find(rowsg,goodRows);

        std::random_shuffle(rowsg.data(),rowsg.data() + roads.rows() -
                noBadRows);

        std::cout << "Population contains " << std::to_string(noBadRows) << " individual(s) with NaN or Inf entries. Replacing with good individuals." << std::endl;

        for (int ii = 0; ii < noBadRows; ii++) {
            roads.row(ii) = roads.row(rowsg(ii));
            costs(ii) = costs(rowsg(ii));
        }
    }
}

void RoadGA::curveEliminationProcedure(int ii, int jj, int kk, int ll,
        Eigen::MatrixXd& children) {

    if ((kk - jj) >= 2) {
        Eigen::VectorXd level = Eigen::VectorXd::LinSpaced(kk-jj,0,
                kk-jj-1);
        Eigen::MatrixXd xvals = (children(ii,3*jj) +
                (level*(children(ii,3*kk) - children(ii,3*jj))/
                ((double)(kk - jj))).array()).matrix();
        Eigen::VectorXi xIdxI = Eigen::VectorXi::LinSpaced(kk-jj,3*jj,
                3*(kk-1));
        Eigen::VectorXi IdxJ = Eigen::VectorXi::Constant(kk-jj,ii);
        igl::slice_into(xvals,xIdxI,IdxJ,children);

        Eigen::MatrixXd yvals = (children(ii,3*jj+1) +
                (level*(children(ii,3*kk+1) - children(ii,3*jj+1))/
                ((double)(kk - jj))).array()).matrix();
        Eigen::VectorXi yIdxI = Eigen::VectorXi::LinSpaced(kk-jj,
                3*jj+1,3*(kk-1)+1);
        igl::slice_into(yvals,yIdxI,IdxJ,children);
    }

    if ((ll - kk) >= 2) {
        Eigen::VectorXd level = Eigen::VectorXd::LinSpaced(ll-kk,1,
                ll-kk);
        Eigen::MatrixXd xvals = (children(ii,3*kk) +
                (level*(children(ii,3*ll) - children(ii,3*kk))/
                ((double)(ll - kk))).array()).matrix();
        Eigen::VectorXi xIdxI = Eigen::VectorXi::LinSpaced(ll-kk,3*(
                kk+1),3*ll);
        Eigen::VectorXi xIdxJ = Eigen::VectorXi::Constant(ll-kk,ii);
        igl::slice_into(xvals,xIdxI,xIdxJ,children);

        Eigen::MatrixXd yvals = (children(ii,3*kk) +
                (level*(children(ii,3*ll+1) - children(ii,3*kk+1))/
                ((double)(kk - jj))).array()).matrix();
        Eigen::VectorXi yIdxI = Eigen::VectorXi::LinSpaced(ll-kk,
                3*(kk+1)+1,3*ll+1);
        Eigen::VectorXi yIdxJ = Eigen::VectorXi::Constant(kk-jj,ii);
        igl::slice_into(yvals,yIdxI,yIdxJ,children);
    }
}

Eigen::MatrixXd RoadGA::surrogateResultsMTE(RoadPtr road) {
    Eigen::MatrixXd mteResult(3,this->species.size());

    road->designRoad();
    road->evaluateRoad(true);
    std::vector<SpeciesRoadPatchesPtr> species =
            road->getSpeciesRoadPatches();

    for (int ii = 0; ii < this->species.size(); ii++) {
        Eigen::VectorXd iars = species[ii]->getInitAAR();
        mteResult(0,ii) = iars(iars.size()-1);
        mteResult(1,ii) = species[ii]->getEndPopMean();
        mteResult(2,ii) = species[ii]->getEndPopSD();
    }

    return mteResult;
}

Eigen::MatrixXd RoadGA::surrogateResultsROVCR(RoadPtr road) {
    Eigen::MatrixXd mteResult(1,this->species.size()+2);

    road->designRoad();
    road->evaluateRoad(true);
    std::vector<SpeciesRoadPatchesPtr> species =
            road->getSpeciesRoadPatches();

    mteResult(0,1) = road->getAttributes()->getTotalUtilisationROV();
    mteResult(0,2) = road->getAttributes()->getTotalUtilisationROVSD();
    for (int ii = 0; ii < this->species.size(); ii++) {
        Eigen::VectorXd iars = species[ii]->getInitAAR();
        mteResult(0,ii+2) = iars(iars.size()-1);
    }

    return mteResult;
}
