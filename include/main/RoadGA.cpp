#include "../transportbase.h"

RoadGA::RoadGA() : Optimiser() {
    // Do not use for now
    this->theta = 0;
    this->generation = 0;
    this->scale = scale;
    this->noSamples = 0;
    this->maxLearnNo = 100;
    this->minLearnNo = 10;
    this->learnPeriod = 10;
    this->surrThresh = 0.05;
    this->surrDimRes = 20;
}

RoadGA::RoadGA(double mr, double cf, unsigned long gens, unsigned long popSize,
    double stopTol, double confInt, double confLvl, unsigned long habGridRes,
    unsigned long surrDimRes, std::string solScheme, unsigned long noRuns,
    Optimiser::Type type, double scale, unsigned long learnPeriod, double
    surrThresh, unsigned long maxLearnNo, unsigned long minLearnNo, unsigned
    long sg, RoadGA::Selection selector, RoadGA::Scaling fitscale, double
    topProp, double maxSurvivalRate, int ts, double msr, bool gpu,
    Optimiser::ROVType rovType, Optimiser::InterpolationRoutine interp) :
    Optimiser(mr, cf, gens, popSize, stopTol, confInt, confLvl, habGridRes,
            surrDimRes, solScheme, noRuns, type, sg, msr, gpu, rovType,
            interp) {

    this->theta = 0;
    this->generation = 0;
    this->scale = scale;

    this->maxLearnNo = maxLearnNo;
    this->minLearnNo = minLearnNo;
    this->learnPeriod = learnPeriod;
    this->surrThresh = surrThresh;
    this->noSamples = 0;
    this->selector = selector;
    this->fitScaling = fitscale;
    this->topProp = topProp;
    this->maxSurvivalRate = maxSurvivalRate;
    this->tournamentSize = ts;
}

void RoadGA::initialiseStorage() {
    Optimiser::initialiseStorage();

    this->xO = Eigen::VectorXd::Zero(this->designParams->
            getIntersectionPoints());
    this->yO = Eigen::VectorXd::Zero(this->designParams->
            getIntersectionPoints());
    this->zO = Eigen::VectorXd::Zero(this->designParams->
            getIntersectionPoints());
    this->dU = Eigen::VectorXd::Zero(this->designParams->
            getIntersectionPoints());
    this->dL = Eigen::VectorXd::Zero(this->designParams->
            getIntersectionPoints());

    int noSpecies = this->species.size();
    this->costs = Eigen::VectorXd::Zero(this->populationSizeGA);
    this->profits = Eigen::VectorXd::Zero(this->populationSizeGA);
    this->iarsCurr = Eigen::MatrixXd::Zero(this->populationSizeGA,noSpecies);
    this->popsCurr = Eigen::MatrixXd::Zero(this->populationSizeGA,noSpecies);
    this->useCurr = Eigen::VectorXd::Zero(this->populationSizeGA);

    this->best = Eigen::VectorXd::Zero(this->generations);
    this->av = Eigen::VectorXd::Zero(this->generations);
    this->surrFit = Eigen::VectorXd::Zero(this->generations);

    this->iars = Eigen::MatrixXd::Zero(this->generations*this->
            populationSizeGA*this->maxSampleRate,noSpecies);
    this->pops = Eigen::MatrixXd::Zero(this->generations*this->
            populationSizeGA*this->maxSampleRate,noSpecies);
    this->use = Eigen::VectorXd::Zero(this->generations*this->populationSizeGA*
            this->maxSampleRate);
    this->popsSD = Eigen::MatrixXd::Zero(this->generations*this->
            populationSizeGA*this->maxSampleRate,noSpecies);
    this->useSD = Eigen::VectorXd::Zero(this->generations*this->
            populationSizeGA*this->maxSampleRate);
}

void RoadGA::creation() {

    unsigned long individualsPerPop = this->populationSizeGA/5;

    double sx = this->designParams->getStartX();
    double sy = this->designParams->getStartY();
    double ex = this->designParams->getEndX();
    double ey = this->designParams->getEndY();

    // Region limits. We ignore the two outermost cell boundaries
    double minLon = (this->getRegion()->getX())(2,1);
    double maxLon = (this->getRegion()->getX())(this->getRegion()->getX().
            rows()-3,1);
    double minLat = (this->getRegion()->getY())(1,2);
    double maxLat = (this->getRegion()->getY())(1,this->getRegion()->getY().
            cols()-3);

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

    //std::default_random_engine generator;
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
    double minLon = (this->getRegion()->getX())(2,1);
    double maxLon = (this->getRegion()->getX())(this->getRegion()->getX().
            rows()-3,1);
    double minLat = (this->getRegion()->getY())(1,2);
    double maxLat = (this->getRegion()->getY())(1,this->getRegion()->getY().
            cols()-3);

    try {
        assert(parentsIdx.size() == children.rows());
    } catch (int err) {
        std::cerr << "Number of children must be equal to the number of parents in GA"
                 << std::endl;
    }

    unsigned long intersectPts = this->designParams->getIntersectionPoints();

    //std::default_random_engine generator;
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
                int jj = twoPts(0);
                int kk = twoPts(1);

                // X values
                Eigen::VectorXd level = Eigen::VectorXd::LinSpaced(kk-jj-1,1,
                        kk-jj-1);
                Eigen::MatrixXd xvals = (children(ii,3*jj) +
                        (level*(children(ii,3*kk) - children(ii,3*jj))/
                        ((double)(kk - jj))).array()).matrix();
                Eigen::VectorXi xIdxJ = Eigen::VectorXi::LinSpaced(kk-jj-1,
                        3*(jj+1),3*(kk-1));
                Eigen::VectorXi IdxI = Eigen::VectorXi::Constant(kk-jj-1,ii);
                Utility::sliceIntoPairs(xvals,IdxI,xIdxJ,children);

                // Y values
                Eigen::MatrixXd yvals = (children(ii,3*jj+1) +
                        (level*(children(ii,3*kk+1) - children(ii,3*jj+1))/
                        ((double)(kk - jj))).array()).matrix();
                Eigen::VectorXi yIdxJ = Eigen::VectorXi::LinSpaced(kk-jj-1,
                        3*(jj+1)+1,3*(kk-1)+1);
                Utility::sliceIntoPairs(yvals,IdxI,yIdxJ,children);

                // Z values
                Eigen::MatrixXd zvals = (children(ii,3*jj+2) +
                        (level*(children(ii,3*kk+2) - children(ii,3*jj+2))/
                        ((double)(kk - jj))).array()).matrix();
                Eigen::VectorXi zIdxJ = Eigen::VectorXi::LinSpaced(kk-jj-1,
                        3*(jj+1)+2,3*(kk-1)+2);
                Utility::sliceIntoPairs(zvals,IdxI,zIdxJ,children);
            }


        } else if (routine == 3) {
            // MUTATION 3: Non-Uniform Mutation ///////////////////////////////
            int kk = cross1(generator);
            double rd1 = mutate3(generator);
            double rd2 = mutate3(generator);

            double scaling = pow((1.0 - (double)this->generation/((double)
                    this->generations)),(double)this->scale);

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

            // Randomly select two points, one on either side (jj and ll), to
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

                // New X value
                if (rd1 == 0) {
                    children(ii,3*kk) = children(ii,3*kk) - (children(ii,3*kk)
                            - minLon)*mutate1(generator)*scaling;
                } else {
                    children(ii,3*kk) = children(ii,3*kk) + (maxLon - children(
                            ii,3*kk))*mutate1(generator)*scaling;
                }

                // New Y value
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

    igl::slice(this->currentRoadPopulation,parentsIdx,colIdx,children);
}

void RoadGA::optimise(bool plot) {
    // Initialise surrogate-related info
    if (this->type == Optimiser::MTE) {
        this->surrErr.resize(this->species.size());
        this->surrFit.resize(this->generations,this->species.size());

    } else if (this->type == Optimiser::CONTROLLED) {
        this->surrErr.resize(1);
        this->surrFit.resize(this->generations,1);
    }

    // Run parent-level initial commands
    Optimiser::optimise(plot);

    if (plot) {
        GnuplotPtr plotPtr(new Gnuplot);

        this->plothandle.reset();
        this->plothandle = plotPtr;

        if (this->type > Optimiser::SIMPLEPENALTY) {
            GnuplotPtr surrPlotPtr(new Gnuplot);

            this->surrPlotHandle.reset();
            this->surrPlotHandle = surrPlotPtr;
        }
    }

    this->creation();

    int status = 0;

    // Initially, the surrogate model is simply a straight line for all
    // AAR values (100% survival)
    //this->defaultSurrogate();
    // Evaluate once to initialise the surrogate if we are using MTE or ROV
//    if (this->type > Optimiser::SIMPLEPENALTY) {
//        this->evaluateGeneration();
//        this->computeSurrogate();
//    } else {
//        surrErr = 0;
//    }

    while ((status == 0) && (this->generation <= this->generations)) {
        // Evaluate the current generation using the new surrogate
        this->evaluateGeneration();
        this->output();
        this->assignBestRoad();

        // Prepare for next generation
        if (this->type > Optimiser::SIMPLEPENALTY) {
            this->computeSurrogate();
        }

        int pc = 2*floor((this->populationSizeGA * this->crossoverFrac)/2);
        int pm = 2*floor((this->populationSizeGA * this->mutationRate)/2);
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
                this->selector);

        // Perform crossover, mutation and elite subpopulation creation
        this->crossover(parentsCrossover,crossoverChildren);
        this->mutation(parentsMutation,mutationChildren);
        this->elite(parentsElite,eliteChildren);

        // Assign the new population
        this->currentRoadPopulation.block(0,0,pc,cols) = crossoverChildren;
        this->currentRoadPopulation.block(pc,0,pm,cols) = mutationChildren;
        this->currentRoadPopulation.block(pc+pm,0,pe,cols) = eliteChildren;

        // Print important results to console
        std::cout << "Generation: " << this->generation << "\t Average Cost: "
                << this->costs(this->generation) << "\t Best Road: " <<
                this->best(this->generation);

        if (this->type > Optimiser::SIMPLEPENALTY) {
            std::cout << "\t Surrogate Error: " <<
                    this->surrFit(this->generation);
        }

        std::cout << std::endl;

//        Eigen::MatrixXd xvals;
//        Eigen::MatrixXd yvals;
//        Eigen::MatrixXd zvals;

//        Eigen::VectorXi rows = Eigen::VectorXi::LinSpaced(pm,pc,pc+pm-1);
//        int ip = this->designParams->getIntersectionPoints();
//        ip += 2;
//        Eigen::VectorXi xcols = Eigen::VectorXi::LinSpaced(ip,0,ip-1);
//        Eigen::VectorXi ycols = 3*xcols.array()+1;
//        Eigen::VectorXi zcols = 3*xcols.array()+2;
//        igl::slice(this->currentRoadPopulation,rows,xcols,xvals);
//        igl::slice(this->currentRoadPopulation,rows,ycols,yvals);
//        igl::slice(this->currentRoadPopulation,rows,zcols,zvals);
//        double xmax = xvals.maxCoeff();
//        double ymax = yvals.maxCoeff();
//        double xmin = xvals.minCoeff();
//        double ymin = yvals.minCoeff();

//        Eigen::MatrixXi valid = (this->currentRoadPopulation.unaryExpr([](double v){ return
//                std::isfinite(v); }).cast<int>());
//        Eigen::MatrixXi invalid = (valid.array() == 0).cast<int>();
//        long noBadRows = invalid.count();

        Optimiser::optimise(plot);
        this->plotResults(plot);

        status = this->stopCheck();

        if (status != 0) {
            continue;
        }

        this->generation++;
    }

    std::cout << "Press any key to end..." << std::endl;
    std::cin.get();
}

void RoadGA::plotResults(bool plot) {

    if (plot) {
        // Need to insert a routine to exploit multithreading to speed this up
        RoadPtr bestRoad = this->bestRoads[this->scenario->
                getCurrentScenario()][this->scenario->getRun()];
        bestRoad->designRoad();

        // Prepare terrain data
        std::vector<std::vector<std::vector<double>>> terr;
        terr.resize(region->getX().rows());

        for (int ii = 0; ii < this->region->getX().rows()-2; ii++) {
            terr[ii].resize(this->region->getX().rows()-2);
            for (int jj = 0; jj < this->region->getX().cols()-2; jj++) {
                terr[ii][jj].resize(3);
            }
        }

        Eigen::MatrixXd X = this->region->getX();
        Eigen::MatrixXd Y = this->region->getY();
        Eigen::MatrixXd Z = this->region->getZ();

        // We first need to adjust the terrain elevation along the road path so
        // that it coincides with the road's elevation at each station
        for (int ii = 0; ii < bestRoad->getRoadCells()->getUniqueCells().
                size(); ii++) {
            Eigen::VectorXd validCells(bestRoad->getRoadCells()->getCellRefs()
                    .size());

            validCells = (bestRoad->getRoadCells()->getCellRefs().array()
                    == bestRoad->getRoadCells()->getUniqueCells()(ii))
                    .cast<double>();

            double adjLevel = (validCells.array()*bestRoad->getRoadCells()->
                    getZ().segment(0,validCells.size()).array() + (1 -
                    validCells.array())*bestRoad->getRoadCells()->getZ().
                    maxCoeff()).minCoeff();

//            double adjLevel = ((bestRoad->getRoadCells()->getCellRefs().array()
//                    == bestRoad->getRoadCells()->getUniqueCells()(ii)))
//                    .select(bestRoad->getRoadSegments()->getZ().array(),
//                    std::numeric_limits<double>::infinity()).minCoeff();
            Eigen::MatrixXi adjMat = Eigen::MatrixXi::Zero(
                    this->region->getZ().rows(),this->region->getZ().cols());
            Eigen::VectorXd cellX(1);
            Eigen::VectorXd cellY(1);
            Eigen::VectorXd cellVal(1);
            cellVal(0) = bestRoad->getRoadCells()->getUniqueCells()(ii);

            Utility::ind2sub(adjMat,cellVal,cellX,cellY);

            adjMat(cellX(0),cellY(0)) = 1;
            adjMat(cellX(0)+1,cellY(0)) = 1;
            adjMat(cellX(0),cellY(0)+1) = 1;
            adjMat(cellX(0)+1,cellY(0)+1) = 1;

            Z = Z.array()*(1 - adjMat.array().cast<double>()) +
                    adjMat.array().cast<double>()*adjLevel;
        }

        for (int ii = 1; ii < this->region->getX().rows()-1; ii++) {
            for (int jj = 1; jj < this->region->getX().rows()-1; jj++) {
                terr[ii-1][jj-1][0] = X(ii,jj);
                terr[ii-1][jj-1][1] = Y(ii,jj);
                terr[ii-1][jj-1][2] = Z(ii,jj);
            }
        }

        // Prepare vegetation data
        std::vector<std::vector<std::vector<double>>> veg;
        veg.resize(this->region->getVegetation().rows()-2);

        for (int ii = 0; ii < this->region->getVegetation().rows()-2; ii++) {
            veg[ii].resize(this->region->getVegetation().rows()-2);
            for (int jj = 0; jj < this->region->getVegetation().rows()-2;
                    jj++) {
                veg[ii][jj].resize(3);
            }
        }

        Eigen::MatrixXi V = this->region->getVegetation();

        // Plot the road cells in the image
        for (int ii = 0; ii < bestRoad->getRoadCells()->getUniqueCells().
                size(); ii++) {

            V.data()[bestRoad->getRoadCells()->getUniqueCells()[ii]] = 0;
        }

        for (int ii = 1; ii < this->region->getVegetation().rows()-1; ii++) {
            for (int jj = 1; jj < this->region->getVegetation().rows()-1; jj++) {
                veg[ii-1][jj-1][0] = X(ii,jj);
                veg[ii-1][jj-1][1] = Y(ii,jj);
                veg[ii-1][jj-1][2] = V(ii,jj);
            }
        }

        // Prepare road path data
        std::vector<std::vector<double>> elev;
        elev.resize(bestRoad->getRoadSegments()->getX().size(),
                std::vector<double>(3));

        for (int ii = 0; ii < bestRoad->getRoadSegments()->getX().size();
                ii++) {
            elev[ii][0] = bestRoad->getRoadSegments()->getX()(ii);
            elev[ii][1] = bestRoad->getRoadSegments()->getY()(ii);
            elev[ii][2] = bestRoad->getRoadSegments()->getZ()(ii);
        }

        // Prepare best roads data
        std::vector<std::vector<double>> genCostsBest;
        std::vector<std::vector<double>> genCostsAv;

        genCostsBest.resize((this->generation+1),std::vector<double>(2));
        genCostsAv.resize((this->generation+1),std::vector<double>(2));

        for (int ii = 0; ii <= this->generation; ii++) {
            genCostsBest[ii][0] = ii+1;
            genCostsBest[ii][1] = this->best(ii);
            genCostsAv[ii][0] = ii+1;
            genCostsAv[ii][1] = this->av(ii);
        }

        // PLOTS //////////////////////////////////////////////////////////////
        // Clear Previous plot
        if (this->type > Optimiser::SIMPLEPENALTY) {
            (*this->plothandle) << "set multiplot layout 1,3\n";
            (*this->surrPlotHandle) << "set multiplot layout " <<
                    this->species.size() <<",2\n";
        } else {
            (*this->plothandle) << "set multiplot layout 1,3\n";
        }

        // Plot 1 (Terrain with Road)
        (*this->plothandle) << "set title '3D View of Terrain'\n";
        (*this->plothandle) << "set grid\n";
        (*this->plothandle) << "set hidden3d\n";
        //(*this->plothandle) << "unset logscale y\n";
        (*this->plothandle) << "unset key\n";
        (*this->plothandle) << "unset view\n";
        (*this->plothandle) << "unset pm3d\n";
        (*this->plothandle) << "unset xlabel\n";
        (*this->plothandle) << "unset ylabel\n";
        (*this->plothandle) << "set xrange [*:*]\n";
        (*this->plothandle) << "set yrange [*:*]\n";
        (*this->plothandle) << "set view 45,45\n";
        (*this->plothandle) << "splot '-' with lines, '-' with lines lw 2\n";
        (*this->plothandle).send2d(terr);
        (*this->plothandle).send1d(elev);
        (*this->plothandle).flush();

        // Plot 2 (Road on Vegetation)
        (*this->plothandle) << "set title 'Road Path Through Vegetation'\n";
        (*this->plothandle) << "unset key\n";
        (*this->plothandle) << "set pm3d\n";
        (*this->plothandle) << "set view map\n";
        (*this->plothandle) << "plot '-' with image\n";
        (*this->plothandle).send2d(veg);
        (*this->plothandle).flush();

        // Plot 3 (Stopping Criteria/Best and Average)
        // Best and Average Option
        (*this->plothandle) << "set title 'Best and Average Roads'\n";
        (*this->plothandle) << "unset key\n";
        (*this->plothandle) << "set grid\n";
        (*this->plothandle) << "set xlabel 'Generation'\n";
        (*this->plothandle) << "set ylabel 'Cost (AUD)'\n";
        //(*this->plothandle) << "set logscale y\n";
        (*this->plothandle) << "set xrange [0:" + std::to_string(
                this->generations) + "]\n";
        //(*this->plothandle) << "set logscale y\n";
        (*this->plothandle) << "set yrange [" + std::to_string(this->best
                .segment(0,this->noSamples).minCoeff()/2) + ":" +
                std::to_string(this->av.maxCoeff()) + "]\n";
        //(*this->plothandle) << xrange;
        (*this->plothandle) << "plot '-' with points pointtype 5, '-' with points pointtype 7 \n";
        (*this->plothandle).send1d(genCostsBest);
        (*this->plothandle).send1d(genCostsAv);

        // Stopping Criteria Option

        if (this->type == Optimiser::MTE) {
            // Prepare surrogate data for species present
            for (int ii = 0; ii < this->species.size(); ii++) {
                std::vector<std::vector<double>> surrData;
                surrData.resize(this->noSamples,std::vector<double>(2));
                std::vector<std::vector<double>> surrSDData;
                surrSDData.resize(this->noSamples,std::vector<double>(2));
                std::vector<std::vector<double>> surrErrData;
                surrErrData.resize(this->generation+1,std::vector<double>(2));

                // Surrogate sample data
                for (int jj = 0; jj < this->noSamples; jj++) {
                    surrData[jj][0] = this->iars(jj,ii);
                    surrData[jj][1] = this->pops(jj,ii);
                    surrSDData[jj][0] = this->iars(jj,ii);
                    surrSDData[jj][1] = this->popsSD(jj,ii);
                }

                // Surrogate error data
                for (int jj = 0; jj <= this->generation; jj++) {
                    surrErrData[jj][0] = jj;
                    surrErrData[jj][1] = this->surrFit(jj,ii);
                }

                // Surrogate model
                double minIAR = 0;
                double maxIAR = this->iars.col(ii).maxCoeff();

                Eigen::VectorXd iarsTemp = Eigen::VectorXd::LinSpaced(200,
                        minIAR,maxIAR);
                std::vector<std::vector<double>> surrModel;
                surrModel.resize(200,std::vector<double>(2));

                if (this->interp == Optimiser::CUBIC_SPLINE) {
                    for (int jj = 0; jj < 200; jj++) {
                        surrModel[jj][0] = iarsTemp(jj);

                        surrModel[jj][1] = alglib::spline1dcalc(this->
                                surrogate[2*this->scenario->
                                getCurrentScenario()][this->scenario->getRun()]
                                [ii],iarsTemp(jj));
                    }

                } else if (this->interp == Optimiser::MULTI_LOC_LIN_REG) {
                    Eigen::VectorXd results(200);

                    SimulateGPU::interpolateSurrogateMulti(this->surrogateML[
                            2*this->scenario->getCurrentScenario()][this->
                            scenario->getRun()][ii],iarsTemp,results,this->
                            surrDimRes,1);

                    for (int jj = 0; jj < 200; jj++) {
                        surrModel[jj][0] = iarsTemp(jj);
                        surrModel[jj][1] = results(jj);
//                        std::cout << surrModel[jj][0] << " " << surrModel[jj][1] << std::endl;
                    }
                }

                // Plot 4ii (Surrogate Model)
                (*this->surrPlotHandle) << "set title 'Surrogate Model, Species"
                        << ii+1 << "'\n";
                (*this->surrPlotHandle) << "unset key\n";
                (*this->surrPlotHandle) << "set grid\n";
                (*this->surrPlotHandle) << "set xlabel 'IAR'\n";
                (*this->surrPlotHandle) << "set ylabel 'End Population'\n";
                (*this->surrPlotHandle) << "set xrange [0:" + std::to_string(
                        maxIAR) + "]\n";
                (*this->surrPlotHandle) << "set yrange [0:" + std::to_string(
                        this->pops.col(ii).maxCoeff()) + "]\n";
                (*this->surrPlotHandle) << "plot '-' with lines, '-' with points pointtype 7 \n";
                (*this->surrPlotHandle).send1d(surrModel);
                (*this->surrPlotHandle).send1d(surrData);

                // Plot 5ii (Surrogate Model Error)
                (*this->surrPlotHandle) << "set title 'Surrogate Error'\n";
                (*this->surrPlotHandle) << "unset key\n";
                (*this->surrPlotHandle) << "set grid\n";
                (*this->surrPlotHandle) << "set xlabel 'Generation'\n";
                (*this->surrPlotHandle) << "set ylabel 'Model Error'\n";
                //(*this->surrPlotHandle) << "set logscale y\n";
                (*this->surrPlotHandle) << "set xrange [0:" + std::to_string(
                        this->generations) + "]\n";
                (*this->surrPlotHandle) << "set yrange [" + std::to_string(this
                        ->surrFit.block(0,ii,this->noSamples,1).minCoeff()/2) +
                        ":" + std::to_string(this->surrFit.col(ii).maxCoeff())
                        + "]\n";
                (*this->surrPlotHandle) << "plot '-' with points pointtype 7 \n";
                (*this->surrPlotHandle).send1d(surrErrData);
            }
        }

        (*this->plothandle) << "unset multiplot\n";
    }
}

void RoadGA::assignBestRoad() {

    // For some reason, igl does not allow passing a vector as the template
    // does not state the first input as a derived type
    Eigen::Matrix<double,Eigen::Dynamic,1> Y(1);
    Eigen::Matrix<int,Eigen::Dynamic,1> I(1);
    Eigen::Matrix<double,Eigen::Dynamic,Eigen::Dynamic> costs(
            this->populationSizeGA,1);
    costs = this->costs.cast<double>();

    igl::mat_min(costs,1,Y,I);

    RoadPtr road(new Road(this->me(),this->currentRoadPopulation.row(I(0))));

    this->bestRoads[this->scenario->getCurrentScenario()][this->scenario->
            getRun()] = road;
}

void RoadGA::output() {

    this->best(this->generation) = this->costs.minCoeff();
    this->av(this->generation) = this->costs.mean();
    this->surrFit.row(this->generation) = this->surrErr.transpose();
    // Include code to plot using gnuplot utilising POSIX pipes to plot the
    // best road
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
                    std::min((this->surrErr.maxCoeff() - this->surrThresh)*(
                    (double)(this->surrErr.maxCoeff() > this->surrThresh))*
                    this->maxLearnNo,1.0),10.0/this->populationSizeGA);
        } else {
            pFull = std::max((this->maxLearnNo/this->populationSizeGA)*
                    std::min((this->surrErr.maxCoeff() - this->surrThresh)*(
                    (double)(this->surrErr.maxCoeff() > this->surrThresh))*
                    this->maxLearnNo,1.0),3.0/this->populationSizeGA);
        }
    } else if (this->type == Optimiser::CONTROLLED) {
        // We treat ROV the same as MTE for now even though it should be much
        // much slower (we may have to resort to ignoring forward path
        // recomputation to improve speed at the expense of accuracy.
        if (this->generation < this->learnPeriod) {
            pFull = std::max((this->maxLearnNo/this->populationSizeGA)*
                    std::min((this->surrErr.maxCoeff() - this->surrThresh)*(
                    (double)(this->surrErr.maxCoeff() > this->surrThresh))*
                    this->maxLearnNo,1.0/this->populationSizeGA),10.0/this->
                    populationSizeGA);
        } else {
            pFull = std::max((this->maxLearnNo/this->populationSizeGA)*
                    std::min((this->surrErr.maxCoeff() - this->surrThresh)*(
                    (double)(this->surrErr.maxCoeff() > this->surrThresh))*
                    this->maxLearnNo,1.0/this->populationSizeGA),3.0/this->
                    populationSizeGA);
        }
    }

    // Actual number of roads to sample
    int newSamples = ceil(pFull * this->populationSizeGA);
    Eigen::VectorXi sampleRoads(newSamples);

//    // Select individuals for computing learning from the full model
//    // We first sort the roads by cost
//    Eigen::VectorXi sortedIdx(this->populationSizeGA);
//    Eigen::VectorXd sorted(this->populationSizeGA);
//    igl::sort(this->costs,1,true,sorted,sortedIdx);

//    // We will always test the three best roads
//    sampleRoads.segment(0,3) = sortedIdx.segment(0,3);

//    // Now fill up the test road indices
//    // N.B. NEED TO PUT IN A ROUTINE HERE THAT SELECTS ROADS THAT HAVE DESIGN
//    // POINTS THAT ARE MOST DISSIMILAR FROM THE REST OF THE POPULATION
//    // CURRENTLY IN THE LIST OF SURROGATE DATA POINTS.
//    std::random_shuffle(sortedIdx.data()+3,sortedIdx.data() +
//            this->populationSizeGA);
//    sampleRoads.segment(3,newSamples-3) = sortedIdx.segment(3,
//            newSamples-3);

    if (this->type == Optimiser::MTE) {
        Eigen::MatrixXd current(this->noSamples,this->species.size());
        Eigen::MatrixXd candidates(this->populationSizeGA,this->species
                .size());

        for (int ii = 0; ii < this->species.size(); ii++) {
            current.block(0,ii,this->noSamples,1) = this->iars.block(0,ii,
                    this->noSamples,1);

            candidates.block(0,ii,this->populationSizeGA,1) = this->iarsCurr
                    .block(0,ii,this->populationSizeGA,1);
        }

        this->generateSample(current,candidates,newSamples,sampleRoads);

    } else if (this->type == Optimiser::CONTROLLED) {
        Eigen::MatrixXd current(this->noSamples,this->species.size()+1);
        Eigen::MatrixXd candidates(this->populationSizeGA,this->species
                .size()+1);

        for (int ii = 0; ii < this->species.size(); ii++) {
            current.block(0,ii,this->noSamples,1) = this->iars.block(0,ii,
                    this->noSamples,1);

            candidates.block(0,ii,this->populationSizeGA,1) = this->iarsCurr
                    .block(0,ii,this->populationSizeGA,1);
        }

        current.block(0,species.size(),this->noSamples,1) = this->use.segment(
                0,this->noSamples);
        candidates.block(0,species.size(),this->populationSizeGA,1) = this->
                useCurr;

        this->generateSample(current,candidates,newSamples,sampleRoads);
    }

    // Call the thread pool. The computed function and form of the surrogate
    // models are different under each scenario (MTE vs CONTROLLED). The thread
    // pool is called WITHIN the functions to compute the surrogate data.
    if (this->type == Optimiser::MTE) {

        int validCounter = 0;
        Eigen::VectorXd currErr = Eigen::VectorXd::Zero(this->species.size());

        for (unsigned long ii = 0; ii < newSamples; ii++) {
            // As each full model requires its own parallel routine and this
            // routine is quite intensive, we call the full model for each
            // road serially. Furthermore, we do not want to add new roads to
            // the threadpool
            //      endPop_i = f_i(aar_i)
            RoadPtr road(new Road(this->me(),this->currentRoadPopulation
                    .row(sampleRoads(ii))));
            Eigen::MatrixXd mteResult(3,this->species.size());
            Optimiser::ComputationStatus fullStatus = this->
                    surrogateResultsMTE(road,mteResult);

            if (fullStatus == Optimiser::COMPUTATION_SUCCESS) {
                this->iars.row(this->noSamples + validCounter) = mteResult
                        .row(0);
                this->pops.row(this->noSamples + validCounter) = mteResult
                        .row(1);
                this->popsSD.row(this->noSamples + validCounter) = mteResult
                        .row(2);

                // We need to compute the current period error by evaluating the
                // sample roads against the new surrogate. We compute the error
                // of the mean population. For now we just compute the error of
                // the current samples, not all that have been computed to date
                // (although we could easily do this if desired). We use the
                // old surrogate here for the comparison.
                road->evaluateRoad();

                for (int jj = 0; jj < this->species.size(); jj++) {
                    currErr(jj) += pow((pops(this->noSamples + validCounter,jj)
                            - this->popsCurr(sampleRoads(ii),jj))/pops(this->
                            noSamples + validCounter,jj),2);

//                    std::cout << sampleRoads(ii) << " " << this->pops(this->noSamples + validCounter,jj) << " "
//                            << this->popsCurr(sampleRoads(ii),jj) << " "
//                            << road->getAttributes()->getEndPopMTE()(jj) << " "
//                            << this->iars(this->noSamples + validCounter,jj) << " "
//                            << this->iarsCurr(sampleRoads(ii),jj) << " "
//                            << road->getAttributes()->getIAR()(jj)
//                            << std::endl;

//                    std::cout << this->iars(this->noSamples + validCounter,jj) << std::endl;

//                    std::cout << this->popsCurr(sampleRoads(ii),jj) << std::endl;

//                    // Try 1
//                    road->evaluateRoad();
//                    std::cout << road->getAttributes()->getIAR()(jj) << " " << road->getAttributes()->getTotalValueMean() << std::endl;
                }
                validCounter++;
            }
        }

        for (int ii = 0; ii < this->species.size(); ii++) {
            currErr(ii) = sqrt(currErr(ii)/validCounter);
        }

        this->surrFit.block(this->generation,0,1,currErr.cols()) =currErr
                .transpose();

        this->noSamples += validCounter;

        // Now that we have the results, let's build the new surrogate model!!!
        if (this->interp == Optimiser::CUBIC_SPLINE) {
            // Using cubic spline interpolation
            this->buildSurrogateModelMTE();
        } else if (this->interp == Optimiser::MULTI_LOC_LIN_REG) {
            // Using local linear regression
            this->buildSurrogateModelMTELocalLinear();
        }

    } else if (this->type == Optimiser::CONTROLLED) {

        for (unsigned long ii = 0; ii < newSamples; ii++) {

            // As each full model requires its own parallel routine and this
            // routine is quite intensive, we call the full model for each
            // road serially. Furthermore, we do not want to add new roads to
            // the threadpool
            //      endPop_i = f_i(aar_i)
            RoadPtr road(new Road(this->me(),
                    this->currentRoadPopulation.row(ii)));
            Eigen::MatrixXd rovResult(1,this->species.size()+2);
            this->surrogateResultsROVCR(road,rovResult);

            this->iars.row(this->noSamples + ii) = rovResult.block(0,2,
                    1,this->species.size());
            this->use(this->noSamples + ii) = rovResult(0,0);
            this->useSD(this->noSamples + ii) = rovResult(0,1);
        }

        this->noSamples += newSamples;

        // Now that we have the results, let's build the surrogate model!!!
        this->buildSurrogateModelROVCR();
    }
}

void RoadGA::evaluateGeneration() {
    // Initialise the current vectors used to zero
    this->costs = Eigen::VectorXd::Zero(this->populationSizeGA);
    this->profits = Eigen::VectorXd::Zero(this->populationSizeGA);
    this->iarsCurr = Eigen::MatrixXd::Zero(this->populationSizeGA,
            this->species.size());
    this->popsCurr = Eigen::MatrixXd::Zero(this->populationSizeGA,
            this->species.size());
    this->useCurr = Eigen::VectorXd::Zero(this->populationSizeGA);

    if (this->type == Optimiser::CONTROLLED) {
        this->profits = Eigen::VectorXd::Zero(this->populationSizeGA);
    }

    // Computes the current population of roads using a surrogate function
    // instead of the full model where necessary
    ThreadManagerPtr threader = this->getThreadManager();
    unsigned long roads = this->getGAPopSize();

    std::vector<std::future<void>> results(roads);

    if (threader != nullptr) {
        // If multithreading is enabled
        for (unsigned long ii = 0; ii < roads; ii++) {
            // Push onto the pool with a lambda expression
            results[ii] = threader->push([this,ii](int id){
                RoadPtr road(new Road(this->me(),
                        this->currentRoadPopulation.row(ii)));
                road->designRoad();
                road->evaluateRoad();

                this->costs(ii) = road->getAttributes()->getTotalValueMean();
                this->profits(ii) = road->getAttributes()->getVarProfitIC();

                if (this->type > Optimiser::SIMPLEPENALTY) {
                    for (int jj = 0; jj < this->species.size(); jj++) {
                        if (this->type == Optimiser::MTE) {
                            this->iarsCurr(ii,jj) = road->getAttributes()->
                                    getIAR()(jj);
                            this->popsCurr(ii,jj) = road->getAttributes()->
                                    getEndPopMTE()(jj);
                        } else if (this->type == Optimiser::CONTROLLED) {
                            // We only need the aar at the highest flow rate
                            int controls = this->programs[this->scenario->
                                    getProgram()]->getFlowRates().size();
                            this->iarsCurr(ii,jj) = road->getAttributes()->
                                    getIAR()(jj,controls-1);
                            this->useCurr(ii) = road->getAttributes()->
                                    getInitialUnitCost();
                        }
                    }
                }
            });
        }

        for (unsigned long ii = 0; ii < roads; ii++) {
            results[ii].get();
        }

    } else {
        // Run serially
        for (unsigned long ii = 0; ii < roads; ii++) {
            RoadPtr road(new Road(this->me(),
                    this->currentRoadPopulation.row(ii)));
//            clock_t begin = clock();
            road->designRoad();
            road->evaluateRoad();
//            clock_t end = clock();
//            double elapsed_secs = double(end - begin) / CLOCKS_PER_SEC;
//            std::cout << "Road Design & Evaluation Time " << ii << ": " << elapsed_secs << " s" << std::endl << std::endl;

            this->costs(ii) = road->getAttributes()->getTotalValueMean();
            this->profits(ii) = road->getAttributes()->getVarProfitIC();

            if (this->type > Optimiser::SIMPLEPENALTY) {
                for (int jj = 0; jj < this->species.size(); jj++) {
                    if (this->type == Optimiser::MTE) {
                        this->iarsCurr(ii,jj) = road->getAttributes()->
                                getIAR()(jj);
                        this->popsCurr(ii,jj) = road->getAttributes()->
                                getEndPopMTE()(jj);
                    } else if (this->type == Optimiser::CONTROLLED) {
                        // We only need the aar at the highest flow rate
                        int controls = this->programs[this->scenario->
                                getProgram()]->getFlowRates().size();
                        this->iarsCurr(ii,jj) = road->getAttributes()->
                                getIAR()(jj,controls-1);
                        this->useCurr(ii) = road->getAttributes()->
                                getInitialUnitCost();
                    }
                }
            }
        }
    }
}

void RoadGA::scaling(RoadGA::Scaling scaleType, Eigen::VectorXi& parents,
        Eigen::VectorXd& scaling) {
    // Select individuals for computing learning from the full model
    // We first sort the roads by cost
    igl::sort(this->costs,1,true,scaling,parents);

    switch (scaleType) {
    case RoadGA::RANK:
    {
        Eigen::VectorXd ranks = Eigen::VectorXd::LinSpaced(parents.size(),
                1,parents.size());
        scaling = 1.0 / (ranks.array() + 1);
        break;
    }
    case RoadGA::PROPORTIONAL:
    {
        // This requires adding the difference between the best and worst roads
        // to all costs to make the numbers positive.
        double diff = this->costs.maxCoeff() - this->costs.minCoeff();
        scaling.array() += diff;
        scaling = 1.0 / (scaling.array());
        break;
    }
    case RoadGA::TOP:
    {
        // Only the top proportion of individuals remain
        unsigned long noParents = this->maxSurvivalRate*this->populationSizeGA;

        scaling.segment(0,noParents) = Eigen::VectorXd::Constant(noParents,1);
        scaling.segment(noParents,scaling.size() - noParents) =
                Eigen::VectorXd::Constant(scaling.size() - noParents, 0);
        break;
    }
    case RoadGA::SHIFT:
    {
        double maxScore = this->costs.maxCoeff();
        double minScore = this->costs.minCoeff();
        double meanScore = this->costs.mean();

        if (maxScore == minScore) {
            scaling = Eigen::VectorXd::Constant(scaling.size(),1);
        } else {
            double desiredMean = 1.0;

            double scale = desiredMean*(this->maxSurvivalRate - 1)/(maxScore -
                    meanScore);

            double offset = desiredMean - (scale*meanScore);

            if (offset + scale*minScore < 0) {
                scale = desiredMean/(meanScore - minScore);
                offset = desiredMean - (scale*meanScore);
            }

            scaling = offset + scale*scaling.array();
        }
        break;
    }
    default:
        break;
    }

    // Adjust the scaling vector so that its sum is equal to the number of
    // elements
    double factor = this->populationSizeGA/scaling.sum();
    scaling = scaling*factor;
}

void RoadGA::selection(Eigen::VectorXi& pc, Eigen::VectorXi& pm,
        Eigen::VectorXi& pe, RoadGA::Selection selector) {

    // Random number generator for random
    //std::default_random_engine generator;
    std::uniform_real_distribution<double> randomVal(0,1);
    std::uniform_int_distribution<unsigned long> randomParent(0,
            this->populationSizeGA-1);

    // The fit scalings correspond the ordered list of parents. We call the
    // scaling function to find the best parents (in order of increasing cost)
    // and the corresponding scaling, as determined by the scaling routine we
    // use.
    Eigen::VectorXi orderedParents(this->populationSizeGA);
    Eigen::VectorXd fitScalings(this->populationSizeGA);
    this->scaling(this->fitScaling, orderedParents, fitScalings);

    unsigned long eliteParents = pe.rows();

    // Elite parents selection is easy
    if (eliteParents > 0) {
        pe = orderedParents.segment(0,eliteParents);
    }

    // Now select parents for crossover and mutation
    switch (selector) {

    case RoadGA::STOCHASTIC_UNIFORM:
    {
        Eigen::VectorXd wheelBase(this->populationSizeGA);

        igl::cumsum(fitScalings,1,wheelBase);

        // First, crossover
        {
            Eigen::VectorXd wheel = wheelBase/(wheelBase(wheelBase.size()-1));
            double stepSize = 1.0/(((double)pc.size()) + 1.0);

            std::uniform_real_distribution<double> start(0,stepSize);
            double position = start(generator);

            unsigned long lowest = 0;

            for (unsigned long ii = 0; ii < pc.size(); ii++) {
                for (unsigned long jj = lowest; jj < wheelBase.size(); jj++) {
                    if (position < wheel(jj)) {
                        pc(ii) = orderedParents(jj);
                        lowest = jj;
                        break;
                    }
                }
                position += stepSize;
            }
        }

        // Shuffle the values randomly
        std::random_shuffle(pc.data(),pc.data()+pc.size());

        // Next, mutation
        {
            Eigen::VectorXd wheel = wheelBase/(wheelBase(wheelBase.size()-1));
            double stepSize = 1.0/(((double)pm.size()) + 1.0);

            std::uniform_real_distribution<double> start(0,stepSize);
            double position = start(generator);

            unsigned long lowest = 0;

            for (unsigned long ii = 0; ii < pm.size(); ii++) {
                for (unsigned long jj = lowest; jj < wheelBase.size(); jj++) {
                    if (position < wheel(jj)) {
                        pm(ii) = orderedParents(jj);
                        lowest = jj;
                        break;
                    }
                }
                position += stepSize;
            }

            // Shuffle the values randomly
            std::random_shuffle(pm.data(),pm.data()+pm.size());
        }
        break;
    }
    case RoadGA::REMAINDER:
    {
        // We need to keep the scalings vector for both crossover and mutation
        Eigen::VectorXd fitScalingsTemp = fitScalings;

        // First, crossover
        {
            unsigned long next = 0;

            // First we assign the integral parts deterministically.
            // Load up the sure parents and leave the fractional
            // remainder in newScores.
            for (unsigned long ii = 0; ii < orderedParents.size(); ii++) {
                if (next >= pc.size()) {
                    break;
                }
                while (fitScalings(ii) >= 1.0) {
                    if (next >= pc.size()) {
                        break;
                    }
                    pc(next) = orderedParents(ii);
                    next++;
                    fitScalings(ii) -= 1.0;
                }
            }

            // If all new scores were integers, we are done
            if (next < pc.size()) {
                // Scale the remaining scores to be probabilities
                Eigen::VectorXd intervals(this->populationSizeGA);

                igl::cumsum(fitScalings,1,intervals);
                intervals = intervals / intervals(intervals.size()-1);

                // Now use the remainders as probabilities
                for (unsigned long ii = next; ii < pc.size(); ii++) {
                    double r = randomVal(generator);
                    for (unsigned long jj = 0; jj < fitScalings.size(); jj++) {
                        if (r <= intervals(ii)) {
                            pc(ii) = orderedParents(jj);

                            // Make sure this is not picked again
                            fitScalings(jj) = 0;
                            igl::cumsum(fitScalings,1,intervals);

                            if (intervals(intervals.size()-1) != 0.0) {
                                intervals = intervals / intervals(intervals.size()
                                        - 1);
                            }
                            break;
                        }
                    }
                }
            }
        }

        // Randomise the order of the parents
        std::random_shuffle(pc.data(),pc.data()+pc.size());

        // Next, mutation
        {
            fitScalings = fitScalingsTemp;
            unsigned long next = 0;

            // First we assign the integral parts deterministically.
            // Load up the sure parents and leave the fractional
            // remainder in newScores.
            for (unsigned long ii = 0; ii < orderedParents.size(); ii++) {
                if (next >= pm.size()) {
                    break;
                }
                while (fitScalings(ii) >= 1.0) {
                    if (next >= pm.size()) {
                        break;
                    }
                    pm(next) = orderedParents(ii);
                    next++;
                    fitScalings(ii) -= 1.0;
                }
            }

            // If all new scores were integers, we are done
            if (next < pm.size()) {
                // Scale the remaining scores to be probabilities
                Eigen::VectorXd intervals(this->populationSizeGA);

                igl::cumsum(fitScalings,1,intervals);
                intervals = intervals / intervals(intervals.size()-1);

                // Now use the remainders as probabilities
                for (unsigned long ii = next; ii < pc.size(); ii++) {
                    double r = randomVal(generator);
                    for (unsigned long jj = 0; jj < fitScalings.size(); jj++) {
                        if (r <= intervals(ii)) {
                            pm(ii) = orderedParents(jj);

                            // Make sure this is not picked again
                            fitScalings(jj) = 0;
                            igl::cumsum(fitScalings,1,intervals);

                            if (intervals(intervals.size()-1) != 0.0) {
                                intervals = intervals / intervals(intervals.size()
                                        - 1);
                            }
                            break;
                        }
                    }
                }
            }
        }
        std::random_shuffle(pm.data(),pm.data()+pm.size());
        break;
    }
    case RoadGA::ROULETTE:
    {
        Eigen::VectorXd wheelBase(this->populationSizeGA);

        igl::cumsum(fitScalings,1,wheelBase);

        // Crossover
        {
            for (unsigned long ii = 0; ii < pc.size(); ii++) {
                double r = randomVal(generator)*this->populationSizeGA;

                for (unsigned long jj = 0; jj < wheelBase.size(); jj++) {
                    if (r < wheelBase(jj)) {
                        pc(ii) = orderedParents(jj);
                        break;
                    }
                }
            }
        }

        // Mutation
        {
            for (unsigned long ii = 0; ii < pm.size(); ii++) {
                double r = randomVal(generator)*this->populationSizeGA;

                for (unsigned long jj = 0; jj < wheelBase.size(); jj++) {
                    if (r < wheelBase(jj)) {
                        pm(ii) = orderedParents(jj);
                        break;
                    }
                }
            }
        }
        break;
    }
    case RoadGA::TOURNAMENT:
    {
        Eigen::MatrixXi playerList(pc.size()+pm.size(), this->tournamentSize);

        // The player list refers to the index in 'orderedParents' and not to
        // the original parent number. I.e. a player value of 20 indicates the
        // 20th lowest cost parent as stored in 'orderedParents' and not the
        // 20th parent in the current population list.
        //
        // This is fine as the tournament pools of parents are random anyway.
        for (unsigned long ii = 0; ii < pc.size()+pm.size(); ii++) {
            for (unsigned int jj = 0; jj < this->tournamentSize; jj++) {
                playerList(ii,jj) = this->populationSizeGA*randomVal(generator);
            }
        }

        Eigen::MatrixXi players(1,this->tournamentSize);
        // Crossover parents
        for (unsigned long ii = 0; ii < pc.size(); ii++) {
            players = playerList.row(ii);
            // For each tournament
            unsigned long winner = players(0);
            for (int jj = 1; jj < players.size(); jj++) {
                double score1 = fitScalings(winner);
                double score2 = fitScalings(players(jj));
                if (score2 > score1) {
                    // The problem is single-objective so we only have one
                    // score per road
                    winner = players(jj);
                }
            }
            pc(ii) = orderedParents(winner);
        }

        // Mutation parents
        for (unsigned long ii = 0; ii < pm.size(); ii++) {
            players = playerList.row(ii+pc.size());
            // For each tournament
            unsigned long winner = players(0);
            for (int jj = 1; jj < players.size(); jj++) {
                double score1 = fitScalings(winner);
                double score2 = fitScalings(players(jj));
                if (score2 > score1) {
                    // The problem is single-objective so we only have one
                    // score per road
                    winner = players(jj);
                }
            }
            pm(ii) = orderedParents(winner);
        }
        break;
    }
    case RoadGA::UNIFORM:
    {
        // Do not use for actual convergence of the GA
        // Crossover parents
        {
            for (unsigned long ii = 0; ii < pc.size(); ii++) {
                pc(ii) = randomParent(generator);
            }
        }

        // Mutation parents
        {
            for (unsigned long ii = 0; ii < pm.size(); ii++) {
                pm(ii) = randomParent(generator);
            }
        }
        break;
    }
    default:
    {
        break;
    }
    }
}

int RoadGA::stopCheck() {
    if (this->generation >= this->generations) {
        // Adequate convergence not achieved. Generations exceeded
        std::cout << "Optimisation ended: maximum number of generations reached."
                  << std::endl;
        return -1;
    }

    if (this->stallTest()) {

        if (this->stallGen >= this->stallGenerations) {
            // Number of stall generations exceeded
            std::cout << "Optimisation ended: maximum number of stall generations reached."
                    << std::endl;
            return -2;
        }

        this->stallGen++;

    } else {
        this->stallGen = 0;
    }

    // Is the error of the surrogate model acceptable?
    bool surrGood = true;

    if (this->type == Optimiser::MTE) {
        for (int ii = 0; ii < this->species.size(); ii++) {
            if (surrErr(ii) > surrThresh) {
                surrGood = false;
                break;
            }
        }

    } else if (this->type = Optimiser::CONTROLLED) {
        if (surrErr(0) > surrThresh) {
            surrGood = false;
        }
    }

    if (surrGood) {
        // Also covers the case where we do not need a surrogate

        // If the best road has not changed materially for five generations
        if (this->generation > this->learnPeriod) {
            if ((abs((this->best(this->generation) - this->best(
                    this->generation - 1))/(this->best(
                    this->generation - 1))) < this->stoppingTol) &&
                    (abs((this->best(this->generation) - this->best(
                    this->generation - 2))/(this->best(this->generation - 2)))
                    < this->stoppingTol) && (abs((this->best(this->generation)
                    - this->best(this->generation - 3))/(this->best(
                    this->generation - 3))) < this->stoppingTol) &&
                    (abs((this->best(this->generation) - this->best(
                    this->generation - 4))/(this->best(this->generation - 4)))
                    < this->stoppingTol) && (abs((this->best(this->generation)
                    - this->best(this->generation - 5))/(this->best(
                    this->generation - 5))) < this->stoppingTol)) {
                // Solution successfully found
                std::cout << "Optimisation ended: stopping tolerance achieved."
                        << std::endl;

                return 1;

            } else {
                return 0;
            }
        } else {
            // We should perform a minimum number of generations
            return 0;
        }
    } else {
        // Continue computing to reduce the surrogate error
        return 0;
    }
}

bool RoadGA::stallTest() {
    // First, apply geometric weightings to all generations PRIOR to the
    // current

    if (this->generation > 0) {
        double total = 1.0;
        double totalWeight = 0.0;

        for (int ii = this->generation - 1; ii > 0; ii--) {
            double w = pow(0.5,this->generation - 1 - ii);
            total = total*pow(this->av(ii),w);
            totalWeight += w;
        }

        total = pow(total,1.0/totalWeight);

        if (abs(this->av(this->generation - 1) - total)/(total) <
                this->stoppingTol) {
            return true;
        } else {
            return false;
        }
    } else {
        return false;
    }
}

void RoadGA::randomXYOnPlanes(const long &individuals, const long&
        intersectPts, const long& startRow) {

    // As the equation is parametrised, the random matrix is the same for the
    // X and Y coordinates
    Eigen::MatrixXd randomMatrix = Eigen::MatrixXd::Random(individuals,
            intersectPts);
    randomMatrix = 0.5*(randomMatrix.array() + 1);
    Eigen::MatrixXd xvals = Eigen::MatrixXd::Zero(individuals,intersectPts);
    Eigen::MatrixXd yvals = Eigen::MatrixXd::Zero(individuals,intersectPts);

    xvals = ((this->xO.array() + this->dL.array()*cos(this->theta)).rowwise().
            replicate(individuals).transpose() + (randomMatrix.array()*
            (((this->dU - this->dL)*cos(theta)).rowwise().replicate(
            individuals).transpose().array()))).matrix();

    yvals = ((this->yO.array() + this->dL.array()*sin(this->theta)).rowwise().
            replicate(individuals).transpose() + (randomMatrix.array()*
            (((this->dU - this->dL)*sin(theta)).rowwise().replicate(
            individuals).transpose().array()))).matrix();

    // Place the genome values in the initial population matrix
    Eigen::VectorXi rowIdx = Eigen::VectorXi::LinSpaced(individuals,
            startRow,startRow + individuals - 1);
    Eigen::RowVectorXi colIdx = Eigen::RowVectorXi::LinSpaced(intersectPts,3,
            3*intersectPts);

    Utility::sliceIntoPairs(xvals,rowIdx.rowwise().replicate(intersectPts),
            colIdx.colwise().replicate(individuals),
            this->currentRoadPopulation);
    Utility::sliceIntoPairs(yvals,rowIdx.rowwise().replicate(intersectPts),
            (colIdx.array() + 1).colwise().replicate(individuals),
            this->currentRoadPopulation);
}

void RoadGA::randomXYinRegion(const long &individuals, const long
        &intersectPts, const long &startRow) {

    double minLon = (this->getRegion()->getX())(2,1);
    double maxLon = (this->getRegion()->getX())(this->getRegion()->getX().
            rows()-3,1);
    double minLat = (this->getRegion()->getY())(1,2);
    double maxLat = (this->getRegion()->getY())(1,this->getRegion()->getY().
            cols()-3);

    Eigen::MatrixXd randomMatrixX = Eigen::MatrixXd::Random(individuals,
            intersectPts);
    randomMatrixX = 0.5*(randomMatrixX.array() + 1);
    Eigen::MatrixXd randomMatrixY = Eigen::MatrixXd::Random(individuals,
            intersectPts);
    randomMatrixY = 0.5*(randomMatrixY.array() + 1);
    Eigen::MatrixXd xvals = Eigen::MatrixXd::Zero(individuals,intersectPts);
    Eigen::MatrixXd yvals = Eigen::MatrixXd::Zero(individuals,intersectPts);

    xvals = randomMatrixX.array()*(maxLon - minLon) + minLon;
    yvals = randomMatrixY.array()*(maxLat - minLat) + minLat;

    // Place the genome values in the initial population matrix
    Eigen::VectorXi rowIdx = Eigen::VectorXi::LinSpaced(individuals,
            startRow,startRow + individuals - 1);
    Eigen::RowVectorXi colIdx = Eigen::RowVectorXi::LinSpaced(intersectPts,3,
            3*intersectPts);

    Utility::sliceIntoPairs(xvals,rowIdx.rowwise().replicate(intersectPts),
            colIdx.colwise().replicate(individuals),
            this->currentRoadPopulation);
    Utility::sliceIntoPairs(yvals,rowIdx.rowwise().replicate(intersectPts),
            (colIdx.array() + 1).colwise().replicate(individuals),
            this->currentRoadPopulation);
}

void RoadGA::randomZWithinRange(const long &individuals, const long&
            intersectPts, const long& startRow, Eigen::MatrixXd& population) {

    double gmax = this->designParams->getMaxGrade();

    Eigen::MatrixXd sTemp = Eigen::MatrixXd::Zero(individuals,
            intersectPts + 2);

    for (long ii = 0; ii < individuals; ii++) {
        Eigen::RowVectorXd xtemp(intersectPts+2);
        Eigen::RowVectorXd ytemp(intersectPts+2);

        Eigen::VectorXi row = Eigen::VectorXi::Constant(1,ii + startRow);
        Eigen::VectorXi colIdx = Eigen::VectorXi::LinSpaced(intersectPts+2,0,
                intersectPts + 1);
        Eigen::VectorXi col = (3*colIdx.array()).transpose();

        igl::slice(population,row,col,xtemp);
        col = col.array() + 1;
        igl::slice(population,row,col,ytemp);

        RoadPtr road(new Road(this->me(),xtemp,ytemp,
                Eigen::VectorXd::Zero(intersectPts+2)));
        road->computeAlignment();

        sTemp.row(ii) = road->getVerticalAlignment()->getSDistances().
                transpose();
    }

    for (long ii = 0; ii < intersectPts; ii++) {
        Eigen::VectorXd randVec = Eigen::VectorXd::Random(individuals);
        randVec = 0.5*(randVec.array() + 1);

        Eigen::VectorXd zL = (population.block(startRow,
                3*ii+2,individuals,1) - (sTemp.col(ii+1) - sTemp.col(ii))*
                gmax/100).array().max((population.block(
                startRow,3*(intersectPts + 2) - 1,individuals,1) -
                (sTemp.col(intersectPts + 1) - sTemp.col(ii+1))*gmax/100).
                array()).matrix();

        Eigen::VectorXd zU = (population.block(startRow,
                3*ii+2,individuals,1) + (sTemp.col(ii+1) - sTemp.col(ii))*
                gmax/100).array().min((population.block(
                startRow,3*(intersectPts + 2) - 1,individuals,1) +
                (sTemp.col(intersectPts + 1) - sTemp.col(ii+1))*gmax/100).
                array()).matrix();

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
        Eigen::RowVectorXd xtemp(intersectPts + 2);
        Eigen::RowVectorXd ytemp(intersectPts + 2);

        Eigen::VectorXi row = Eigen::VectorXi::Constant(1, ii + startRow);
        Eigen::VectorXi colIdx = Eigen::VectorXi::LinSpaced(intersectPts + 2,0,
                intersectPts + 1);
        Eigen::VectorXi col = (3*colIdx.array()).transpose();

        igl::slice(population,row,col,xtemp);
        col = col.array() + 1;
        igl::slice(population,row,col,ytemp);

        RoadPtr road(new Road(this->me(),xtemp,ytemp,
                Eigen::VectorXd::Zero(intersectPts+2)));
        road->computeAlignment();

        sTemp.row(ii) = road->getVerticalAlignment()->getSDistances();
    }

    for (long ii = 0; ii < intersectPts; ii++) {

        Eigen::VectorXd zL = (population.block(startRow,
                3*ii+2,individuals,1) - (sTemp.col(ii+1) - sTemp.col(ii))*
                gmax/100).array().max((population.block(
                startRow,3*(intersectPts + 2) - 1,individuals,1) -
                (sTemp.col(intersectPts + 1) - sTemp.col(ii+1))*gmax/100).
                array()).matrix();

        Eigen::VectorXd zU = (population.block(startRow,
                3*ii+2,individuals,1) + (sTemp.col(ii+1) - sTemp.col(ii))*
                gmax/100).array().min((population.block(
                startRow,3*(intersectPts + 2) - 1,individuals,1) +
                (sTemp.col(intersectPts + 1) - sTemp.col(ii+1))*gmax/100).
                array()).matrix();

        Eigen::VectorXd zE(individuals);

        this->region->placeNetwork(population.block(
                startRow,3*(ii+1), individuals,1), population.
                block(startRow,3*(ii+1)+1, individuals,1), zE);

        Eigen::VectorXd selectGround = ((zE.array() <=
                zU.array()) && (zE.array() >= zL.array())).cast<double>();
        Eigen::VectorXd selectLower = (zE.array() < zL.array()).cast<double>();
        Eigen::VectorXd selectUpper = (zE.array() > zU.array()).cast<double>();

        population.block(startRow,3*(ii+1)+2,individuals,1) =
                ((zE.array()*selectGround.array()) + (zL.array()*selectLower
                .array()) + (zU.array()*selectUpper.array())).matrix();
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
    rows = ((invalid.rowwise().sum()).array() == 0).cast<int>();
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

    // Consider developing a new curve elimination procedure that essentially
    // removes all intervening points. This will allow larger radii of
    // curvature that is not artifically restrained by the virtual points
    // between jj and kk and kk and ll.
    if ((kk - jj) >= 2) {
        Eigen::VectorXd level = Eigen::VectorXd::LinSpaced(kk-jj-1,1,
                kk-jj-1);
        Eigen::MatrixXd xvals = (children(ii,3*jj) +
                (level*(children(ii,3*kk) - children(ii,3*jj))/
                ((double)(kk - jj))).array()).matrix();
        Eigen::VectorXi xIdxJ = Eigen::VectorXi::LinSpaced(kk-jj-1,3*(jj+1),
                3*(kk-1));
        Eigen::VectorXi IdxI = Eigen::VectorXi::Constant(kk-jj-1,ii);
        Utility::sliceIntoPairs(xvals,IdxI,xIdxJ,children);

        Eigen::MatrixXd yvals = (children(ii,3*jj+1) +
                (level*(children(ii,3*kk+1) - children(ii,3*jj+1))/
                ((double)(kk - jj))).array()).matrix();
        Eigen::VectorXi yIdxJ = Eigen::VectorXi::LinSpaced(kk-jj-1,
                3*(jj+1)+1,3*(kk-1)+1);
        Utility::sliceIntoPairs(yvals,IdxI,yIdxJ,children);
    }

    if ((ll - kk) >= 2) {
        Eigen::VectorXd level = Eigen::VectorXd::LinSpaced(ll-kk-1,1,
                ll-kk-1);
        Eigen::MatrixXd xvals = (children(ii,3*kk) +
                (level*(children(ii,3*ll) - children(ii,3*kk))/
                ((double)(ll - kk))).array()).matrix();
        Eigen::VectorXi xIdxJ = Eigen::VectorXi::LinSpaced(ll-kk-1,3*(kk+1),
                3*(ll-1));
        Eigen::VectorXi IdxI = Eigen::VectorXi::Constant(ll-kk-1,ii);
        Utility::sliceIntoPairs(xvals,IdxI,xIdxJ,children);

        Eigen::MatrixXd yvals = (children(ii,3*kk+1) +
                (level*(children(ii,3*ll+1) - children(ii,3*kk+1))/
                ((double)(ll - kk))).array()).matrix();
        Eigen::VectorXi yIdxJ = Eigen::VectorXi::LinSpaced(ll-kk-1,
                3*(kk+1)+1,3*(ll-1)+1);
        Utility::sliceIntoPairs(yvals,IdxI,yIdxJ,children);
    }
}

RoadGA::ComputationStatus RoadGA::surrogateResultsMTE(RoadPtr road,
        Eigen::MatrixXd& mteResult) {

    try {
        road->designRoad();
        road->evaluateRoad(true);
        std::vector<SpeciesRoadPatchesPtr> species =
                road->getSpeciesRoadPatches();

        Eigen::VectorXd initPops(this->species.size());

        for (int ii = 0; ii < this->species.size(); ii++) {
            // We normalise by the starting population
            initPops(ii) = species[ii]->getInitPop();
            Eigen::VectorXd iars = species[ii]->getInitAAR();
            mteResult(0,ii) = iars(iars.size()-1);
            mteResult(1,ii) = species[ii]->getEndPopMean()/initPops(ii);
            mteResult(2,ii) = species[ii]->getEndPopSD()/initPops(ii);
        }

        return RoadGA::COMPUTATION_SUCCESS;

    } catch (int err) {
        return RoadGA::COMPUTATION_FAILED;

        std::cout << "Failure in full model computation MTE" << std::endl;
    }
}

void RoadGA::surrogateResultsROVCR(RoadPtr road, Eigen::MatrixXd& rovResult) {

    road->designRoad();
    road->evaluateRoad(true);
    std::vector<SpeciesRoadPatchesPtr> species =
            road->getSpeciesRoadPatches();

    rovResult(0,1) = road->getAttributes()->getTotalUtilisationROV();
    rovResult(0,2) = road->getAttributes()->getTotalUtilisationROVSD();
    for (int ii = 0; ii < this->species.size(); ii++) {
        Eigen::VectorXd iars = species[ii]->getInitAAR();
        rovResult(0,ii+2) = iars(iars.size()-1);
    }
}

void RoadGA::defaultSurrogate() {

    if (this->type == Optimiser::MTE) {
        // Default is to assume no animals die for any road

        if (this->interp == Optimiser::CUBIC_SPLINE) {
            for (int ii = 0; ii < this->species.size(); ii++) {
                // The default surrogate is a straight line at the initial
                // population
                alglib::ae_int_t m = 100;
                // Amount to penalise non-linearity. We elect small
                // penalisation.
                double rho = 1.0;
                // Exit status of spline fitting
                alglib::ae_int_t info;

                // Convert sample data to usable form for ALGLIB
                alglib::real_1d_array inputX;
                alglib::real_1d_array inputY;

                Eigen::VectorXd abscissa = Eigen::VectorXd::LinSpaced(11,0,1);
                Eigen::VectorXd ordinate = Eigen::VectorXd::Constant(11,1.0);

                inputX.setcontent(abscissa.size(),abscissa.data());
                inputY.setcontent(ordinate.size(),ordinate.data());

                alglib::spline1dfitreport report;
                alglib::spline1dinterpolant s;
                this->surrogate[2*this->scenario->getCurrentScenario()][this->
                        scenario->getRun()][ii] = s;

                // Mean
                alglib::spline1dfitpenalized(inputX,inputY,m,rho,info,this->
                        surrogate[2*this->scenario->getCurrentScenario()][
                        this->scenario->getRun()][ii],report);
                // Standard deviation
                ordinate = Eigen::VectorXd::Zero(11);
                inputY.setcontent(ordinate.size(),ordinate.data());

                alglib::spline1dinterpolant s2;
                this->surrogate[2*this->scenario->getCurrentScenario()+1][
                        this->scenario->getRun()][ii] = s2;

                alglib::spline1dfitpenalized(inputX,inputY,m,rho,info,this->
                        surrogate[2*this->scenario->getCurrentScenario()+1][
                        this->scenario->getRun()][ii],report);
            }
        } else if (this->interp == Optimiser::MULTI_LOC_LIN_REG) {
            for (int ii = 0; ii < this->species.size(); ii++) {
                Eigen::VectorXd surrogate(this->surrDimRes*2);

                surrogate.segment(0,this->surrDimRes) =
                        Eigen::VectorXd::LinSpaced(this->surrDimRes,0,1);

                // Mean
                surrogate.segment(this->surrDimRes,this->surrDimRes) =
                        Eigen::VectorXd::Constant(this->surrDimRes,1.0);

                this->surrogateML[2*this->scenario->getCurrentScenario()][
                        this->scenario->getRun()][ii] = surrogate;

                // Standard deviation
                surrogate.segment(this->surrDimRes,this->surrDimRes) =
                        Eigen::VectorXd::Constant(this->surrDimRes,0.0);

                this->surrogateML[2*this->scenario->getCurrentScenario()+1][
                        this->scenario->getRun()][ii] = surrogate;
            }
        }

    } else if (this->type == Optimiser::CONTROLLED) {
        // Create the predictors
        Eigen::VectorXd surrogate(this->surrDimRes*(this->species.size()+1) +
                (int)pow(this->surrDimRes,this->species.size()+1));

        for (int ii = 0; ii < this->species.size(); ii++) {
            surrogate.segment(ii*this->surrDimRes,this->surrDimRes) =
                    Eigen::VectorXd::LinSpaced(this->surrDimRes,0,1);
        }

        surrogate.segment(this->surrDimRes*this->species.size(),this->
                surrDimRes) = Eigen::VectorXd::LinSpaced(this->surrDimRes,0,
                100);

        // Mean
        surrogate.segment(this->surrDimRes*(this->species.size()+1),
                (int)pow(this->surrDimRes,this->species.size()+1)) =
                Eigen::VectorXd::Constant((int)pow(this->surrDimRes,this->
                species.size()+1),0.0);

        this->surrogateML[2*this->scenario->getCurrentScenario()][this->
                scenario->getRun()][0] = surrogate;


        // Standard deviation
        surrogate.segment(this->surrDimRes*(this->species.size()+1),
                (int)pow(this->surrDimRes,this->species.size()+1)) =
                Eigen::VectorXd::Constant((int)pow(this->surrDimRes,this->
                species.size()+1),0.0);

        this->surrogateML[2*this->scenario->getCurrentScenario()+1][this->
                scenario->getRun()][0] = surrogate;
    }
}

void RoadGA::buildSurrogateModelMTE() {
    // This function takes in full model data and the generation and computes
    // two surrogate models that are used in the Road Fitness function to determine
    // the end populations (and standard deviations) based on species AARs.

    // Compute the window size
    int ww;
    if (this->noSamples < 50) {
        ww = 5;
    } else if (this->noSamples < 100) {
        ww = 7;
    } else if (this->noSamples < 500) {
        ww = std::ceil(0.02*(this->noSamples - 100) + 5);
    } else {
        ww = std::ceil(0.01*(this->noSamples - 500) + 15);
    }

    for (int ii = 0; ii < this->species.size(); ii++) {
        // Compute the surrogate model training points
        // Surrogate domain resolution is currently 100 basis functions/nodes
        // but this can be altered at a later stage to optimiser the number.
        alglib::ae_int_t m = 100;
        // Amount to penalise non-linearity.
        double rho = 1.0;
        // Exit status of spline fitting
        alglib::ae_int_t info;

        // Create the cubic splines and save them
        /*
        Utility::ksrlin_vw(this->iars.col(ii),this->pops.col(ii), ww, 100, rx,
                ry);
        */

        // Convert sample data to usable form for ALGLIB
        alglib::real_1d_array inputX;
        alglib::real_1d_array inputY;

        Eigen::VectorXd abscissa = this->iars.col(ii).cast<double>();
        Eigen::VectorXd ordinate = this->pops.col(ii).cast<double>();

        inputX.setcontent(this->noSamples,abscissa.data());
        inputY.setcontent(this->noSamples,ordinate.data());

        alglib::spline1dfitreport report;

        // We call the 1D penalised spline fitting routine of AGLIB to fit a
        // curve to the data. We will need to tweak this as we investigate the
        // method some more. Hopefully we can optimise the smoothing parameter.
        // If this is inadequate, we will revert to the locally-linear kernel
        // technique, which will require more code or another library. We use
        // the mean for each road as the input.

        // Mean
        alglib::spline1dfitpenalized(inputX,inputY,m,rho,info,this->surrogate[
                2*this->scenario->getCurrentScenario()][this->scenario->
                getRun()][ii],report);
        // Standard deviation
        ordinate = this->popsSD.col(ii).cast<double>();

        inputX.setcontent(this->noSamples,abscissa.data());
        inputY.setcontent(this->noSamples,ordinate.data());
        alglib::spline1dfitpenalized(inputX,inputY,m,rho,info,this->surrogate[
                2*this->scenario->getCurrentScenario()+1][this->scenario->
                getRun()][ii],report);
    }
}

void RoadGA::buildSurrogateModelMTELocalLinear() {
    // WRAPPER
    // This function takes in full model data and the generation and computes a
    // surrogate model for each species

    for (int ii = 0; ii < this->species.size(); ii++) {
        SimulateGPU::buildSurrogateMTECUDA(this->meDerived(),ii);
    }
}

void RoadGA::buildSurrogateModelROVCR() {
    // WRAPPER
    // This function takes in full model data and the generation and computes a
    // surrogate model that is used in the Road Fitness function to determine
    // the road utility based on species AARs.

    // Unlike MTE, we have a multivariate regression rather than multiple
    // single-variable regressions. Therefore, we do not use spline
    // interpolation but multiple local linear regression. This method is
    // better for dealing with fewer sample points than spline interpolation.
    //
    // In the future, we may need to consider clustering to develop a better
    // surrogate model.

    // Mean and standard deviation are done during the same call

    SimulateGPU::buildSurrogateROVCUDA(this->meDerived());
}

void RoadGA::generateSample(const Eigen::MatrixXd &current, const
        Eigen::MatrixXd &candidates, int n, Eigen::VectorXi &sample) {

    // We first normalise the augmented data
    Eigen::MatrixXd C(current.rows()+candidates.rows(),candidates.cols());
    C << current, candidates;

    Eigen::VectorXd maxes = C.colwise().maxCoeff();
    Eigen::VectorXd mins = C.colwise().minCoeff();

    // Normalise in each dimension
    for (int ii = 0; ii < candidates.cols(); ii++) {
        C.col(ii) = (C.col(ii).array() - mins(ii))/(maxes(ii) - mins(ii));
    }

    // Return the values to the current and candidate matrices
    Eigen::MatrixXf currf = C.block(0,0,current.rows(),candidates.cols()).
            cast<float>();
    Eigen::MatrixXf candf = C.block(current.rows(),0,candidates.rows(),
            candidates.cols()).cast<float>();

    // If it is the first generation, We start by first selecting the two most
    // extreme points in the region (the points furthest from each other).
    if (current.size() == 0) {
        currf.resize(2,candidates.cols());
        {
            Eigen::MatrixXf distsInf(candidates.rows(),candidates.rows());

            // We need to compute all distances (only projected distance here)
            knn_cuda_with_indexes::computeDistances(candf.data(),candidates.rows(),
                    candf.data(),candidates.rows(),candidates.cols(),distsInf
                    .data(),true,-1);

            float maxDist = 0.0f;
            int P1 = 0;
            int P2 = 0;

            // Get pair with max distance between them
            for(int ii = 0; ii < candidates.rows(); ii++) {
                for(int jj = 0; jj < candidates.rows(); jj++) {
                    if (distsInf(ii,jj) > maxDist) {
                        P1 = ii;
                        P2 = jj;
                        maxDist = distsInf(ii,jj);
                    }
                }
            }

            sample(0) = P1;
            sample(1) = P2;

            currf.row(0) = candf.row(P1).cast<float>();
            currf.row(1) = candf.row(P2).cast<float>();
        }

        int noSamps = 2;

        // Compute the remaining samples
        for (int ii = 2; ii < n; ii++) {
            // We first need to determine which points are in the threshold
            // region
            Eigen::MatrixXf distsInf(candidates.rows(),currf.rows());
            Eigen::MatrixXf dists2(candidates.rows(),currf.rows());

            knn_cuda_with_indexes::computeDistances(currf.data(),noSamps,
                    candf.data(),candidates.rows(),candidates.cols(),distsInf
                    .data(),true,-1);

            knn_cuda_with_indexes::computeDistances(currf.data(),noSamps,
                    candf.data(),candidates.rows(),candidates.cols(),dists2
                    .data(),false,2);

            // Find the minimum distance in each distance for each candidate
            Eigen::MatrixXf minDI(1,candidates.rows());
            Eigen::MatrixXf minD2(1,candidates.rows());

            minDI = distsInf.rowwise().minCoeff();
            minD2 = dists2.rowwise().minCoeff();

            // Use these to compute the intersite distance
            float d_min = 2*0.5/(float)noSamps;
            float best = 0.0f;
            bool success = false;

            for (int jj = 0; jj < candidates.rows(); jj++) {
                if (minDI(jj) >= d_min) {
                    success = true;
                    if (minD2(jj) > best) {
                        sample(noSamps) = jj;
                        best = minD2(jj);
                    }
                }
            }

            // If there are no points over the projected distance threshold we
            // resort to using the best of the intersite distances.
            for (int jj = 0; jj < candidates.rows(); jj++) {
                float val = ((pow(noSamps + 1,1/candidates.cols()) - 1)/2)*
                        minD2(jj) + ((noSamps + 1)/2)*minDI(jj);

                if (val > best) {
                    sample(noSamps) = jj;
                    best = val;
                }
            }

            Eigen::MatrixXf currtemp = currf;
            currf.resize(noSamps+1,candidates.cols());
            currf.block(0,0,noSamps,candidates.cols()) = currtemp;

            currf.row(noSamps) = candf.row(sample(noSamps)).cast<float>();
            noSamps++;
        }
    } else {
        // We first compute which points are within the permissible regions and
        // then take the one with the maximum minimum distance
        int noSamps = current.rows();

        for (int ii = 0; ii < n; ii++) {
            // We first need to determine which points are in the threshold
            // region
            Eigen::MatrixXf distsInf(candidates.rows(),currf.rows());
            Eigen::MatrixXf dists2(candidates.rows(),currf.rows());

            knn_cuda_with_indexes::computeDistances(currf.data(),noSamps,
                    candf.data(),candidates.rows(),candidates.cols(),distsInf
                    .data(),true,-1);

            knn_cuda_with_indexes::computeDistances(currf.data(),noSamps,
                    candf.data(),candidates.rows(),candidates.cols(),dists2
                    .data(),false,2);

            // Find the minimum distance in each distance for each candidate
            Eigen::MatrixXf minDI(1,candidates.rows());
            Eigen::MatrixXf minD2(1,candidates.rows());

            minDI = distsInf.rowwise().minCoeff();
            minD2 = dists2.rowwise().minCoeff();

            // Use these to compute the intersite distance
            float d_min = 2*0.5/(float)noSamps;
            float best = 0.0f;
            bool success = false;

            for (int jj = 0; jj < candidates.rows(); jj++) {
                if (minDI(jj) >= d_min) {
                    success = true;
                    if (minD2(jj) > best) {
                        sample(ii) = jj;
                        best = minD2(jj);
                    }
                }
            }

            // If there are no points over the projected distance threshold we
            // resort to using the best of the intersite distances.
            for (int jj = 0; jj < candidates.rows(); jj++) {
                float val = ((pow(noSamps + 1,1/candidates.cols()) - 1)/2)*
                        minD2(jj) + ((noSamps + 1)/2)*minDI(jj);

                if (val > best) {
                    sample(ii) = jj;
                    best = val;
                }
            }

            Eigen::MatrixXf currtemp = currf;
            currf.resize(noSamps+1,candidates.cols());
            currf.block(0,0,noSamps,candidates.cols()) = currtemp;

            currf.row(noSamps) = candf.row(sample(ii)).cast<float>();
            noSamps++;
        }
    }
}
