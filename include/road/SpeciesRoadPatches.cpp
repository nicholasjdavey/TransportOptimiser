#include "../transportbase.h"

SpeciesRoadPatches::SpeciesRoadPatches(OptimiserPtr optimiser, SpeciesPtr
        species, RoadPtr road) : Uncertainty(optimiser) {
    this->species = species;
    this->road = road;
    this->initPop = 0.0;
}

SpeciesRoadPatches::SpeciesRoadPatches(OptimiserPtr optimiser, SpeciesPtr
        species, RoadPtr road, bool active, double mean, double stdDev, double
        rev, std::string nm) : Uncertainty(optimiser, nm, mean, stdDev, rev,
        active) {

    this->species = species;
    this->road = road;
    this->initPop = 0.0;
}

SpeciesRoadPatches::~SpeciesRoadPatches() {}

void SpeciesRoadPatches::createSpeciesModel() {
    this->generateHabitatPatchesGrid();
    this->getDistances();
    this->getCrossings();
    this->computeTransitionProbabilities();
    this->computeSurvivalProbabilities();
}

void SpeciesRoadPatches::generateHabitatPatchesGrid() {
    // First initialise the number of habitat patches. We expect there to be no
    // more than n x y where n is the number of habitat patches and y is the
    // number of grid cells.
    RoadPtr roadPtrShared = this->road.lock();
    RegionPtr region = roadPtrShared->getOptimiser()->getRegion();
    const Eigen::MatrixXd& X = region->getX();
    const Eigen::MatrixXd& Y = region->getY();
    const std::vector<HabitatTypePtr>& habTyps = this->species->getHabitatTypes();
    int res = roadPtrShared->getOptimiser()->getGridRes();

    Eigen::VectorXd xspacing = (X.block(1,0,X.rows()-1,1)
            - X.block(0,0,X.rows()-1,1)).transpose();
    Eigen::VectorXd yspacing = Y.block(0,1,1,Y.cols()-1)
            - Y.block(0,0,1,Y.cols()-1);

    // Grid will be evenly spaced upon call
    if ((xspacing.segment(1,xspacing.size()-1)
            - xspacing.segment(0,xspacing.size()-1)).sum() > 1e-4 ||
            (yspacing.segment(1,yspacing.size()-1)
            - yspacing.segment(0,yspacing.size()-1)).sum() > 1e-4) {
        throw std::invalid_argument("Grid must be evenly spaced in both X and Y");
    }

    const Eigen::MatrixXi modHab = this->species->getHabitatMap();
    Eigen::MatrixXi tempHabVec = Eigen::MatrixXi::Constant(1,
            roadPtrShared->getRoadCells()->getUniqueCells().size(),
            (int)(HabitatType::ROAD));
    igl::slice_into(modHab,roadPtrShared->getRoadCells()->getUniqueCells(),
            tempHabVec);

    // We create bins for each habitat type into which we place the patches. We
    // ignore CLEAR and ROAD habitats, hence -2
    unsigned short W = (X.rows());
    unsigned short H = (Y.cols());
    unsigned short xRes = W % res == 0 ? res : res + 1;
    unsigned short yRes = H % res == 0 ? res : res + 1;

    // Number of cells in each coarse grid cell used for creating habitat
    // patches (a sub patch)
    int skpx = std::floor((W-1)/(double)res);
    int skpy = std::floor((H-1)/(double)res);

    this->habPatch = std::vector<HabitatPatchPtr>((pow(res,2)*habTyps.size()));
    // Sub patch area
    double subPatchArea = xspacing(0)*yspacing(0);

    int iterator = 0;
    int patches = 0;
    iterator++;

    // Get number of animals that need to be relocated (animals in road cells)
    double relocateAnimals = ((modHab.array() == (int)(HabitatType::ROAD))
            .cast<double>()*this->species->getPopulationMap().array()).sum();
    double totalPop = (this->species->getPopulationMap()).sum();
    // Factor by which to increase each population
    double factor = totalPop/(totalPop - relocateAnimals);

    for (int ii = 0; ii < habTyps.size(); ii++) {
        if (habTyps[ii]->getType() == HabitatType::ROAD ||
                habTyps[iterator]->getType() == HabitatType::CLEAR) {
            continue;
        }
        Eigen::MatrixXi input = (modHab.array() ==
                (int)(habTyps[iterator]->getType())).cast<int>();
        // Map the input array to a plain integer C-array
        int* cinput;
        Eigen::Map<Eigen::MatrixXi>(cinput,W,H) = input;

        // Map output C-array to an Eigen int matrix
        int* coutput = (int*) malloc(W*H*sizeof(int));
        memset(coutput,0,W*H*sizeof(int));
        Eigen::MatrixXi output = Eigen::Map<Eigen::MatrixXi>(coutput,W,H);

        // Separate contiguous regions of this habitat type
        int regions = LabelImage(W,H,cinput,coutput);
        free(coutput);

        // Iterate through every large cell present in the overall region
        for (int jj = 0; jj < xRes; jj++) {
            for (int kk = 0; kk < yRes; kk++) {
                Eigen::MatrixXi tempGrid = Eigen::MatrixXi::Zero(W,H);

                int blockSizeX;
                int blockSizeY;
                if (jj*skpx <= W) {
                    blockSizeX = skpx;
                } else {
                    blockSizeX = W-jj*skpx;
                }

                if (kk*skpy <= H) {
                    blockSizeY = skpy;
                } else {
                    blockSizeY = H-kk*skpy;
                }

                tempGrid.block(jj*skpx,kk*skpy,blockSizeX,blockSizeY) =
                        Eigen::MatrixXi::Constant(skpx,skpy,1);

                // For every valid habitat type, we must create a new animal
                // patch. If the patch contains valid habitat, we add it to
                // our list of patches for use later.
                for (int ll = 1; ll <= regions; ll++) {
                    Eigen::MatrixXi tempGrid2;
                    tempGrid2 = ((output.array() == ll).cast<int>()
                            *tempGrid.array()).matrix();

                    // If the patch contains this habitat, we continue
                    int noCells = tempGrid2.sum();
                    if (noCells > 0) {
                        HabitatPatchPtr hab(new HabitatPatch());
                        hab->setArea((double)(tempGrid2.sum()*subPatchArea));
                        Eigen::VectorXi xidx =
                                Eigen::VectorXi::LinSpaced(W,1,W);
                        Eigen::VectorXi yidx =
                                Eigen::VectorXi::LinSpaced(H,1,H);

                        hab->setCX((double)(xspacing(0)*((xidx.transpose()*
                                tempGrid2).sum())/noCells + X(0,0)
                                - xspacing(0)));
                        hab->setCY((double)(yspacing(0)*(tempGrid2*yidx).sum()/
                                noCells + Y(0,0) - yspacing(0)));

                        hab->setType(habTyps[ii]);
                        double thisPop = (tempGrid2.array().cast<double>()*
                                (this->species->getPopulationMap()).array()).sum()*factor;
                        hab->setPopulation(thisPop);
                        // For now do not store the indices of the points
                        this->habPatch[patches] = hab;
                        patches++;
                        this->initPop += thisPop;
                        // Find distance to road here?
                    }
                }
            }
        }
        iterator++;
    }

    // Remove excess patches in container
    this->habPatch.resize(--patches);
}

void SpeciesRoadPatches::generateHabitatPatchesBlob() {

}

void SpeciesRoadPatches::habitatPatchDistances() {
    // First copy relevant HabitatPatch components to vectors for ease of use
    int hps = this->habPatch.size();

    Eigen::VectorXd xorg = Eigen::VectorXd::Zero(hps);
    Eigen::VectorXd yorg = Eigen::VectorXd::Zero(hps);

    for (int ii = 0; ii < hps; ii++) {
        xorg(ii) = this->habPatch[ii]->getCX();
        yorg(ii) = this->habPatch[ii]->getCY();
    }

    Eigen::MatrixXd xDests(hps,hps);
    Eigen::MatrixXd yDests(hps,hps);
    Eigen::MatrixXd xOrgs(hps,hps);
    Eigen::MatrixXd yOrgs(hps,hps);

    igl::repmat(xorg,1,hps,xDests);
    igl::repmat(yorg,1,hps,yDests);
    xOrgs = xDests.transpose();
    yOrgs = yDests.transpose();

    this->dists = ((xDests - xOrgs).array().pow(2)
            + (yDests - yOrgs).array().pow(2)).array().sqrt().matrix();
}

void SpeciesRoadPatches::roadCrossings() {

    RoadPtr roadPtrShared = this->road.lock();
    const Eigen::VectorXd& px = roadPtrShared->getRoadSegments()->getX();
    const Eigen::VectorXd& py = roadPtrShared->getRoadSegments()->getY();
    const Eigen::VectorXi& typ = roadPtrShared->getRoadSegments()->getType();
    int noSegs = px.size()-1;
/*
 DEPRECATED FROM MATLAB TEST CODE
    // First copy relevant HabitatPatch components to vectors for ease of use
    Eigen::MatrixXd roadSegsVisible(noSegs,4);
    roadSegsVisible.block(0,0,noSegs,1) = px->segment(0,noSegs);
    roadSegsVisible.block(0,1,noSegs,1) = py->segment(0,noSegs);
    roadSegsVisible.block(0,2,noSegs,1) = px->segment(1,noSegs);
    roadSegsVisible.block(0,3,noSegs,1) = py->segment(1,noSegs);

    Eigen::MatrixXd typMat(noSegs,4);
    typMat.block(0,0,noSegs,1) = (typ->array() == (int)RoadSegments::ROAD)
            .cast<double>();
    typMat.block(0,2,noSegs,1) = (typ->array() == (int)RoadSegments::ROAD)
            .cast<double>();
    typMat.block(0,2,noSegs,1) = (typ->array() == (int)RoadSegments::ROAD)
            .cast<double>();
    typMat.block(0,2,noSegs,1) = (typ->array() == (int)RoadSegments::ROAD)
            .cast<double>();
*/

    // We reduce the computation time by only considering cells that are close
    // enough to the originating cell (i.e. they are close enough to each other
    // that a transition is likely in the absence of a road). We use the 5th
    // percentile distance based on the simple exponential distribution used in
    // Rhodes et al. 2014.
    double lda = (roadPtrShared->getOptimiser()->getVariableParams()
            ->getLambda())(roadPtrShared->getOptimiser()->getScenario()
            ->getLambda());
    double maxDist = log(0.05)/(-(this->species->getLambdaMean() + this->
            species->getLambdaSD()*lda));

    // No need to remove road segments that are not finite or for which the
    // start and end points are coincident as this is already taken care of by
    // RoadSegment.cpp. We simply remove segments that are not open road.
    Eigen::MatrixXd roadSegsVisible(noSegs,4);
    int iterator = 0;
    for (int ii = 0; ii < noSegs; ii++) {
        if (typ(ii) == (int)(RoadSegments::ROAD)) {
            roadSegsVisible(iterator,0) = px(ii);
            roadSegsVisible(iterator,1) = py(ii);
            roadSegsVisible(iterator,2) = px(ii+1);
            roadSegsVisible(iterator,3) = py(ii+1);
            iterator++;
        }
    }
    // We submit all patch transitions simultaneously but if this proves too
    // memory-intensive, we can change the code to do it sequentially.
    int validCrossings = -1;
    Eigen::MatrixXi indices(this->habPatch.size()*(this->habPatch.size()-1),2);
    Eigen::MatrixXd lines(this->habPatch.size()*(this->habPatch.size()-1),4);

    for (int ii = 0; ii < this->habPatch.size(); ii++) {
        for (int jj = ii+1; jj < this->habPatch.size(); jj++) {
            if (this->dists(ii,jj) <= maxDist) {
                validCrossings++;
                indices(validCrossings,1) = ii;
                indices(validCrossings,2) = jj;
                lines.block(validCrossings,0,1,4) <<
                        this->habPatch[ii]->getCX(),
                        this->habPatch[ii]->getCY(),
                        this->habPatch[jj]->getCX(),
                        this->habPatch[jj]->getCY();
            }
        }
    }
    Eigen::MatrixXi crossings = Utility::lineSegmentIntersect(lines,
            roadSegsVisible);
    // Find the number of road crossings for the valid transitions identified
    // above.
    Eigen::MatrixXi orgs = indices.block(0,0,validCrossings,1);
    Eigen::MatrixXi dests = indices.block(0,1,validCrossings,1);

    this->crossings = Eigen::MatrixXi(this->habPatch.size(),
            this->habPatch.size());
    igl::slice_into(crossings,orgs,dests,this->crossings);
    igl::slice_into(crossings,dests,orgs,this->crossings);
}

void SpeciesRoadPatches::computeTransitionProbabilities() {

    RoadPtr roadPtrShared = this->road.lock();

    double lda = (roadPtrShared->getOptimiser()->getVariableParams()
            ->getLambda())(roadPtrShared->getOptimiser()->getScenario()
            ->getLambda());
    double hp = (roadPtrShared->getOptimiser()->getVariableParams()
            ->getHabPref()(roadPtrShared->getOptimiser()->getScenario()
            ->getHabPref()));
    double maxDist = log(0.05)/(-(this->species->getLambdaMean() + this->
            species->getLambdaSD()*lda));

    this->transProbs = Eigen::MatrixXd::Zero(this->habPatch.size(),
            this->habPatch.size());

    for (int ii = 0; ii < this->habPatch.size(); ii++) {
        double summ = 0;

        for (int jj = 0; ii < this->habPatch.size(); jj++) {
            if (this->dists(ii,jj) <= maxDist) {
                if (ii == jj) {
                    this->transProbs(ii,jj) = this->habPatch[jj]->getArea()
                            *lda*exp(this->habPatch[jj]->getType()
                            ->getHabPrefMean() + this->habPatch[jj]->getType()
                            ->getHabPrefSD()*hp);
                } else {
                    this->transProbs(ii,jj) = (this->habPatch[jj]->getArea()
                            *lda*exp(this->habPatch[jj]->getType()
                            ->getHabPrefMean() + this->habPatch[jj]->getType()
                            ->getHabPrefSD()*hp)) / exp(-lda*dists(ii,jj));
                }
                summ += this->transProbs(ii,jj);
            }
        }

        // Normalise all weightings
        this->transProbs.block(ii,0,1,this->transProbs.cols()) =
                this->transProbs.block(ii,0,1,this->transProbs.cols()) / summ;
    }
}

void SpeciesRoadPatches::computeSurvivalProbabilities() {
    double len = this->species->getLengthMean();
    double spd = this->species->getSpeedMean();
    RoadPtr roadPtrShared = this->road.lock();

    TrafficProgramPtr program = (roadPtrShared->getOptimiser()->getPrograms())[
            roadPtrShared->getOptimiser()->getScenario()->getProgram()];
    const std::vector<VehiclePtr>& vehicles = program->getTraffic()->getVehicles();

    double avVehWidth = 0;
    for (int ii = 0; ii < vehicles.size(); ii++) {
        avVehWidth += vehicles[ii]->getWidth()
                *vehicles[ii]->getProportion();
    }

    int controls = program->getFlowRates().size();

    for (int ii = 0; ii < controls; ii++) {
        this->survProbs[ii] = (-this->crossings.array()*(
                (program->getFlowRates())[ii])*(avVehWidth+len)/(spd*3600))
                .exp().cast<double>();
    }
}

void SpeciesRoadPatches::computeAAR(const Eigen::VectorXd& pops,
        Eigen::VectorXd& aar) {
    RoadPtr roadPtrShared = this->road.lock();
    // No asserts are performed here as the population input vector must be
    // checked for sizing requirements before use by the calling function
    TrafficProgramPtr program = (roadPtrShared->getOptimiser()->getPrograms())[
            roadPtrShared->getOptimiser()->getScenario()->getProgram()];
    int controls = program->getFlowRates().size();

    double popInit = pops.sum();

    for (int ii = 0; ii < controls; ii++) {
        Eigen::VectorXd newPops = this->transProbs*
                this->survProbs[ii].transpose()*pops;
        aar(ii) = 1-newPops.sum()/popInit;
    }
}

void SpeciesRoadPatches::computeInitialAAR(Eigen::VectorXd &iar) {

    int hps = this->habPatch.size();
    Eigen::VectorXd pops(hps);

    for (int ii = 0; ii < hps; ii++) {
        pops(ii) = this->habPatch[ii]->getPopulation();
    }

    this->computeAAR(pops,iar);
}
