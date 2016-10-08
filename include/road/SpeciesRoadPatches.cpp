#include "../transportbase.h"

SpeciesRoadPatches::SpeciesRoadPatches(SpeciesPtr species, RoadPtr road) :
        Uncertainty() {
    this->species = species;
    this->road = road;
}

SpeciesRoadPatches::SpeciesRoadPatches(SpeciesPtr species, RoadPtr road,
        bool active, double mean, double stdDev, double rev,
        std::string nm) : Uncertainty(nm, mean, stdDev, rev, active) {

    this->species = species;
    this->road = road;
}

SpeciesRoadPatches::~SpeciesRoadPatches() {}

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
    Eigen::VectorXd* px = this->road->getRoadSegments()->getX();
    Eigen::VectorXd* py = this->road->getRoadSegments()->getY();
    Eigen::VectorXi* typ = this->road->getRoadSegments()->getType();
    int noSegs = px->size()-1;
    int visibleSegs = (typ->array() == ((int)(RoadSegments::ROAD))).sum();
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
    double lda = (*this->road->getOptimiser()->getVariableParams()
            ->getLambda())(this->road->getOptimiser()->getScenario()
            ->getLambda());
    double maxDist = log(0.05)/(-lda);

    // No need to remove road segments that are not finite or for which the
    // start and end points are coincident as this is already taken care of by
    // RoadSegment.cpp. We simply remove segments that are not open road.
    Eigen::MatrixXd roadSegsVisible(noSegs,4);
    int iterator = 0;
    for (int ii = 0; ii < noSegs; ii++) {
        if ((*typ)(ii) == (int)(RoadSegments::ROAD)) {
            roadSegsVisible(iterator,0) = (*px)(ii);
            roadSegsVisible(iterator,1) = (*py)(ii);
            roadSegsVisible(iterator,2) = (*px)(ii+1);
            roadSegsVisible(iterator,3) = (*py)(ii+1);
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
    double lda = (*this->road->getOptimiser()->getVariableParams()
            ->getLambda())(this->road->getOptimiser()->getScenario()
            ->getLambda());
    double maxDist = log(0.05)/(-lda);

    this->transProbs = Eigen::MatrixXd::Zero(this->habPatch.size(),
            this->habPatch.size());

    for (int ii = 0; ii < this->habPatch.size(); ii++) {
        double summ = 0;

        for (int jj = 0; ii < this->habPatch.size(); jj++) {
            if (this->dists(ii,jj) <= maxDist) {
                if (ii == jj) {
                    this->transProbs(ii,jj) = this->habPatch[jj]->getArea()
                            *lda*exp(this->habPatch[jj]->getType()
                            ->getHabPrefMean());
                } else {
                    this->transProbs(ii,jj) = (this->habPatch[jj]->getArea()
                            *lda*exp(this->habPatch[jj]->getType()
                            ->getHabPrefMean())) / exp(-lda*dists(ii,jj));
                }
                summ += this->transProbs(ii,jj);
            }
        }

        // Normalise all shares
        this->transProbs.block(ii,0,1,this->transProbs.cols()) =
                this->transProbs.block(ii,0,1,this->transProbs.cols()) / summ;
    }
}

void SpeciesRoadPatches::computeSurvivalProbabilities() {
    double len = this->species->getLengthMean();
    double spd = this->species->getSpeedMean();

    TrafficProgramPtr program = (*this->road->getOptimiser()->getPrograms())[
            this->road->getOptimiser()->getScenario()->getProgram()];
    std::vector<VehiclePtr>* vehicles = program->getTraffic()->getVehicles();

    double avVehWidth = 0;
    for (int ii = 0; ii < vehicles->size(); ii++) {
        avVehWidth += (*vehicles)[ii]->getWidth()
                *(*vehicles)[ii]->getProportion();
    }

    int controls = program->getFlowRates()->size();

    for (int ii = 0; ii < controls; ii++) {
        this->survProbs[ii] = (-this->crossings.array()*(
                (*program->getFlowRates())[ii])*(avVehWidth+len)/(spd*3600))
                .exp().cast<double>();
    }
}

Eigen::VectorXd SpeciesRoadPatches::computeAAR(Eigen::VectorXd *pops) {
    // No asserts are performed here as the population input vector must be
    // checked for sizing requirements before use by the calling function
    TrafficProgramPtr program = (*this->road->getOptimiser()->getPrograms())[
            this->road->getOptimiser()->getScenario()->getProgram()];
    int controls = program->getFlowRates()->size();

    Eigen::VectorXd aars(controls);
    double popInit = pops->sum();

    for (int ii = 0; ii < controls; ii++) {
        Eigen::VectorXd newPops = this->transProbs*
                this->survProbs[ii].transpose()*(*pops);
        aars(ii) = 1-newPops.sum()/popInit;
    }

    return aars;
}
