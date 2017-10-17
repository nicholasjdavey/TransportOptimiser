#include "../transportbase.h"

double Costs::unitRevenueVar = 0.0;

Costs::Costs(RoadPtr road) {
    this->road = road;
    this->accidentFixed = 0.0;
    this->accidentVar = 0.0;
    this->earthwork = 0.0;
    this->lengthFixed = 0.0;
    this->lengthVar = 0.0;
    this->location = 0.0;
    this->penaltyCost = 0.0;
}

Costs::Costs(RoadPtr road, double af, double av, double e, double lf, double
        lv, double loc, double pc) {
    this->road = road;
    this->accidentFixed = af;
    this->accidentVar = av;
    this->earthwork = e;
    this->lengthFixed = lf;
    this->lengthVar = lv;
    this->location = loc;
    this->penaltyCost = pc;
}

Costs::~Costs() {
}

void Costs::computeUnitRevenue(OptimiserPtr optimiser) {
    const std::vector<VehiclePtr>& vehicles = optimiser->getTraffic()->
            getVehicles();

    Costs::unitRevenueVar = 0;

    for (int ii = 0; ii < vehicles.size(); ii++) {
        Costs::unitRevenueVar += 6570*(vehicles[ii]->getProportion()
                *vehicles[ii]->getMaximumLoad());
    }
}

void Costs::computeEarthworkCosts() {
    // Get segment lengths and segment average depths.
    // Negative values represent fill, positive values
    // represent cuts.
    RoadPtr roadPtrShared = this->road.lock();

    long int segs = roadPtrShared->getRoadSegments()->getDists().size() - 1;
    Eigen::VectorXd depth = roadPtrShared->getRoadSegments()->getE() -
            roadPtrShared->getRoadSegments()->getZ();
    Eigen::VectorXd segLen = roadPtrShared->getRoadSegments()->getDists().
            segment(1,segs) - roadPtrShared->getRoadSegments()->getDists().
            segment(0,segs);
    const Eigen::VectorXd& rw = roadPtrShared->getRoadSegments()->getWidths();
    double repC = roadPtrShared->getOptimiser()->getDesignParameters()
            ->getCutRep();
    double repF = roadPtrShared->getOptimiser()->getDesignParameters()
            ->getFillRep();

    Eigen::VectorXd avDepth = 0.5*(depth.segment(1,segs)
            + depth.segment(0,segs));
    Eigen::VectorXi type = Eigen::VectorXi::Constant(segs,
            (int)(RoadSegments::ROAD));
    roadPtrShared->getRoadSegments()->setType(type);

    // Costs are USD per m^3 in 2015 based on Chew, Goh & Fwa 1988 indexed.
    // When different soil types are used, they shall be included as well.
    // The first zero in each of the followin two vectors corresponds the to
    // the case where the segment is actually filled. This allows us to use
    // the vectors in matrix calculations to speed up the computation.
    const Eigen::VectorXd& cDepths = roadPtrShared->getOptimiser()->
            getEarthworkCosts()->getDepths();
    const Eigen::VectorXd& cCosts = roadPtrShared->getOptimiser()->
            getEarthworkCosts()->getCutCosts();
    double fCost = roadPtrShared->getOptimiser()->getEarthworkCosts()
            ->getFillCost();

    Eigen::VectorXd cut = (avDepth.array() > 0).cast<double>();
    Eigen::VectorXd fill = 1 - cut.array();

    // CutLevel entry of 0 indicates that it is actually a fill. This is used
    // for indexing in the matrices that follow.
    Eigen::MatrixXd dWidth = Eigen::MatrixXd::Zero(cDepths.size(),
            avDepth.size());
    Eigen::VectorXd cutCosts = Eigen::VectorXd::Zero(avDepth.size());
    Eigen::VectorXd fillCosts = Eigen::VectorXd::Zero(avDepth.size());
    Eigen::VectorXd finalLayerCost = Eigen::VectorXd::Zero(avDepth.size());
    Eigen::VectorXd finalLayerDepth = Eigen::VectorXd::Zero(avDepth.size());
    Eigen::VectorXd finalLayerWidth = Eigen::VectorXd::Zero(avDepth.size());
    Eigen::VectorXd segCosts = Eigen::VectorXd::Zero(avDepth.size());
    finalLayerCost = cut.array()*cCosts(0);
    dWidth.block(0,0,1,avDepth.size()) =
            (cut.array()*(2*avDepth.array())/(tan(repC))
            + rw.array()).transpose();

    // Compute the cross section width at the start of this cost level for each
    // road segment.
    for (int ii = 1; ii < cDepths.size(); ii++) {
        Eigen::VectorXi lessDepth =
                (avDepth.array() >= cDepths(ii-1)).cast<int>();

        dWidth.block(ii,0,1,avDepth.size()) =
                (lessDepth.cast<double>().array()*(2*(avDepth.array()
                -cDepths(ii)))/(tan(repC))+rw.array()).transpose();

        finalLayerCost += (lessDepth.cast<double>().array()*
                (cCosts(ii)-finalLayerCost.array())).matrix();
        finalLayerDepth += (lessDepth.cast<double>().array()*
                (cDepths(ii)-finalLayerDepth.array())).matrix();

        // If we are in the first depth band, the following will add
        // nothing. The value for this will be added after the loop for
        // segments where the overall depth falls within this band.
        // Otherwise, we add the whole depth.
        cutCosts += (lessDepth.cast<double>().array()*(cDepths(ii)
                -cDepths(ii-1))*0.5*cCosts(ii-1)*
                (dWidth.block(ii,0,1,avDepth.size()).transpose().array()
                +dWidth.block(ii-1,0,1,avDepth.size()).transpose().array()))
                .matrix();
    }

    // Compute the costs associated with the deepest excavated cost level
    // and add to the cut costs. The width of the bottom is simply the road
    // width.
    finalLayerWidth = (2*(avDepth.array() - finalLayerDepth.array())/tan(repC)
            + rw.array()).matrix();
    cutCosts += ((cut.array()*finalLayerCost.array())*0.5*
            ((avDepth.array()-finalLayerDepth.array())*(finalLayerWidth.array()
            +rw.array()))).matrix();

    // If the cut cost for a segment is too great, build a tunnel instead.
    // If the fill cost for a segment is too great, build a bridge instead.
    // Ignore tunnel and bridge costs for now.

    // Compute the final costs for each segment
    cutCosts = (cut.array()*segLen.array()*cutCosts.array()).matrix();
    fillCosts = (fill.array()*segLen.array()*fCost*(rw.array()+
            avDepth.array().abs()/tan(repF))*(avDepth.array().abs()))
            .matrix();

    // bridgeCosts = fill.*segLen.*cBridge;
    // tunnelCosts = cut.*segLen.*cTunnel;
    // typ = typ + 2*cut.*(cutCosts > tunnelCosts) + fill.*(fillCosts > ...
    //     bridgeCosts);

    // segCosts = cut.*(min(cutCosts,tunnelCosts))+fill.*(min(fillCosts,...
    //     bridgeCosts));

    // Ignore bridges and tunnels for now
    segCosts = (cut.array()*cutCosts.array() + fill.array()*fillCosts.array())
            .matrix();
    this->setEarthwork(segCosts.sum());
}

void Costs::computeLocationCosts() {
    RoadPtr roadPtrShared = this->road.lock();

    RoadCellsPtr segments = roadPtrShared->getRoadCells();
    RegionPtr region = roadPtrShared->getOptimiser()->getRegion();

    Eigen::VectorXi purch(segments->getTypes().size());
    purch = (segments->getTypes().array() !=
            (int)RoadSegments::TUNNEL).cast<int>();

    Eigen::VectorXi interact(segments->getTypes().size());
    interact = (segments->getTypes().array() ==
            (int)RoadSegments::ROAD).cast<int>();

    // Acquisition costs
    Eigen::VectorXd ac(segments->getTypes().size());
    const Eigen::MatrixXd& acCostsPtr = region->getAcquisitionCost();
    Utility::sliceIdx(acCostsPtr,segments->getCellRefs(),ac);

    double acqCost = (purch.segment(0,purch.size()-1).cast<double>().array()*
            ac.segment(0,ac.size()-1).cast<double>().array()*
            (segments->getAreas().array())).sum();

    // Soil stabilisation costs
    Eigen::VectorXd stab(segments->getTypes().size());
    const Eigen::MatrixXd& stabCostsPtr = region->getSoilStabilisationCost();
    Utility::sliceIdx(stabCostsPtr,segments->getCellRefs(),stab);
    double stabCost = (purch.segment(0,purch.size()-1).cast<double>().array()*
            stab.segment(0,stab.size()-1).cast<double>().
            array()*(segments->getAreas()).array()).sum();

    // Species habitat damage costs (accumulates over each species)
    const std::vector<SpeciesPtr>& species = roadPtrShared->getOptimiser()
            ->getSpecies();

    double regionPenaltyCost = 0;
    if (roadPtrShared->getOptimiser()->getType() == Optimiser::SIMPLEPENALTY) {

        for (int ii = 0; ii < species.size(); ii++) {
            const Eigen::MatrixXi& habMapPtr = species[ii]->getHabitatMap();
            const std::vector<HabitatTypePtr>& habitatTypes =
                    species[ii]->getHabitatTypes();

            Eigen::VectorXi habTypes(segments->getVegetation().size());
            Utility::sliceIdx(habMapPtr,segments->getCellRefs(),habTypes);

            for (int jj = 0; jj < habitatTypes.size(); jj++) {
                Eigen::VectorXi ofType = (habTypes.array() ==
                        habitatTypes[jj]->getType()).cast<int>();

                regionPenaltyCost += (ofType.cast<double>() *
                        habitatTypes[jj]->getCostPerM2()).sum();
            }
        }
    }

    this->location = acqCost + stabCost + regionPenaltyCost;
}

void Costs::computeLengthCosts() {
    // All costs end up as per unit hourly traffic volume PER YEAR
    RoadPtr roadPtrShared = this->road.lock();

    RoadCellsPtr cells = roadPtrShared->getRoadCells();
    double length = roadPtrShared->getAttributes()->getLength();
    DesignParametersPtr desParams = roadPtrShared->getOptimiser()
            ->getDesignParameters();
    UnitCostSPtr unitCosts = roadPtrShared->getOptimiser()
            ->getUnitCosts();
    const Eigen::VectorXd& z = roadPtrShared->getRoadSegments()->getZ();
    const Eigen::VectorXd& s = roadPtrShared->getRoadSegments()->getDists();
    const Eigen::VectorXd& v = roadPtrShared->getRoadSegments()->getVelocities();
    const Eigen::VectorXd& areas = cells->getAreas();

    this->lengthFixed = (desParams->getCostPerSM()*
            areas.array()).sum();

    Eigen::VectorXd gr = Eigen::VectorXd::Zero(z.size()-1);
    gr = 100*((z.segment(1,z.size()-1)
            - z.segment(0,z.size()-1)).array()/((s.segment(1,s.size()-1)
            - s.segment(0,s.size()-1)).array()));

    Eigen::VectorXd vel = 3.6*(v.segment(0,v.size()-1));
    // Travel time in hours of a single journey
    double travelTime = ((s.segment(1,s.size()-1).array() -
            s.segment(0,s.size()-1).array())/(1000*vel.array())).sum();

    // Enviro cost for full length for one vehicle for a whole year
    double enviroCost = 6570*0.001*0.01*length*(
            unitCosts->getAirPollution()
            + unitCosts->getNoisePollution()
            + unitCosts->getWaterPollution()
            + unitCosts->getOilExtraction()
            + unitCosts->getLandUse()
            + unitCosts->getSolidChemWaste());

    double K = roadPtrShared->getOptimiser()->getTraffic()->
            getPeakProportion();
    double D = roadPtrShared->getOptimiser()->getTraffic()->
            getDirectionality();
    double Hp = roadPtrShared->getOptimiser()->getTraffic()->
            getPeakHours();

    Eigen::VectorXd Q(3);
    Q << 1.0e-6*K*D*154.5*18,
            1.0e-6*K*(1-D)*154.5*18,
            5.0e-7*(6570-309*Hp)*(1-K)/(18-Hp)*18;

    const std::vector<VehiclePtr>& vehicles = roadPtrShared->getOptimiser()
            ->getTraffic()->getVehicles();

    Eigen::MatrixXd coeffSE(vehicles.size(),4);
    Eigen::MatrixXd coeffES(vehicles.size(),4);
    Eigen::VectorXd prop(vehicles.size());
    Eigen::VectorXd cphr(vehicles.size());

    for (int ii = 0; ii < vehicles.size(); ii++) {
        coeffSE.block(ii,0,1,4) << vehicles[ii]->getConstant(),
                vehicles[ii]->getGradeCoefficient(),
                -vehicles[ii]->getVelocityCoefficient(),
                vehicles[ii]->getVelocitySquared();
        coeffES.block(ii,0,1,4) << vehicles[ii]->getConstant(),
                -vehicles[ii]->getGradeCoefficient(),
                -vehicles[ii]->getVelocityCoefficient(),
                vehicles[ii]->getVelocitySquared();

        prop(ii) = vehicles[ii]->getProportion();
        cphr(ii) = vehicles[ii]->getHourlyCost();
    }

    Eigen::MatrixXd params(4,gr.size());

    // Velocities in km/hr
    params.block(0,0,1,gr.size()) = Eigen::RowVectorXd::Constant(gr.size(),1);
    params.block(1,0,1,gr.size()) = gr.transpose(); // G in percent
    params.block(2,0,1,gr.size()) = 3.6*v.segment(0,gr.size()).transpose();
    params.block(3,0,1,gr.size()) = 12.96*(v.segment(0,gr.size()).transpose()
            .array().pow(2));

    Eigen::VectorXd diffs = (s.segment(1,gr.size())-s.segment(0,gr.size()));
    Eigen::MatrixXd fcinter = (coeffSE*params + coeffES*params)*diffs;
    Eigen::VectorXd fc = (fcinter.array()*prop.array()).matrix();

    Eigen::MatrixXd fcrep(vehicles.size(),3);
    igl::repmat(fc,1,3,fcrep);

    // Fuel usage per year
    this->unitFuelVar = fcrep*Q;

    Eigen::Vector3d repTcphr = Eigen::Vector3d::Constant(prop.transpose()*
            cphr);
    double travelCost = (2.0e6)*travelTime*repTcphr.transpose()*Q;

    this->lengthVar = enviroCost + travelCost;

    // If we are dealing with the situation where we have an existing
    // alternative route, we convert these variable costs to variable profits
    // by taking away the cost of the existing road.
    if (this->getRoad()->getOptimiser()->getComparisonRoad() != nullptr) {
        RoadPtr compRoad = this->getRoad()->getOptimiser()->
                getComparisonRoad();
        this->unitFuelVar = this->unitFuelVar.array() - compRoad->getCosts()->
                getUnitFuelCost().array()*this->getRoad()->getOptimiser()->
                getVariableParams()->getCompRoad()(this->getRoad()->
                getOptimiser()->getScenario()->getCompRoad());
        this->lengthVar -= compRoad->getCosts()->getLengthVariable()*this->
                getRoad()->getOptimiser()->getVariableParams()->getCompRoad()
                (this->getRoad()->getOptimiser()->getScenario()->getCompRoad());
    }
}

void Costs::computeAccidentCosts() {
    RoadPtr roadPtrShared = this->road.lock();

    // 18 hour day
    const Eigen::VectorXd& delta = roadPtrShared->getHorizontalAlignment()->getDeltas();
    const Eigen::VectorXd& R = roadPtrShared->getHorizontalAlignment()->getRadii();
    const Eigen::VectorXd& ssd = roadPtrShared->getVerticalAlignment()->getSSDs();
    const Eigen::VectorXd& vels = roadPtrShared->getRoadSegments()->getVelocities();
    double width = roadPtrShared->getOptimiser()->getDesignParameters()
            ->getRoadWidth();
    double accCost = roadPtrShared->getOptimiser()->getUnitCosts()
            ->getAccidentCost();

    double qfac = 6570*(1.0e-6);
    bool spiral = roadPtrShared->getOptimiser()->getDesignParameters()
            ->getSpiral();

    // Zegeer et al.
    this->accidentVar = accCost*qfac*(0.96*(R.array())*(delta.array())/1000
            +0.14*(delta.array()*180/M_PI - 0.12*(double)spiral)).sum() *
            pow(0.978,3.28*width-30);

    // As with the length costs above, if we are comparing this to an existing
    // road, we only care about the difference.
    if (this->getRoad()->getOptimiser()->getComparisonRoad() != nullptr) {
        RoadPtr compRoad = this->getRoad()->getOptimiser()->
                getComparisonRoad();
        this->accidentVar -= compRoad->getCosts()->getAccidentVariable();
    }

    Eigen::VectorXd mActual = R.array()*(1-cos(delta.array()/2));
    Eigen::VectorXd mReq = R.array()/(1-cos(28.65*ssd.array()*M_PI/
            (R.array()*180)));
    Eigen::VectorXd diff = mActual - mReq;
    Eigen::VectorXd diffsq = diff.array().pow(2);
    Eigen::VectorXd tooSmall = (mActual.array() > mReq.array()).cast<double>();
    this->accidentFixed = (accCost/5)*tooSmall.transpose()*diffsq;

    // Penalise roads that require a speed that is less than half the original
    // design speed
    OptimiserPtr optimiserPtrShared = roadPtrShared->getOptimiser();
    double velReq = 0.5*optimiserPtrShared->getDesignParameters()->
            getDesignVelocity();
    if (vels.minCoeff() < velReq) {
        Eigen::VectorXd invalidCurves = (vels.array() < velReq).cast<double>();
        Eigen::VectorXd velDiff = invalidCurves.array()*(velReq -
                vels.array());

        // For each road curve that is below the minimum (half the design
        // speed) we add a penalty that is equal to the difference multiplied
        // by the fixed road costs so far. This is arbitrary but encourages
        // safer curves.
        double unitPenalty = this->lengthFixed;

        this->accidentFixed += (velDiff.array().pow(2)*unitPenalty).sum();
    }
}

void Costs::computePenaltyCost() {
    RoadPtr roadPtrShared = this->road.lock();

    const std::vector<SpeciesRoadPatchesPtr>& speciesRoadPatches =
            roadPtrShared->getSpeciesRoadPatches();

    this->penaltyCost = 0;
    // The target confidence interval we are using
    double cumpr = 1 - roadPtrShared->getOptimiser()->getConfidenceInterval();
    double sd = Utility::NormalCDFInverse(cumpr);

    for (int ii = 0; ii < speciesRoadPatches.size(); ii++) {
        // Population at target confidence interval. For now we just assume
        // that the sample data are normally distributed. This needs to be
        // changed at a later date.
        double endPop = speciesRoadPatches[ii]->getEndPopMean()
                +sd*speciesRoadPatches[ii]->getEndPopSD();
        double thresh = speciesRoadPatches[ii]->getSpecies()->
                getThreshold();
        if (thresh > endPop) {
            this->penaltyCost += (thresh - endPop)*
                    speciesRoadPatches[ii]->getSpecies()->
                    getCostPerAnimal();
        }
    }
}
