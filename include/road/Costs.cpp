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

    for (int ii = 0; ii < vehicles.size(); ii++) {
        Costs::unitRevenueVar += 6570*(vehicles[ii]->getProportion()
                *vehicles[ii]->getMaximumLoad());
    }
}

void Costs::computeEarthworkCosts() {
	// Get segment lengths and segment average depths.
	// Negative values represent fill, positive values
	// represent cuts.
    long int segs = this->road->getRoadSegments()->getDists().size() - 1;
    Eigen::VectorXd depth = this->road->getRoadSegments()->getE() -
            road->getRoadSegments()->getZ();
    Eigen::VectorXd segLen = this->road->getRoadSegments()->getDists().
            segment(1,segs+1) - this->road->getRoadSegments()->getDists().
			segment(0,segs);
    const Eigen::VectorXd& rw = this->road->getRoadSegments()->getWidths();
	double repC = this->road->getOptimiser()->getDesignParameters()
			->getCutRep();
	double repF = this->road->getOptimiser()->getDesignParameters()
			->getFillRep();

	Eigen::VectorXd avDepth = 0.5*(depth.segment(1,segs+1)
			+ depth.segment(0,segs));
    Eigen::VectorXi type = Eigen::VectorXi::Constant(segs,
            (int)(RoadSegments::ROAD));
    this->road->getRoadSegments()->setType(type);

	// Costs are USD per m^3 in 2015 based on Chew, Goh & Fwa 1988 indexed.
    // When different soil types are used, they shall be included as well.
    // The first zero in each of the followin two vectors corresponds the to
    // the case where the segment is actually filled. This allows us to use
    // the vectors in matrix calculations to speed up the computation.
    const Eigen::VectorXd& cDepths = this->road->getOptimiser()->
            getEarthworkCosts()->getDepths();
    const Eigen::VectorXd& cCosts = this->road->getOptimiser()->
            getEarthworkCosts()->getCutCosts();
	double fCost = this->road->getOptimiser()->getEarthworkCosts()
			->getFillCost();

	Eigen::VectorXd cut = (avDepth.array() > 0).cast<double>();
	Eigen::VectorXd fill = 1 - avDepth.array();

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

	// Compute the cross section width at the start of this cost level for each
	// road segment.
    for (int ii = 2; ii< cDepths.size(); ii++) {
		Eigen::VectorXi lessDepth =
                (avDepth.array() >= cDepths(ii-1)).cast<int>();

		dWidth.block(ii-1,0,1,avDepth.size()) =
				((lessDepth.cast<double>().array()*(2*avDepth.array()
                -cDepths(ii-1)))/((tan(repC)+rw.array()))).matrix();
		
		finalLayerCost += (lessDepth.cast<double>().array()*
                (cCosts(ii)-cCosts(ii-1))).matrix();
		finalLayerDepth += (lessDepth.cast<double>().array()*
                (cDepths(ii)-cDepths(ii-1))).matrix();

		// If we are in the first depth band, the following will add
        // nothing. The value for this will be added after the loop for
        // segments where the overall depth falls within this band.
        // Otherwise, we add the whole depth.
        cutCosts += (lessDepth.cast<double>().array()*(cDepths(ii-1)
            -cDepths(ii-2))*0.5*cCosts(ii-1)*
			(dWidth.block(ii-1,0,1,avDepth.size()).array()
			+dWidth.block(ii-2,0,1,avDepth.size()).array())).matrix();
	}

	// Compute the costs associated with the deepest excavated cost level
    // and add to the cut costs. The width of the bottom is simply the road
    // width.
	finalLayerWidth = (2*(avDepth.array() - finalLayerDepth.array())/
			((tan(repC)+rw.array()))).matrix();
	cutCosts += ((cut.array())*finalLayerCost.array()*
			(avDepth.array()-finalLayerDepth.array())*(finalLayerWidth.array()
			+rw.array()*0.5)).matrix();

	// If the cut cost for a segment is too great, build a tunnel instead.
    // If the fill cost for a segment is too great, build a bridge instead.
    // Ignore tunnel and bridge costs for now.

	// Compute the final costs for each segment
	cutCosts = (cut.array()*segLen.array()*cutCosts.array()).matrix();
	fillCosts = (fill.array()*segLen.array()*fCost*(2*rw.array()+2*
			avDepth.array().abs()/tan(repF))*avDepth.array().abs())
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
    RoadCellsPtr segments = this->road->getRoadCells();
    RegionPtr region = this->road->getOptimiser()->getRegion();

    Eigen::VectorXi purch(segments->getTypes().size());
    purch = (segments->getTypes().array() !=
             (int)RoadSegments::TUNNEL).cast<int>();

    Eigen::VectorXi interact(segments->getTypes().size());
    interact = (segments->getTypes().array() ==
             (int)RoadSegments::ROAD).cast<int>();

    // Acquisition costs
    Eigen::VectorXd ac(segments->getTypes().size());
    const Eigen::MatrixXd& acCostsPtr = region->getAcquisitionCost();
    igl::slice(acCostsPtr,segments->getCellRefs(),ac);

    double acqCost = (purch.cast<double>().array()*ac.cast<double>().array()*
            (segments->getAreas().array())).sum();

    // Soil stabilisation costs
    Eigen::VectorXd stab(segments->getTypes().size());
    const Eigen::MatrixXd& stabCostsPtr = region->getSoilStabilisationCost();
    igl::slice(stabCostsPtr,segments->getCellRefs(),stab);
    double stabCost = (purch.cast<double>().array()*stab.cast<double>().
            array()*(segments->getAreas()).array()).sum();

    // Species habitat damage costs (accumulates over each species)
    const std::vector<SpeciesPtr>& species = this->road->getOptimiser()
            ->getSpecies();

    double regionPenaltyCost = 0;
    if (this->road->getOptimiser()->getType() == Optimiser::SIMPLEPENALTY) {

        for (int ii = 0; ii < species.size(); ii++) {
            const Eigen::MatrixXi& habMapPtr = species[ii]->getHabitatMap();
            const std::vector<HabitatTypePtr>& habitatTypes =
                    species[ii]->getHabitatTypes();

            Eigen::VectorXi habTypes(segments->getVegetation().size());
            igl::slice(habMapPtr,segments->getCellRefs(),habTypes);

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
    RoadCellsPtr cells = this->road->getRoadCells();
    double length = this->road->getAttributes()->getLength();
    DesignParametersPtr desParams = this->road->getOptimiser()
            ->getDesignParameters();
    UnitCostSPtr unitCosts = this->road->getOptimiser()
            ->getUnitCosts();
    const Eigen::VectorXd& z = this->road->getRoadSegments()->getZ();
    const Eigen::VectorXd& s = this->road->getRoadSegments()->getDists();
    const Eigen::VectorXd& v = this->road->getRoadSegments()->getVelocities();
    const Eigen::VectorXd& areas = cells->getAreas();

    this->lengthFixed = (desParams->getCostPerSM()*
            areas.array()).sum();

    Eigen::VectorXd gr = 100*((z.segment(1,z.size()-1)
            - z.segment(0,z.size()-1)).array()/(s.segment(1,s.size()-1)
            - s.segment(0,s.size()-1)).array());

    Eigen::VectorXd vel = 3.6*(v.segment(0,v.size()-1));
    double travelTime = (s.segment(1,s.size()-1).array()/vel.array()).sum();

    double enviroCost = 0.001*length*(
            unitCosts->getAirPollution()
            + unitCosts->getNoisePollution()
            + unitCosts->getWaterPollution()
            + unitCosts->getOilExtraction()
            + unitCosts->getLandUse()
            + unitCosts->getSolidChemWaste());

    double K = this->road->getOptimiser()->getTraffic()->
            getPeakProportion();
    double D = this->road->getOptimiser()->getTraffic()->
            getDirectionality();
    double Hp = this->road->getOptimiser()->getTraffic()->
            getPeakHours();

    Eigen::VectorXd Q(3);
    Q << 1.0e-6*K*D*154.5,
            1.0e-6*K*(1-D)*154.5,
            5.0e-7*(6570-309*Hp)*(1-K)/(18-Hp);

    const std::vector<VehiclePtr>& vehicles = this->road->getOptimiser()
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

    Eigen::MatrixXd params(4,gr.size()-1);

    params.block(0,0,1,gr.size()) = Eigen::RowVectorXd::Constant(gr.size(),1);
    params.block(1,0,1,gr.size()) = gr.transpose();
    params.block(2,0,1,gr.size()) = v.transpose();
    params.block(3,0,1,gr.size()) = v.transpose().array().pow(2);

    Eigen::MatrixXd fcinter = (coeffSE*params + coeffES*params)*
            (s.segment(1,gr.size())-s.segment(0,gr.size()));
    Eigen::VectorXd fc = (fcinter.array()*prop.array()).matrix();

    Eigen::MatrixXd fcrep(vehicles.size(),3);
    igl::repmat(fc,0,3,fcrep);

    this->unitFuelVar = fcrep*Q;

    Eigen::Vector3d repTcphr = Eigen::Vector3d::Constant(prop.transpose()*
            cphr);
    double travelCost = 2000*travelTime*repTcphr.transpose()*Q;

    this->lengthVar = enviroCost + travelCost;
}

void Costs::computeAccidentCosts() {
    // 18 hour day
    const Eigen::VectorXd& delta = this->road->getHorizontalAlignment()->getDeltas();
    const Eigen::VectorXd& R = this->road->getHorizontalAlignment()->getRadii();
    const Eigen::VectorXd& ssd = this->road->getVerticalAlignment()->getSSDs();
    double width = this->road->getOptimiser()->getDesignParameters()
            ->getRoadWidth();
    double accCost = this->road->getOptimiser()->getUnitCosts()
            ->getAccidentCost();

    double qfac = 6570*(1.0e-6);

    this->accidentVar = accCost*qfac*(0.96*(R.array())*(delta.array())/1000
            +0.14*(delta.array()*180/M_PI - 0.12)).sum() *
            pow(0.978,3.28*width-30);

    Eigen::VectorXd mActual = R.array()*(1-cos(delta.array()/2));
    Eigen::VectorXd mReq = R.array()/(1-cos(28.65*ssd.array()*M_PI/
            (R.array()*180)));
    Eigen::VectorXd diff = mActual - mReq;
    Eigen::VectorXd diffsq = diff.array().pow(2);
    Eigen::VectorXd tooSmall = (mActual.array() > mReq.array()).cast<double>();
    this->accidentFixed = (accCost/5)*tooSmall.transpose()*diffsq;
}

void Costs::computePenaltyCost() {
    const std::vector<SpeciesRoadPatchesPtr>& speciesRoadPatches =
            this->road->getSpeciesRoadPatches();

    this->penaltyCost = 0;
    // The target confidence interval we are using
    double cumpr = 1 - this->road->getOptimiser()->getConfidenceInterval();
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
