#include "../transportbase.h"

Road::Road() {
    // Assign an empty Attributes object (always has the same size)
    AttributesPtr att(new Attributes(this->me()));
    this->attributes = att;
}

Road::Road(OptimiserPtr op,const Eigen::VectorXd &xCoords, const
        Eigen::VectorXd &yCoords, const Eigen::VectorXd &zCoords) {

    this->optimiser = op;
    this->xCoords = xCoords;
    this->yCoords = yCoords;
    this->zCoords = zCoords;

    // Assign optimiser
    this->optimiser = op;

    // Assign an empty Attributes object (always has the same size)
    AttributesPtr att(new Attributes(this->me()));
    this->attributes = att;
}

Road::Road(OptimiserPtr op, const Eigen::RowVectorXd& genome) {
    // First, ensure that the genome length is a multiple of three and that it
    // is equal to the number of design points (intersection points + start and
    // end points)
    int intersectPts = op->getDesignParameters()->getIntersectionPoints() + 2;

    if (genome.size() != 3*(intersectPts)) {
        throw std::invalid_argument("Genome must be 3x the number of design points.");
    }

    this->xCoords.resize(intersectPts);
    this->yCoords.resize(intersectPts);
    this->zCoords.resize(intersectPts);

    Eigen::VectorXi rowIdx(1);
    rowIdx(0) = 0;
    Eigen::VectorXi colIdx = Eigen::VectorXi::LinSpaced(intersectPts,0,
            intersectPts-1);

    igl::slice(genome,rowIdx,colIdx,this->xCoords);
    colIdx = (colIdx.array() + 1).matrix();
    igl::slice(genome,rowIdx,colIdx,this->yCoords);
    colIdx = (colIdx.array() + 1).matrix();
    igl::slice(genome,rowIdx,colIdx,this->zCoords);

    // Assign optimiser
    this->optimiser.reset();
    this->optimiser = op;
}

Road::~Road() {
}

RoadPtr Road::me() {
    return shared_from_this();
}

void Road::designRoad() {
    AttributesPtr att(new Attributes(this->me()));
    this->attributes = att;
    this->computeAlignment();
    this->plotRoadPath();    
    const Eigen::VectorXd& s = this->segments->getDists();
    this->attributes->setLength(s(s.size()));
    this->computeRoadCells();
    this->computeCostElements();
}

void Road::evaluateRoad(bool learning) {
    // Compute unit cost and revenue
    // Compute the following line only once
    this->getAttributes()->setFixedCosts(1.05*(this->getCosts()->getEarthwork()
            + this->getCosts()->getLengthFixed()
            + this->getCosts()->getLocation())
            + this->getCosts()->getAccidentFixed());

    // Sets the unit variable costs (minus fuel for now)
    this->getAttributes()->setUnitVarCosts(
            this->getCosts()->getAccidentVariable()
            + this->getCosts()->getLengthVariable());
    this->getAttributes()->setUnitVarRevenue(this->getCosts()->
            getUnitRevenue());

    this->computeOperating(learning);

    this->getAttributes()->setTotalValueMean(this->attributes->
            getVarProfitIC() + this->attributes->getFixedCosts());
}

void Road::computeOperating(bool learning) {

    OptimiserPtr optPtrShared = this->optimiser.lock();

    switch (optPtrShared->getType()) {
        case Optimiser::SIMPLEPENALTY:
            {
                // The penalty here refers to the road passing through habitat
                // areas. Therefore, the separate penalty related to the actual.
                // population number is already accounted for and is not
                // computed here.
                this->getCosts()->setPenalty(0.0);

                // If there is no uncertainty in the fuel or commodity prices,
                // just treat the operating valuation as a simple annuity. if
                // there is uncertainty, simulate the fuel and commodity prices
                // first to get expected values of one unit of constant use
                // over the entire horizon.
                if ((optPtrShared->getVariableParams()
                        ->getCommoditySDMultipliers().size() == 1)) {
                    double r = optPtrShared->getEconomic()->getRRR();
                    double t = optPtrShared->getEconomic()->getYears();
                    double g = optPtrShared->getTraffic()->getGR();
                    const Eigen::VectorXd& Qs = optPtrShared
                            ->getTrafficProgram()->getFlowRates();
                    const std::vector<VehiclePtr>& vehicles = optPtrShared
                            ->getTraffic()->getVehicles();
                    double Q = Qs(Qs.size());

                    double factor = (1/(r-g) - (1/(r-g))*pow((1+g)/(1+r),t));

                    Eigen::VectorXd fuelPrices;

                    for (int ii = 0; ii < vehicles.size(); ii++) {
                        fuelPrices(ii) = vehicles[ii]->getFuel()->getMean();
                    }

                    // Compute the price of a tonne of raw ore
                    double orePrice = 0.0;

                    const std::vector<CommodityPtr>& commodities = optPtrShared
                            ->getEconomic()->getCommodities();

                    for (int ii = 0; ii < commodities.size(); ii++) {
                        orePrice += commodities[ii]->getOreContent()*
                                commodities[ii]->getMean();
                    }

                    this->getAttributes()->setVarProfitIC(Q*(this
                            ->getAttributes()->getUnitVarCosts() + (this
                            ->getCosts()->getUnitFuelCost()).transpose()*
                            fuelPrices - this->getCosts()->getUnitRevenue()*
                            orePrice)*factor);

                    // There is no variability in the final value of the
                    // operating stage
                    this->getAttributes()->setTotalValueSD(0.0);

                } else {
                    // Call the routine to evaluate operating costs
                    this->computeVarProfitICFixedFlow();

                    // We need to extend this to account for variability. This
                    // will need to take into account covariances and is left
                    // as future work. For now, we set the variability to zero
                    this->getAttributes()->setTotalValueSD(0.0);
                }
            }
            break;

        case Optimiser::MTE:
            {
                // Compute the road value in the MTE case. We progressively
                // learn a surrogate function to limit the calls to the
                // computationally-intensive animal simulation model. This
                // surrogate is stored as a function pointer in the Optimiser
                // class that is called here and updated during the
                // optimisation process. By default, we compute the full model
                // for the best three roads in the GA population to verify the
                // model accuracy.
                // Whether the full model is called now or not is based on
                // whether the function is called from within the population
                // evaluation routine or the surrogate model learning routine.

                // We use the largest, uninhibited traffic flow in the
                // simulations

                // Call the surrogate model or full simulation.

                if (learning) {
                    // Animal penalty component
                    // Full simulation
                    this->computeSimulationPatches();
                    SimulatorPtr simulator(new Simulator(this->me()));
                    this->simulator.reset();
                    this->simulator = simulator;
                    this->simulator->simulateMTE();
                    // Need to write the routine for full simulation in the
                    // Simulation class
                    // NOTE: THIS ROUTINE IS RUN IN PARALLEL!!!!!!!!!!!!!!!!!!!
                    // Profits, populations, penalties, iar etc. are computed
                    // within the full model, and not here as is the case when
                    // we use a surrogate instead.

                } else {
                    // Surrogate model

                    TrafficProgramPtr program = (this->getOptimiser()
                            ->getPrograms())[this->getOptimiser()->getScenario()
                            ->getProgram()];
                    int controls = program->getFlowRates().size();

                    // First compute initial AAR (this is an input to the
                    // surrogate)
                    this->computeSimulationPatches();
                    int noSpecies = this->optimiser.lock()->getSpecies().size();

                    Eigen::VectorXd aar(noSpecies);

                    for (int ii = 0; ii < noSpecies; ii++) {
                        Eigen::VectorXd speciesAARs(controls);
                        this->srp[ii]->computeInitialAAR(speciesAARs);
                        aar[ii] = speciesAARs(speciesAARs.size());
                    }
                    this->attributes->setIAR(aar);

                    // Vector of end populations
                    Eigen::VectorXd endPops(this->optimiser.lock()->
                            getSpecies().size());
                    Eigen::VectorXd endPopsSD(this->optimiser.lock()->
                            getSpecies().size());
                    this->optimiser.lock()->evaluateSurrogateModelMTE(
                            this->me(),endPops,endPopsSD);

                    this->attributes->setEndPopMTE(endPops);
                    this->attributes->setEndPopMTESD(endPopsSD);

                    // Call the routine to evaluate the operating costs
                    this->computeVarProfitICFixedFlow();

                    // The penalty here refers to the end population being below
                    // the required end population. We base the penalty on the
                    // Xth percentile, where we wish the likelihood of the end
                    // populations being above threshold to be X per cent.

                    double penalty = 0.0;

                    for (int ii = 0; ii < noSpecies; ii++) {
                        double roadPopXconf = Utility::NormalCDFInverse((1 -
                                this->optimiser.lock()->
                                getConfidenceInterval()));
                        double threshold = this->optimiser.lock()->
                                getSpecies()[ii]->getThreshold();
                        double perAnimalPenalty = (this->optimiser.lock())->
                                getSpecies()[ii]->getCostPerAnimal();

                        penalty += (double)(threshold > (roadPopXconf*
                                endPopsSD(ii) + endPops(ii)))*(threshold -
                                (roadPopXconf*endPopsSD(ii) + endPops(ii)))*
                                perAnimalPenalty;
                    }
                    this->getCosts()->setPenalty(0.0);
                    this->getAttributes()->setTotalValueSD(0.0);
                }
            }
            break;

        case Optimiser::CONTROLLED:
            {
                // Compute the road value in the ROV case. We progressively
                // learn a surrogate function to limit the calls to the
                // computationally-intensive animal simulation model. This
                // surrogate is stored as a function pointer in the Optimiser
                // class that is called here and updated during the
                // optimisation process. By default, we compute the full model
                // for the best three roads in the GA population to verify the
                // model accuracy.
                // Whether the full model is called now or not is based on
                // whether the function is called from within the population
                // evaluation routine or the surrogate model learning routine.

                // First compute the initial AAR under the full traffic flow
                // (this is an input to the surrogate)
                TrafficProgramPtr program = (this->getOptimiser()
                        ->getPrograms())[this->getOptimiser()->getScenario()
                        ->getProgram()];
                int controls = program->getFlowRates().size();

                int noSpecies = this->optimiser.lock()->getSpecies().size();
                std::vector<Eigen::VectorXd> aar(noSpecies);

                for (int ii = 0; ii < noSpecies; ii++) {
                    Eigen::VectorXd speciesAARs(controls);
                    this->srp[ii]->computeInitialAAR(speciesAARs);
                    aar[ii] = speciesAARs;
                }

                // Call the surrogate model or full simulation.
                if (learning) {
                    // Full simulation
                    SimulatorPtr simulator(new Simulator(this->me()));
                    this->simulator.reset();
                    this->simulator = simulator;
                    this->simulator->simulateROVCR();
                }
                // There is no penalty for this road as traffic is controlled
                // to maintain the populations above critical thresholds.
                this->getCosts()->setPenalty(0.0);
            }
            break;
        default:
            {}
            break;
    }
}

void Road::addSpeciesPatches(SpeciesPtr species) {
    SpeciesRoadPatchesPtr srp(new SpeciesRoadPatches(this->optimiser.lock(),
        species, this->me()));
    this->srp.push_back(srp);
}

// PRIVATE ROUTINES ///////////////////////////////////////////////////////////

void Road::computeAlignment() {
	// Initialise referenced parameters
    //double* designVel = optPtrShared->getDesignParameters()
	//		->getDesignVelocity();

	// First ensure that no two or more successive points have the same x and
	// y values without having the same z value
	for (int ii = 1; ii < this->xCoords.size(); ii++) {
		if ((this->xCoords(ii) == this->xCoords(ii-1)) &&
				(this->yCoords(ii) == this->yCoords(ii-1)) &&
				(this->zCoords(ii) != this->zCoords(ii-1))) {
			throw std::invalid_argument("Impossible road alignment");
		}
	}

    // The code below must be run in order for a valid road
    this->computeHorizontalAlignment();
    this->computeVerticalAlignment();
}

void Road::computeHorizontalAlignment() {
    HorizontalAlignmentPtr ha(new HorizontalAlignment(this->me()));
    this->horizontalAlignment = ha;
    this->horizontalAlignment->computeAlignment();
}

void Road::computeVerticalAlignment() {
    VerticalAlignmentPtr va(new VerticalAlignment(this->me()));
    this->verticalAlignment = va;
    this->verticalAlignment->computeAlignment();
}

void Road::plotRoadPath() {
    // First initialise values
    RoadSegmentsPtr rs(new RoadSegments(this->me()));
    this->segments = rs;
    this->segments->computeSegments();
    this->segments->placeNetwork();
}

void Road::computeRoadCells() {
    RoadCellsPtr rc(new RoadCells(this->me()));
    this->roadCells = rc;
    this->roadCells->computeRoadCells();
}

void Road::computeCostElements() {
    this->costs->computeEarthworkCosts();
    this->costs->computeLocationCosts();
    this->costs->computeLengthCosts();
    this->costs->computeAccidentCosts();
}

void Road::computeSimulationPatches() {
    const std::vector<SpeciesPtr>& species = this->optimiser.lock()
            ->getSpecies();

    this->srp.resize(species.size());

    for (int ii = 0; ii < species.size(); ii++) {
        SpeciesRoadPatchesPtr speciesPatches(new SpeciesRoadPatches(
                this->getOptimiser(),species[ii],this->me()));
        speciesPatches->createSpeciesModel();
        this->srp[ii].reset();
        this->srp[ii] = speciesPatches;
    }
}

void Road::computeVarProfitICFixedFlow() {
    // Call the simulation routine to vary the commodity prices
    // using Monte Carlo simulation. N.B. These commodity
    // prices could represent any sort of benefit or economic
    // feature, so the principle used in this software is
    // actually useful in many other scenarios.
    //
    // N.B. The routine must have been called prior so that the
    // expected and standard deviations of the present value of
    // the uncertainty are available.
    OptimiserPtr optPtrShared = this->optimiser.lock();

    const Eigen::VectorXd& Qs = optPtrShared
            ->getTrafficProgram()->getFlowRates();
    const std::vector<VehiclePtr>& vehicles = optPtrShared
            ->getTraffic()->getVehicles();
    double Q = Qs(Qs.size());

    Eigen::VectorXd fuelExpPV1UnitTraffic;

    for (int ii = 0; ii < vehicles.size(); ii++) {
        fuelExpPV1UnitTraffic(ii) = vehicles[ii]->getFuel()->
                getExpPV();
    }

    // Compute the price of a tonne of raw ore
    double orePrice = 0.0;
    double r = optPtrShared->getEconomic()->getRRR();
    double t = optPtrShared->getEconomic()->getYears();
    double g = optPtrShared->getTraffic()->getGR();
    double factor = (1/(r-g) - (1/(r-g))*pow((1+g)/(1+r),t));

    const std::vector<CommodityPtr>& commodities = optPtrShared
            ->getEconomic()->getCommodities();

    for (int ii = 0; ii < commodities.size(); ii++) {
        orePrice += commodities[ii]->getOreContent()*
                commodities[ii]->getExpPV();
    }

    this->getAttributes()->setVarProfitIC(Q*(this
            ->getAttributes()->getUnitVarCosts()*factor
            + (this->getCosts()->getUnitFuelCost()).transpose()
            *fuelExpPV1UnitTraffic - this->getCosts()
            ->getUnitRevenue()*orePrice));
}
