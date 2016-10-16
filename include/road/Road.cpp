#include "../transportbase.h"

Road::Road() {
	// Assign an empty Attributes object (always has the same size)
	AttributesPtr att(new Attributes(this->me()));
	this->attributes = att;
}

Road::Road(OptimiserPtr op, SimulatorPtr sim, std::string testName,
        const Eigen::VectorXd &xCoords, const Eigen::VectorXd &yCoords,
        const Eigen::VectorXd &zCoords) {

	this->optimiser = op;
	this->simulator = sim;
	this->testName = testName;
    this->xCoords = xCoords;
    this->yCoords = yCoords;
    this->zCoords = zCoords;

	// Assign an empty Attributes object (always has the same size)
	AttributesPtr att(new Attributes(this->me()));
	this->attributes = att;
}

Road::~Road() {
}

RoadPtr Road::me() {
	return shared_from_this();
}

void Road::designRoad() {
    this->computeAlignment();
    this->plotRoadPath();
    this->computeRoadCells();
    this->computeCostElements();
}

void Road::evaluateRoad() {
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
    this->getAttributes()->setUnitVarRevenue(this->getCosts()->getUnitRevenue());

    this->computeOperating();
}

void Road::computeOperating(bool learning) {

    OptimiserPtr optPtrShared = this->optimiser.lock();

    switch (optPtrShared->getType()) {
        case Optimiser::SIMPLEPENALTY:
            {
                // The penalty here refers to the end population being below
                // the required end population
                this->getCosts()->setPenalty(0.0);

                // If there is no uncertainty in the fuel or commodity prices,
                // just treat the operating valuation as a simple annuity. if
                // there is uncertainty, simulate the fuel and commodity prices
                // first to get expected values of one unit of constant use
                // over the entire horizon.
                if ((optPtrShared->getVariableParams()->getFuelVariable()
                        .size() == 1) && (optPtrShared->getVariableParams()
                        ->getCommodityVariable().size() == 1)) {
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
                } else {
                    // Call the simulation routine to vary the commodity prices
                    // using Monte Carlo simulation. N.B. These commodity
                    // prices could represent any sort of benefit or economic
                    // feature, so the principle used in this software is
                    // actually useful in many other scenarios.
                    //
                    // The routine must have been called prior
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

                // We use the largest, uninhibited traffic flow in the
                // simulations
                TrafficProgramPtr program = (this->getOptimiser()
                        ->getPrograms())[this->getOptimiser()->getScenario()
                        ->getProgram()];
                int controls = program->getFlowRates().size();

                // First compute initial AAR (this is an input to the
                // surrogate)
                this->computeSimulationPatches();
                int noSpecies = this->optimiser.lock()->getSpecies().size();

                std::vector<Eigen::VectorXd> aar(noSpecies);

                for (int ii = 0; ii < noSpecies; ii++) {
                    Eigen::VectorXd speciesAARs(controls);
                    this->srp[ii]->computeInitialAAR(speciesAARs);
                    aar[ii] = speciesAARs;
                }

                // Make a decision to simulate the model. As this function is
                // in a multithreaded environment, we need to put a mutex around
                // any calls that
            }
            break;            
        case Optimiser::CONTROLLED:
            {}
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
	this->plotRoadPath();
    const Eigen::VectorXd& s = this->segments->getDists();
    this->attributes->setLength(s(s.size()));
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
