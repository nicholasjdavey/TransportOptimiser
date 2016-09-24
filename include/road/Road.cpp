#include "../transportbase.h"

Road::Road() {
	// Assign an empty Attributes object (always has the same size)
	AttributesPtr att(new Attributes(this->me()));
	this->attributes = att;
}

Road::Road(OptimiserPtr op, SimulatorPtr sim, std::string testName,
		Eigen::VectorXd* xCoords, Eigen::VectorXd* yCoords,
		Eigen::VectorXd* zCoords) {

	this->optimiser = op;
	this->simulator = sim;
	this->testName = testName;
	this->xCoords = *xCoords;
	this->yCoords = *yCoords;
	this->zCoords = *zCoords;

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

void Road::computeOperating() {

    switch (this->optimiser->getType()) {
        case Optimiser::SIMPLEPENALTY:
            // The penalty here refers to the end population being below the
            // required end population
            this->getCosts()->setPenalty(0.0);

            // If there is no uncertainty in the fuel or commodity prices,
            // just treat the operating valuation as a simple annuity.
            if (this->getOptimiser()->getVariableParams()->getFuelVariable() &&
                    this->getOptimiser()->getVariableParams()
                    ->getCommodityVariable()) {
                double r = this->getOptimiser()->getEconomic()->getRRR();
                double t = this->getOptimiser()->getEconomic()->getYears();
                double g = this->getOptimiser()->getTraffic()->getGR();
                Eigen::VectorXd* Qs = this->getOptimiser()
                        ->getTrafficProgram()->getFlowRates();
                std::vector<VehiclePtr>* vehicles = this->getOptimiser()
                        ->getTraffic()->getVehicles();
                double Q = (*Qs)(Qs->size());

                double factor = (1/(r-g) - (1/(r-g))*pow((1+g)/(1+r),t));

                Eigen::VectorXd fuelPrices;

                for (int ii = 0; ii < vehicles->size(); ii++) {
                    fuelPrices(ii) = (*vehicles)[ii]->getFuel()->getMean();
                }

                // Compute the price of a tonne of raw ore
                double orePrice = 0.0;

                std::vector<CommodityPtr>* commodities = this->getOptimiser()
                        ->getEconomic()->getCommodities();

                for (int ii = 0; ii < commodities->size(); ii++) {
                    orePrice += (*commodities)[ii]->getOreContent()*
                            (*commodities)[ii]->getMean();
                }

                this->getAttributes()->setUnitVarCosts(this->getAttributes()
                        ->getUnitVarCosts() + (this->getCosts()
                        ->getUnitFuelCost())->transpose()*fuelPrices);
                this->getAttributes()->setVarProfitIC(Q*(this->getAttributes()
                        ->getUnitVarCosts() - this->getCosts()->getUnitRevenue()*
                        orePrice)*factor);
            } else {
                // Call the simulation routine to vary the commodity prices
                // using Monte Carlo simulation. N.B. These commodity prices
                // could represent any sort of benefit or economic feature, so
                // the principle used in this software is actually useful in
                // many other scenarios.
            }
            break;
        case Optimiser::MTE:
            break;
        case Optimiser::CONTROLLED:
            break;
        default:
            break;
    }
}

void Road::computeAlignment() {
	// Initialise referenced parameters
	//double* designVel = this->getOptimiser()->getDesignParameters()
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
	Eigen::VectorXd* s = this->segments->getDists();
	this->attributes->setLength((*s)(s->size()));
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
