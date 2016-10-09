#ifndef ATTRIBUTES_H
#define ATTRIBUTES_H

class Road;
typedef std::shared_ptr<Road> RoadPtr;

class Attributes;
typedef std::shared_ptr<Attributes> AttributesPtr;

/**
 * Class for managing road attributes
 */
class Attributes : public std::enable_shared_from_this<Attributes> {

public:
	// CONSTRUCTORS AND DESTRUCTORS ///////////////////////////////////////////

	/**
	 * Constructor I
	 *
	 * Constructs an empty %Attributes object
	 */
	Attributes(RoadPtr road);

	/**
	 * Constructor II
	 *
	 * Constructs an %Attributes object with assigned values
	 */
	Attributes(double iar, double endpop, double uvc, double uvr,
			double length, double vpic, double tvmte, double tvrov,
			RoadPtr road);

	/**
	 * Destructor
	 */
	~Attributes();

	// ACCESSORS //////////////////////////////////////////////////////////////

	/**
	 * Returns the road
	 *
	 * @return Road as RoadPtr
	 */
	RoadPtr getRoad() {
		return this->road;
	}
	/**
	 * Sets the road
	 *
	 * @param road as RoadPtr
	 */
	void setRoad(RoadPtr road) {
        this->road.reset();
		this->road = road;
	}

	/**
	 * Returns the initial animals at risk
	 *
	 * @return IAR as double
	 */
	double getIAR() {
		return this->initAAR;
	}
	/**
	 * Sets the initial animals at risk
	 *
	 * @param iar as double
	 */
	void setIAR(double iar) {
		this->initAAR = iar;
	}

	/**
	 * Returns the end population at full traffic flow
	 *
	 * @return End population as double
	 */
	double getEndPopMTE() {
		return this->endPopMTE;
	}
	/**
	 * Sets the end population at full traffic flow
	 *
	 * @param endpop as double
	 */
	void setEndPopMTE(double endpop) {
		this->endPopMTE = endpop;
	}

    /**
     * Returns the road fixed costs
     *
     * @return Fixed costs as double
     */
    double getFixedCosts() {
        return this->fixedCosts;
    }
    /**
     * Sets the road fixed costs
     *
     * @param fc as double
     */
    void setFixedCosts(double fc) {
        this->fixedCosts = fc;
    }

	/**
	 * Returns the unit variable costs
	 *
	 * @return Unit variable costs as double
	 */
	double getUnitVarCosts() {
		return this->unitVarCosts;
	}
	/**
	 * Sets the unit variable costs
	 *
	 * @param uvc as double
	 */
	void setUnitVarCosts(double uvc) {
		this->unitVarCosts = uvc;
	}

	/**
	 * Returns the unit variable revenue
	 *
	 * @return Unit variable revenue as double
	 */
	double getUnitVarRevenue() {
		return this->unitVarRevenue;
	}
	/**
	 * Sets the unit variable revenue
	 *
	 * @param uvr as double
	 */
	void setUnitVarRevenue(double uvr) {
		this->unitVarRevenue = uvr;
	}

	/**
	 * Returns the road length
	 *
	 * @return Length as double
	 */
	double getLength() {
		return this->length;
	}
	/**
	 * Sets the road length
	 *
	 * @param len as double
	 */
	void setLength(double len) {
		this->length = len;
	}

	/**
	 * Returns the variable profit using ROV
	 *
	 * @return ROV profit as double
	 */
	double getVarProfitIC() {
		return this->varProfitIC;
	}
	/**
	 * Sets the variable profit using ROV
	 *
	 * @param vpic as double
	 */
	void setVarProfitIC(double vpic) {
		this->varProfitIC = vpic;
	}

	/**
	 * Returns the total value with fixed traffic flow
	 *
	 * @return Total value as double
	 */
	double getTotalValueMTE() {
		return this->totalValueMTE;
	}
	/**
	 * Sets the total value with fixed traffic flow
	 *
	 * @param tvmte as double
	 */
	void setTotalValueMTE(double tvmte) {
		this->totalValueMTE = tvmte;
	}

	/**
	 * Returns the total value with ROV
	 *
	 * @return Total value as double
	 */
	double getTotalValueROV() {
		return this->totalValueROV;
	}
	/**
	 * Sets the total value with ROV
	 *
	 * @param tvrov as double
	 */
	void setTotalValueROV(double tvrov) {
		this->totalValueROV = tvrov;
	}
	// STATIC ROUTINES ////////////////////////////////////////////////////////

	// CALCULATION ROUTINES ///////////////////////////////////////////////////
	
private:
	RoadPtr road;					/**< Road with these attributes */
	double initAAR;					/**< Initial animals at risk */
	double endPopMTE;				/**< End population if constant full flow */
    double fixedCosts;              /**< Fixed road costs */
    double unitVarCosts;			/**< Variable costs per unit traffic per hour per year */
	double unitVarRevenue;			/**< Variable revenue per unit traffic per hour per year */
	double length;					/**< Total road length (m) */
    double varProfitIC;				/**< Operating value */
	double totalValueMTE;			/**< Overall value, MTE */
	double totalValueROV;			/**< Overall value, ROV */
};

#endif
