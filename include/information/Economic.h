#ifndef ECONOMIC_H
#define ECONOMIC_H

class Commodity;
typedef std::shared_ptr<Commodity> CommodityPtr;

class Fuel;
typedef std::shared_ptr<Fuel> FuelPtr;

class Economic;
typedef std::shared_ptr<Economic> EconomicPtr;

/**
 * Class for managing economic information
 */
class Economic : public std::enable_shared_from_this<Economic> {

public:
	// CONSTRUCTORS AND DESTRUCTORS ///////////////////////////////////////////
	/**
	 * Constructor I
	 *
	 * Constructs an empty %Economic object
	 */
	Economic();

	/**
	 * Constructor II
	 *
	 * Constructs an %Economic object with assigned values
	 */
    Economic(const std::vector<CommodityPtr>& commodities,
            const std::vector<FuelPtr>& fuels, double rr, double ny);

	/**
	 * Destructor
	 */
	~Economic();

	// ACCESSORS //////////////////////////////////////////////////////////////
	
	/**
	 * Returns the commodities used
	 *
     * @return Commodities as const std::vector<CommodityPtr>&
	 */
    const std::vector<CommodityPtr>& getCommodities() {
        return this->commodities;
	}
	/**
	 * Sets the commodities
	 *
     * @param comm as const std::vector<CommodityPtr>&
	 */
    void setCommodities(const std::vector<CommodityPtr>& comm) {
        this->commodities = comm;
	}

	/**
	 * Returns the fuels
	 *
     * @return Fuels as const std::vector<FuelPtr>&
	 */
    const std::vector<FuelPtr>& getFuels() {
        return this->fuels;
	}
	/**
	 * Sets the fuels
	 *
     * @param fuels as const std::vector<FuelPtr>&
	 */
    void setFuels(const std::vector<FuelPtr>& fuels) {
        this->fuels = fuels;
	}

	/**
	 * Returns the requried rate of return
	 *
	 * @return Required rate of return p.a. as double
	 */
	double getRRR() {
		return this->reqRate;
	}
	/**
	 * Sets the required rate of return
	 *
	 * @param rrr as double
	 */
	void setRRR(double rrr) {
		this->reqRate = rrr;
	}

	/**
	 * Returns the design horizon in years
	 *
	 * @return Design horizon as double
	 */
	double getYears() {
		return this->nYears;
	}
	/**
	 * Sets the design horizon in years
	 *
	 * @param years as double
	 */
	void setYears(double years) {
		this->nYears = years;
	}

	// STATIC ROUTINES ////////////////////////////////////////////////////////

	// CALCULATION ROUTINES ///////////////////////////////////////////////////

private:
	std::vector<CommodityPtr> commodities;	/**< Relevant commodities */
	std::vector<FuelPtr> fuels;		/**< Relevant fuels */
	double reqRate;				/**< Required rate of return p.a. */
	double nYears;				/**< Design horizon */
};

#endif
