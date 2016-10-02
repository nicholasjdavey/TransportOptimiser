#ifndef VARIABLEPARAMETERS_H
#define VARIABLEPARAMETERS_H

/**
 * Class for storing details that can be varied for sensitivity analysis
 */
class VariableParameters : public std::enable_shared_from_this<VariableParameters> {

public:
	// CONSTRUCTORS AND DESTRUCTORS //////////////////////////////////////////////

	/**
	 * Constructor
	 *
	 * Constructs a %VariableParameters object with default values.
	 */
	VariableParameters(Eigen::VectorXd* popLevels, bool bridge,
		Eigen::VectorXd* hp, Eigen::VectorXd* l, Eigen::VectorXd* b,
		bool pgr, bool f, bool c);
	/**
	 * Destructor
	 */
	~VariableParameters();

	// ACCESSORS /////////////////////////////////////////////////////////////////
	/**
	 * Returns the different population levels as a percentage of starting pop.
	 *
	 * @return Population level as Eigen::VectorXd
	 */
	Eigen::VectorXd* getPopulationLevels() {
		return &this->populationLevels;
	}
	/**
	 * Sets the population levels
	 *
	 * @param levels as Eigen::VectorXd
	 */
	void setPopulationLevels(Eigen::VectorXd* levels) {
		this->populationLevels = *levels;
	}

	/**
     * Returns animal bridge usage scenarios
	 *
     * @return Animal bridge usage scenarios as Eigen::VectorXd*
	 */
    Eigen::VectorXd* getBridge() {
        return &this->animalBridge;
	}
	/**
     * Returns animal bridge usage scenarios
	 *
     * @param bridge as Eigen::VectorXd*
	 */
    void setBridge(Eigen::VectorXd* bridge) {
        this->animalBridge = *bridge;
	}

	/**
	 * Returns the number of standard deviations away from the mean the habitat
	 * preference used is (for sensitivity analysis).
	 *
	 * @return Habitat preference standard deviations as Eigen::VectorXd
	 */
	Eigen::VectorXd* getHabPref() {
		return &this->habPref;
	}
	/**
	 * Sets the number of standard deviations away from the mean the habitat
	 * preference used is (for sensitivity analysis).
	 *
	 * @param habPref as Eigen::VectorXd
	 */
	void setHabPref(Eigen::VectorXd* habPref) {
		this->habPref = *habPref;
	}

	/**
	 * Returns the number of standard deviations away from the mean the movement
	 * propensity parameter used is (for sensitivity analysis).
	 *
	 * @return lambda as Eigen::VectorXd
	 */
	Eigen::VectorXd* getLambda() {
		return &this->lambda;
	}
	/**
	 * Sets the number of standard deviations away from the mean the movement
	 * propensity parameter used is (for sensitivity analysis).
	 *
	 * @param lambda as Eigen::VectorXd
	 */
	void setLambda(Eigen::VectorXd* lambda) {
		this->lambda = *lambda;
	}

	/**
	 * Returns the number of standard deviations away from the mean the ranging
	 * coefficient used is (for sensitivity analysis).
	 *
	 * @return Beta as Eigen::VectorXd
	 */
	Eigen::VectorXd* getBeta() {
		return &this->beta;
	}
	/**
	 * Sets the number of standard deviations away from the mean the ranging
	 * coefficient used is (for sensitivity analysis).
	 *
	 * @param beta as Eigen::VectorXd
	 */
	void setBeta(Eigen::VectorXd* beta) {
		this->beta = *beta;
	}

	/**
     * Returns the population growth rate standard deviation multiplier
     *
     * @return Growth rate standard deviation multiplier as Eigen::VectorXd*
	 */
    Eigen::VectorXd* getGrowthRateSDMultipliers() {
        return &this->popGR;
	}
	/**
     * Sets the population growth rate standard deviation multiplier
	 *
     * @param rate as Eigen::VectorXd*
	 */
    void setGrowthRateSDMultipliers(Eigen::VectorXd* rate) {
        this->popGR = *rate;
	}

	/**
     * Sets the fuel price standard deviation multiplier
	 *
     * @return Fuell price standard deviation multiplier as Eigen::VectorXd*
	 */
    Eigen::VectorXd* getFuelVariable() {
        return &this->fuel;
	}
	/**
     * Sets the fuel price standard deviation multiplier
	 *
     * @param fuel as Eigen::VectorXd*
	 */
    void setFuelVariable(Eigen::VectorXd* fuel) {
        this->fuel = *fuel;
	}

	/**
     * Returns the commodity price standard deviation multiplier
	 *
     * @return Commodity price standard deviation multiplier as Eigen::VectorXd*
	 */
    Eigen::VectorXd* getCommodityVariable() {
        return &this->commodity;
	}
	/**
     * Sets the commodity price standard deviation multiplier
	 *
     * @param commodity as Eigen::VectorXd*
	 */
    void setCommodityVariable(Eigen::VectorXd* commodity) {
        this->commodity = *commodity;
    }

	// STATIC ROUTINES ///////////////////////////////////////////////////////////

	// CALCULATION ROUTINES //////////////////////////////////////////////////////

	private:
        Eigen::VectorXd populationLevels;
		Eigen::VectorXd habPref;
		Eigen::VectorXd lambda;
        Eigen::VectorXd beta;
        Eigen::VectorXd animalBridge;
        Eigen::VectorXd popGR;
        Eigen::VectorXd fuel;
        Eigen::VectorXd commodity;
};

#endif
