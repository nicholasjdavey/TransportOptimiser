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
    VariableParameters(const Eigen::VectorXd& popLevels, const Eigen::VectorXd&
        bridge, const Eigen::VectorXd& hp, const Eigen::VectorXd& l, const
        Eigen::VectorXd& b, const Eigen::VectorXd& pgr, const Eigen::VectorXd& f,
        const Eigen::VectorXd& c);
	/**
	 * Destructor
	 */
	~VariableParameters();

	// ACCESSORS /////////////////////////////////////////////////////////////////
	/**
	 * Returns the different population levels as a percentage of starting pop.
	 *
     * @return Population level as const Eigen::VectorXd&
	 */
    const Eigen::VectorXd& getPopulationLevels() {
        return this->populationLevels;
	}
	/**
	 * Sets the population levels
	 *
     * @param levels as const Eigen::VectorXd&
	 */
    void setPopulationLevels(const Eigen::VectorXd& levels) {
        this->populationLevels = levels;
	}

	/**
     * Returns animal bridge usage scenarios
	 *
     * @return Animal bridge usage scenarios as const Eigen::VectorXd&
	 */
    const Eigen::VectorXd& getBridge() {
        return this->animalBridge;
	}
	/**
     * Returns animal bridge usage scenarios
	 *
     * @param bridge as const Eigen::VectorXd&
	 */
    void setBridge(const Eigen::VectorXd& bridge) {
        this->animalBridge = bridge;
	}

	/**
	 * Returns the number of standard deviations away from the mean the habitat
	 * preference used is (for sensitivity analysis).
	 *
     * @return Habitat preference standard deviations as const Eigen::VectorXd&
	 */
    const Eigen::VectorXd& getHabPref() {
        return this->habPref;
	}
	/**
	 * Sets the number of standard deviations away from the mean the habitat
	 * preference used is (for sensitivity analysis).
	 *
     * @param habPref as const Eigen::VectorXd&
	 */
    void setHabPref(const Eigen::VectorXd habPref) {
        this->habPref = habPref;
	}

	/**
	 * Returns the number of standard deviations away from the mean the movement
	 * propensity parameter used is (for sensitivity analysis).
	 *
     * @return lambda as const Eigen::VectorXd&
	 */
    const Eigen::VectorXd& getLambda() {
        return this->lambda;
	}
	/**
	 * Sets the number of standard deviations away from the mean the movement
	 * propensity parameter used is (for sensitivity analysis).
	 *
     * @param lambda as const Eigen::VectorXd&
	 */
    void setLambda(const Eigen::VectorXd& lambda) {
        this->lambda = lambda;
	}

	/**
	 * Returns the number of standard deviations away from the mean the ranging
	 * coefficient used is (for sensitivity analysis).
	 *
     * @return Beta as const Eigen::VectorXd&
	 */
    const Eigen::VectorXd& getBeta() {
        return this->beta;
	}
	/**
	 * Sets the number of standard deviations away from the mean the ranging
	 * coefficient used is (for sensitivity analysis).
	 *
     * @param beta as const Eigen::VectorXd&
	 */
    void setBeta(const Eigen::VectorXd& beta) {
        this->beta = beta;
	}

	/**
     * Returns the population growth rate standard deviation multiplier
     *
     * @return Growth rate standard deviation multiplier as const Eigen::VectorXd&
	 */
    const Eigen::VectorXd& getGrowthRateSDMultipliers() {
        return this->popGR;
	}
	/**
     * Sets the population growth rate standard deviation multiplier
	 *
     * @param rate as const Eigen::VectorXd&
	 */
    void setGrowthRateSDMultipliers(const Eigen::VectorXd rate) {
        this->popGR = rate;
	}

	/**
     * Sets the fuel price standard deviation multiplier
	 *
     * @return Fuell price standard deviation multiplier as const Eigen::VectorXd&
	 */
    const Eigen::VectorXd& getFuelVariable() {
        return this->fuel;
	}
	/**
     * Sets the fuel price standard deviation multiplier
	 *
     * @param fuel as const Eigen::VectorXd&
	 */
    void setFuelVariable(const Eigen::VectorXd& fuel) {
        this->fuel = fuel;
	}

	/**
     * Returns the commodity price standard deviation multiplier
	 *
     * @return Commodity price standard deviation multiplier as const Eigen::VectorXd&
	 */
    const Eigen::VectorXd& getCommodityVariable() {
        return this->commodity;
	}
	/**
     * Sets the commodity price standard deviation multiplier
	 *
     * @param commodity as const Eigen::VectorXd&
	 */
    void setCommodityVariable(const Eigen::VectorXd& commodity) {
        this->commodity = commodity;
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
