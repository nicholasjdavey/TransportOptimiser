#ifndef VARIABLEPARAMETERS_H
#define VARIABLEPARAMETERS_H

class VariableParameters;
typedef std::shared_ptr<VariableParameters> VariableParametersPtr;

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
    VariableParameters(const Eigen::VectorXd& popLevels, const Eigen::VectorXi
            &bridge, const Eigen::VectorXd& hp, const Eigen::VectorXd& l, const
            Eigen::VectorXd& b, const Eigen::VectorXd& pgr, const
            Eigen::VectorXd& pgrsd, const Eigen::VectorXd& c, const
            Eigen::VectorXd& csd);
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
     * @return Animal bridge usage scenarios as const Eigen::VectorXi&
     */
    const Eigen::VectorXi& getBridge() {
        return this->animalBridge;
    }
    /**
     * Returns animal bridge usage scenarios
     *
     * @param bridge as const Eigen::VectorXi&
     */
    void setBridge(const Eigen::VectorXi& bridge) {
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
     * Returns the multipliers of the base growth rates used in the animal model
     *
     * @return Growth rate mean multipliers as const Eigen::VectorXd&
     */
    const Eigen::VectorXd& getGrowthRatesMultipliers() {
        return this->popGR;
    }
    /**
     * Sets the multipliers of the base growth rates used in the animal model
     *
     * @param rates as const Eigen::VectorXd&
     */
    void setGrowthRatesMultipliers(const Eigen::VectorXd& rates) {
        this->popGR = rates;
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
     * Returns the commodity price means
     *
     * @return Commodity price means as const Eigen::VectorXd&
     */
    const Eigen::VectorXd& getCommodityMultipliers() {
        return this->commodity;
    }
    /**
     * Sets the commodity price means
     *
     * @param commodity as const Eigen::VectorXd&
     */
    void setCommodityMultipliers(const Eigen::VectorXd& commodity) {
        this->commodity = commodity;
    }

    /**
     * Returns the commodity price standard deviation multiplier
     *
     * @note This vector's first element is always 0 (i.e. no
     * uncertainty case).
     * @return Commodity price standard deviation multiplier as const Eigen::VectorXd&
     */
    const Eigen::VectorXd& getCommoditySDMultipliers() {
        return this->commoditySD;
    }
    /**
     * Sets the commodity price standard deviation multiplier
     *
     * @param commodity as const Eigen::VectorXd&
     */
    void setCommoditySDMultipliers(const Eigen::VectorXd& commoditysd) {
        this->commoditySD = commoditysd;
    }

    /**
     * Returns the maximum number of animal bridges to use
     *
     * @return Animal bridges to use as const Eigen::VectorXi&
     */
    const Eigen::VectorXi& getAnimalBridge() {
        return this->animalBridge;
    }
    /**
     * Sets the maximum number of animal bridges to use
     *
     * @param bridge as const Eigen::VectorXi& bridge&
     */
    void setAnimalBridge(const Eigen::VectorXi& bridge) {
        this->animalBridge = bridge;
    }

    // STATIC ROUTINES ///////////////////////////////////////////////////////////

    // CALCULATION ROUTINES //////////////////////////////////////////////////////

    private:
    // Sensitivity variables
    Eigen::VectorXd populationLevels;   /**< Percentage required survival levels to test */
    Eigen::VectorXd habPref;            /**< Standard deviations away from mean habitat preferences to use */
    Eigen::VectorXd lambda;             /**< Standard deviations away from mean lambdas to use */
    Eigen::VectorXd beta;               /**< Standard deviations away from mean betas to use */
    Eigen::VectorXi animalBridge;       /**< Maximum number of animal bridges to use */
    // Stochastic variables:
    Eigen::VectorXd popGR;              /**< Percentage of base population growth rates to use */
    Eigen::VectorXd popGRSD;            /**< Population growth rate standard deviation scalings to use */
    Eigen::VectorXd commodity;          /**< Percentage of base commodity mean price to use */
    Eigen::VectorXd commoditySD;        /**< Commodity price standard deviation scaling to use */
};

#endif
