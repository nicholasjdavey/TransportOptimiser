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
	 * Returns whether the solution will contain an animal bridge.
	 *
	 * @return Existence of a bridge as bool
	 */
	bool getBridge() {
		return this->animalBridge;
	}
	/**
	 * Sets whether the solution will contain an animal bridge.
	 *
	 * @param bridge as bool
	 */
	void setBridge(bool bridge) {
		this->animalBridge = bridge;
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
	 * Returns whether the population growth rate is stochastic. The growth rate
	 * is used in a logistic growth rate model where the population is split
	 * into distinct patches that are analysed separately.
	 *
	 * @return Growth rate stochasticity as bool
	 */
	bool getGrowthRateVariable() {
		return this->popGR;
	}
	/**
	 * Sets whether the population growth rate is stochastic.
	 *
	 * @param rate as bool
	 */
	void setGrowthRateVariable(bool rate) {
		this->popGR = rate;
	}

	/**
	 * Returns whether the fuel price is stochastic or fixed. If stochastic, the
	 * fuel price is mean reverting.
	 *
	 * @return Stochasticity as bool
	 */
	bool getFuelVariable() {
		return this->fuel;
	}
	/**
	 * Sets whether the fuel price is stochastic or fixed.
	 *
	 * @param fuel as bool
	 */
	void setFuelVariable(bool fuel) {
		this->fuel = fuel;
	}

	/**
	 * Returns whether the commodity price is stochastic or fixed. If stochastic,
	 * the commodity price is mean reverting.
	 *
	 * @return Stochasticity as bool
	 */
	bool getCommodityVariable() {
		return this->commodity;
	}
	/**
	 * Sets whether the commodity price is stochastic or fixed.
	 *
	 * @param commodity as bool
	 */
	void setCommodityVariable(bool commodity) {
		this->commodity = commodity;
	}

	// STATIC ROUTINES ///////////////////////////////////////////////////////////

	// CALCULATION ROUTINES //////////////////////////////////////////////////////

	private:
        Eigen::VectorXd populationLevels;
		Eigen::VectorXd habPref;
		Eigen::VectorXd lambda;
		Eigen::VectorXd beta;        
        bool animalBridge;
		bool popGR;
		bool fuel;
		bool commodity;
};

#endif
