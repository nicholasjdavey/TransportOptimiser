#ifndef MONTECARLOROV_H
#define MONTECARLOROV_H

class PolicyMap;
typedef std::shared_ptr<PolicyMap> PolicyMapPtr;

class State;
typedef std::shared_ptr<State> StatePtr;

class Road;
typedef std::shared_ptr<Road> RoadPtr;

class MonteCarloROV;
typedef std::shared_ptr<MonteCarloROV> MonteCarloROVPtr;

/**
 * Class for managing ROV analysis with Monte Carlo simulation
 */
class MonteCarloROV : public std::enable_shared_from_this<MonteCarloROV> {

public:
	// CONSTRUCTORS AND DESTRUCTORS ///////////////////////////////////////////

	/**
	 * Constructor I
	 *
	 * Constructs an empty ROV object
	 */
	MonteCarloROV();

	/**
	 * Destructor
	 */
	~MonteCarloROV();

	// ACCESSORS //////////////////////////////////////////////////////////////

	/**
	 * Returns the State
	 *
	 * @return State as StatePtr
	 */
	StatePtr getState() {
		return this->state;
	}
	/**
	 * Sets the State
	 *
	 * @param state as StatePtr
	 */
	void setState(StatePtr state) {
		this->state = state;
	}

	/**
	 * Returns the ROV policy map
	 *
	 * @return PolicyMap as PolicyMapPtr
	 */
	PolicyMapPtr getPolicyMap() {
		return this->policyMap;
	}
	/**
	 * Sets the ROV policy map
	 *
	 * @param pm as PolicyMapPtr
	 */
	void setPolicyMap(PolicyMapPtr pm) {
		this->policyMap = pm;
	}

	/**
	 * Returns the random number generator
	 *
	 * @return Random number generator as std::string
	 */
	std::string getRandomGenerator() {
		return this->randGenerator;
	}
	/**
	 * Sets the random number generator
	 *
	 * @param rand as std::string
	 */
	void setRandomGenerator(std::string rand) {
		this->randGenerator = rand;
	}

	/**
	 * Returns the seeds for the controls
	 *
	 * @return Seeds as std::vector<double>*
	 */
	std::vector<double>* getControlSeeds() {
		return &this->seedsControl;
	}
	/**
	 * Sets the seeds for the controls
	 *
	 * @param seeds as std::vector<double>*
	 */
	void setControlSeeds(std::vector<double>* seeds) {
		this->seedsControl = *seeds;
	}

	/**
	 * Returns the seeds for exogenous uncertainties
	 *
	 * @return Seeds as std::vector<double>*
	 */
	std::vector<double>* getExogenousSeeds() {
		return &this->seedsExogenous;
	}
	/**
	 * Sets the seeds for exogenous uncertainties
	 *
	 * @param seeds as std::vector<double>*
	 */
	void setExogenousSeeds(std::vector<double>* seeds) {
		this->seedsExogenous = *seeds;
	}

	/**
	 * Returns the seeds for endogenous uncertainties
	 *
	 * @return Seeds as std::vector<double>*
	 */
	std::vector<double>* getEndogenousSeeds() {
		return &this->seedsEndogenous;
	}
	/**
	 * Sets the seeds for endogenous uncertainties
	 *
	 * @param seeds as std::vector<double>*
	 */
	void setEndogenousSeeds(std::vector<double>* seeds) {
		this->seedsEndogenous = *seeds;
	}

	/**
	 * Returns the final valuation
	 *
	 * @return Value as double
	 */
	double getValue() {
		return this->value;
	}
	/**
	 * Sets the final valuation
	 *
	 * @param value as double
	 */
	void setValue(double value) {
		this->value = value;
	}

	/**
	 * Returns the values of every path generated and optimally controlled
	 *
	 * @return End values as Eigen::VectorXd*
	 */
	Eigen::VectorXd* getValues() {
		return &this->values;
	}
	/**
	 * Sets the values of every path generated and optimally controlled
	 *
	 * @param values as Eigen::VectorXd*
	 */
	void setValues(Eigen::VectorXd* values) {
		this->values = *values;
	}

	// STATIC ROUTINES ////////////////////////////////////////////////////////

	// CALCULATION ROUTINES ///////////////////////////////////////////////////

private:
	StatePtr state;						/**< State object used in simulation */
	PolicyMapPtr policyMap;				/**< Generated policy map */
	std::string randGenerator;			/**< Random number generator used */
	std::vector<double> seedsControl;	/**< Seeds for control randomisation */
	std::vector<double> seedsExogenous;	/**< Seeds for exogenous uncertainty */
	std::vector<double> seedsEndogenous;/**< Seeds for endogenous uncertainty */
	double value;						/**< Computed value */
	Eigen::VectorXd values;				/**< Values of all paths (controlled) generated */
};

#endif
