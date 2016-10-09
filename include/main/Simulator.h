#ifndef SIMULATOR_H
#define SIMULATOR_H

class MonteCarloROV;
typedef std::shared_ptr<MonteCarloROV> MonteCarloROVPtr;

class Road;
typedef std::shared_ptr<Road> RoadPtr;

class Simulator;
typedef std::shared_ptr<Simulator> SimulatorPtr;

/**
 * Class for managing simulations
 */
class Simulator : public MonteCarloROV, 
		public std::enable_shared_from_this<Simulator> {

public:
	// CONSTRUCTORS AND DESTRUCTORS ///////////////////////////////////////////

	/**
	 * Constructor I
	 *
	 * Default, blank constructor
	 */
	Simulator();

	/**
	 * Constructor II
	 *
	 * Pass the Road as an argument for initialisation
	 */
	Simulator(RoadPtr road);

	/**
	 * Destructor
	 */
	~Simulator();

	// ACCESSORS //////////////////////////////////////////////////////////////

	/**
	 * Returns the mean end population
	 *
	 * @return Mean end population as double
	 */
	double getEndPop() {
		return this->endPop;
	}
	/**
	 * Sets the mean end population
	 *
	 * @param endPop as double
	 */
	void setEndPop(double endPop) {
		this->endPop = endPop;
	}

	/**
	 * Returns the end population from all runs
	 *
     * @return End populations as const Eigen::VectorXd&
	 */
    const Eigen::VectorXd& getEndPops() {
        return this->endPops;
	}
	/**
	 * Sets the end population from all runs
	 *
     * @param endPops as const Eigen::VectorXd&
	 */
    void setEndPops(const Eigen::VectorXd& endPops) {
        this->endPops = endPops;
	}

	/**
	 * Returns the initial animals at risk
	 *
	 * @return Initial animals at risk as double
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
	 * Returns the extinction dollar cost penalty
	 *
	 * @return Extionction penalty as double
	 */
	double getPenalty() {
		return this->penalty;
	}
	/**
	 * Sets the extinction penalty
	 *
	 * @param ep as double
	 */
	void setPenalty(double ep) {
		this->penalty = ep;
	}
	// STATIC ROUTINES ////////////////////////////////////////////////////////

	// CALCULATION ROUTINES ///////////////////////////////////////////////////

private:
	double endPop;						/**< Mean end population */
	Eigen::VectorXd endPops;		/**< End populations from all sims */
	double initAAR;						/**< Initial animals at risk */
	double penalty;						/**< Extinction penalty */
};

#endif
