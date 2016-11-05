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
     * Returns the road calling the simulator
     *
     * @return Road as RoadPtr
     */
    RoadPtr getRoad() {
        return this->road.lock();
    }
    /**
     * Sets the road calling the simulator
     *
     * @param road as RoadPtr
     */
    void setRoad(RoadPtr road) {
        this->road.reset();
        this->road = road;
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
    // These simulation routines are only called when learning the surrogate
    // model used after each generation in the optimisation routine.

    /**
     * Runs the full flow simulation over the entire design horizon for all
     * species encountered by the road.
     *
     * This method computes the expected end populations and their standard
     * deviations that are then used to compute the road value.
     */
    void simulateMTE();

private:
    std::weak_ptr<Road> road;   /**< Road owning simulator */
    Eigen::VectorXd endPops;    /**< End populations from all sims */
    double initAAR;             /**< Initial animals at risk */
    double penalty;             /**< Extinction penalty */
};

#endif
