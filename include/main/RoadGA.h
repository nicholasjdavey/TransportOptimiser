#ifndef ROADGA_H
#define ROADGA_H

class Optimiser;
typedef std::shared_ptr<Optimiser> OptimiserPtr;

class RoadGA;
typedef std::shared_ptr<RoadGA> RoadGAPtr;

class Road;
typedef std::shared_ptr<Road> RoadPtr;

/**
 * Class for managing the optimal road design process using Genetic Algorithms
 *
 * @note Based on the road design procedure of Jong and Schonfeld (2003)
 */
class RoadGA : public Optimiser,
        public std::enable_shared_from_this<RoadGA> {

public:
    // ENUMERATIONS ///////////////////////////////////////////////////////////
    typedef enum {
        TOURNAMENT      /**< Tournament selection routine */
    } Selection;

    // CONSTRUCTORS AND DESTRUCTORS ///////////////////////////////////////////

    /**
     * Constructor I
     *
     * Constructs a RoadGA object with default values.
     */
    RoadGA();

    /**
     * Constructor II
     *
     * Constructs a RoadGA object with assigned values
     */
    RoadGA(const std::vector<TrafficProgramPtr>& programs, OtherInputsPtr
            oInputs, DesignParametersPtr desParams, EarthworkCostsPtr
            earthworks, UnitCostsPtr unitCosts, VariableParametersPtr
            varParams, const std::vector<SpeciesPtr>& species, EconomicPtr
            economic, TrafficPtr traffic, RegionPtr region, double mr, unsigned
            long cf, unsigned long gens, unsigned long popSize, double stopTol,
            double confInt, double confLvl, unsigned long habGridRes,
            std::string solScheme, unsigned long noRuns, Optimiser::Type type,
            double elite, double scale);

    // ACCESSORS //////////////////////////////////////////////////////////////

    // STATIC ROUTINES ////////////////////////////////////////////////////////

    // CALCULATION ROUTINES ///////////////////////////////////////////////////

    /** Virtual Routines *****************************************************/

    /**
     * Creates the initial population used in the optimisation process
     *
     * Creates the initial population for the genetic algorithm that will
     * evolve into a solution.
     *
     * @note This function overrides any base class function of the same name.
     */
    virtual void creation();

    /**
     * Performs crossover for the population used in the optimisation process
     *
     * @param (input) parentsIdx as Eigen::VectorXi&
     * @param (output) children as Eigen::MatrixXd&
     *
     * @note This function overrides any base class function of the same name.
     */
    virtual void crossover(const Eigen::VectorXi &parentsIdx,
            Eigen::MatrixXd& children);

    /**
     * Mutates population used in the optimisation process
     *
     * @param (input) parentsIds as Eigen::VectorXi&
     * @param (output) children as Eigen::MatrixXd&
     *
     * @note This function overrides any base class function of the same name.
     */
    virtual void mutation(const Eigen::VectorXi &parentsIdx,
            Eigen::MatrixXd& children);

    /**
     * Creates the elite children from the current population
     *
     * @param (input) parentsIdx as Eigen::VectorXi&
     * @param (output) children as Eigen::MatrixXd&
     */
    virtual void elite(const Eigen::VectorXi &parentsIdx,
            Eigen::MatrixXd& children);
    /**
     * Runs the optimisation algorithm to devise the best Road
     */
    virtual void optimise();

    /**
     * Evaluates a single generation of the GA
     */
    virtual void evaluateGeneration();

    /**
     * Routine to select between individuals to become parents for the GA
     */
    virtual void selection(Eigen::VectorXi &pc, Eigen::VectorXi &pm,
            Eigen::VectorXi &pe, RoadGA::Selection selector =
            RoadGA::TOURNAMENT);

    /**
     * Returns the outputs at specific stages of the GA
     *
     * This function saves the best road cost, surrogate fitness, and
     * average cost for each generation.
     */
    virtual void output();

    /**
     * Checks whether the program can stop and if so, under what conditions:
     * 1. 1 = Successfully found solution within stopping criteria
     * 2. 0 = Number of generations exceeded
     * 3. -1 = Fatal error
     *
     * This function also updates the surrogate model if the program needs
     * to run for another generation.
     *
     * @return Stopping flag as int
     */
    virtual int stopCheck();

    /**
     * Computes the surrogate function that is updated at each GA iteration
     *
     * @note This function also updates the costs for roads based on the
     */
    virtual void computeSurrogate();

    /**
     * Stores the best road to the matrix of bests roads per test
     */
    virtual void assignBestRoad();

private:
    double scale;               /**< Cooling rate for mutation 3 */
    Eigen::VectorXd xO;         /**< X coordinate of plane origins */
    Eigen::VectorXd yO;         /**< Y coordinate of plane origins */
    Eigen::VectorXd zO;         /**< Z coordinate of plane origins */
    Eigen::VectorXd dU;         /**< Upper limits for plane domains */
    Eigen::VectorXd dL;         /**< Lower limits for plane domains */
    double theta;               /**< Cutting plane angle (to x axis) */
    // Values for current road population /////////////////////////////////////
    Eigen::VectorXd costs;      /**< Total costs of current population */
    Eigen::VectorXd profits;    /**< Unit operating profit per unit time for all roads */
    Eigen::MatrixXd iarsCurr;   /**< IARs of current population */
    Eigen::MatrixXd popsCurr;   /**< Full traffic flow end pops of current population */
    Eigen::VectorXd useCurr;    /**< Utility (ROV) of current population */
    // Values for progression of GA ///////////////////////////////////////////
    Eigen::VectorXd best;       /**< Best cost for each generation */
    Eigen::VectorXd av;         /**< Average cost for each generation */
    // Retained values for surrogates models //////////////////////////////////
    Eigen::MatrixXd iars;       /**< IARS for surrogate model (each column for each species) */
    Eigen::MatrixXd pops;       /**< Full traffic flow end pops for surrogate model */
    Eigen::VectorXd use;        /**< Utilities (ROV) for surrogate models */
    unsigned long noSamples;    /**< Current number of samples available for building surrogates */

// PRIVATE ROUTINES ///////////////////////////////////////////////////////////

    /**
     * Computes X and Y coordinates of design points for the GA that lie on the
     * perpendicular planes between the start and end points.
     *
     * @param individuals as const long&
     * @param intersectPts as const long&
     * @param startRow as const long&
     */
    void randomXYOnPlanes(const long& individuals, const long &intersectPts,
            const long &startRow);

    /**
     * Computes the X and Y coordinates of design points for the GA, allowing
     * them to lie randomly within the design region.
     *
     * @param individuals as const long&
     * @param intersectPts as const long&
     * @param startRow as const long&
     */
    void randomXYinRegion(const long& individuals, const long& intersectPts,
            const long& startRow);

    /**
     * Computes the Z coordinates of the design points as random elevations
     * within a prescribed permissible range.
     *
     * @param individuals as const long&
     * @param intersectPts as const long&
     * @param startRow as const long&
     * @param population as Eigen::MatrixXd&
     */
    void randomZWithinRange(const long &individuals, const long &intersectPts,
            const long &startRow, Eigen::MatrixXd &population);

    /**
     * Computes the Z coordinates of the design points as close as possible to
     * the actual terrain elevation but within the design grade requirements.
     *
     * @param individuals as const long&
     * @param intersectPts as const long&
     * @param startRow as const long&
     * @param population as Eigen::MatrixXd&
     */
    void zOnTerrain(const long &individuals, const long &intersectPts,
            const long &startRow, Eigen::MatrixXd &population);

    /**
     * Replaces roads with Inf or NaN elements with copies of valid ones.
     *
     * @param roads as Eigen::MatrixXd&
     * @param costs as Eigen::VectorXd&
     */
    void replaceInvalidRoads(Eigen::MatrixXd& roads, Eigen::VectorXd& costs);

    /**
     * Straightens the path between (jj and kk) and (kk and ll)
     *
     * @param ii as int
     * @param jj as int
     * @param kk as int
     * @param ll as int
     * @param children as Eigen::MatrixXd&
     */
    void curveEliminationProcedure(int ii, int jj, int kk, int ll,
            Eigen::MatrixXd& children);
};

#endif
