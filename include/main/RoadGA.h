#ifndef ROADGA_H
#define ROADGA_H

class Optimiser;
typedef std::shared_ptr<Optimiser> OptimiserPtr;

class RoadGA;
typedef std::shared_ptr<RoadGA> RoadGAPtr;

/**
 * Class for managing the optimal road design process using Genetic Algorithms
 *
 * @note Based on the road design procedure of Jong and Schonfeld (2003)
 */
class RoadGA : public Optimiser,
        public std::enable_shared_from_this<RoadGA> {

public:
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
            std::string solScheme, unsigned long noRuns, Optimiser::Type type);

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
     * This function overrides any base class function of the same name.
     */
    virtual void crossover();

    /**
     * Mutates population used in the optimisation process
     *
     * This function overrides any base class function of the same name.
     */
    virtual void mutation();

    /**
     * Runs the optimisation algorithm to devise the best Road
     */
    virtual void optimise();

    /**
     * Returns the outputs at specific stages of the GA
     */
    virtual void output();

    /**
     * Computes the surrogate function that is updated at each GA iteration
     */
    virtual void computeSurrogate();

private:
    Eigen::VectorXd xO;     /**< X coordinate of plane origins */
    Eigen::VectorXd yO;     /**< Y coordinate of plane origins */
    Eigen::VectorXd zO;     /**< Z coordinate of plane origins */
    Eigen::VectorXd dU;     /**< Upper limits for plane domains */
    Eigen::VectorXd dL;     /**< Lower limits for plane domains */
    double theta;           /**< Cutting plane angle (to x axis) */

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
     */
    void randomZWithinRange(const long &individuals, const long &intersectPts,
            const long &startRow);

    /**
     * Computes the Z coordinates of the design points as close as possible to
     * the actual terrain elevation but within the design grade requirements.
     *
     * @param individuals as const long&
     * @param intersectPts as const long&
     * @param startRow as const long&
     */
    void zOnTerrain(const long &individuals, const long &intersectPts,
            const long &startRow);
};

#endif
