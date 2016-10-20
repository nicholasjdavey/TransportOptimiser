#ifndef ROADGA_H
#define ROADGA_H

class Optimiser;
typedef std::shared_ptr<Optimiser> OptimiserPtr;

class RoadGA;
typedef std::shared_ptr<RoadGA> RoadGAPtr;

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

    /**

    // ACCESSORS //////////////////////////////////////////////////////////////

    // STATIC ROUTINES ////////////////////////////////////////////////////////

    // CALCULATION ROUTINES ///////////////////////////////////////////////////

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
};

#endif
