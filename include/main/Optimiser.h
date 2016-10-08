#ifndef OPTIMISER_H
#define OPTIMISER_H

class ExperimentalScenario;
typedef std::shared_ptr<ExperimentalScenario> ExperimentalScenarioPtr;

class Road;
typedef std::shared_ptr<Road> RoadPtr;

class TrafficProgram;
typedef std::shared_ptr<TrafficProgram> TrafficProgramPtr;

class DesignParameters;
typedef std::shared_ptr<DesignParameters> DesignParametersPtr;

class Vehicle;
typedef std::shared_ptr<Vehicle> VehiclePtr;

class EarthworkCosts;
typedef std::shared_ptr<EarthworkCosts> EarthworkCostsPtr;

class UnitCosts;
typedef std::shared_ptr<UnitCosts> UnitCostsPtr;

class Species;
typedef std::shared_ptr<Species> SpeciesPtr;

class UnitCosts;
typedef std::shared_ptr<UnitCosts> UnitCostSPtr;

class OtherInputs;
typedef std::shared_ptr<OtherInputs> OtherInputsPtr;

class VariableParameters;
typedef std::shared_ptr<VariableParameters> VariableParametersPtr;

class Economic;
typedef std::shared_ptr<Economic> EconomicPtr;

class Traffic;
typedef std::shared_ptr<Traffic> TrafficPtr;

class Region;
typedef std::shared_ptr<Region> RegionPtr;

class Optimiser;
typedef std::shared_ptr<Optimiser> OptimiserPtr;

/**
 * Class for managing the optimisation process
 */
class Optimiser : public std::enable_shared_from_this<Optimiser> {

public:
    // ENUMERATIONS ///////////////////////////////////////////////////////////
    typedef enum {
        SIMPLEPENALTY,  /**< Penalty when building in certain areas */
        MTE,            /**< Set a minimum population to maintain per species */
        CONTROLLED      /**< Controlled animal population */
    } Type;

    // CONSTRUCTORS AND DESTRUCTORS ///////////////////////////////////////////

    /**
     * Constructor
     *
     * Constructs an %Optimiser object with default values.
     */
    Optimiser(std::vector<TrafficProgramPtr>* programs, OtherInputsPtr oInputs,
            DesignParametersPtr desParams, EarthworkCostsPtr earthworks,
            UnitCostsPtr unitCosts, VariableParametersPtr varParams,
            std::vector<SpeciesPtr>* species, EconomicPtr economic, TrafficPtr
            traffic, RegionPtr region, double mr, unsigned long cf, unsigned
            long gens, unsigned long popSize, double stopTol, double confInt,
            double confLvl, unsigned long habGridRes, std::string solScheme,
            unsigned long noRuns, Optimiser::Type type);
    /**
     * Destructor
     */
    ~Optimiser();

    // ACCESSORS //////////////////////////////////////////////////////////////

    /**
     * Returns the current ExperimentalScenario
     *
     * @return ExperimentalScenario as ExperimentalScenarioPtr
     * @note Currently this is part of the Optimiser class so we cannot create
     * parallel experiments. If we want to do this, copies of the Optimiser
     * will have to be constructed and passed to different threads where the
     * results are saved to the shared matrix of results. When each experiment
     * is complete, the forked Optimiser can be deleted.
     */
    ExperimentalScenarioPtr getScenario() {
        return this->scenario;
    }
    /**
     * Sets the current ExperimentalScenario
     *
     * @param scenario as ExperimentalScenarioPtr
     */
    void setScenario(ExperimentalScenarioPtr scenario) {
        this->scenario = scenario;
    }

    /**
     * Returns the type of optimisation process used
     *
     * @return Ecological attribution method as Optimiser::Type
     */
    Optimiser::Type getType() {
        return this->type;
    }
    /**
     * Sets the type of optimisation process used
     *
     * @param type as Optimiser::Type
     */
    void setType(Optimiser::Type type) {
        this->type = type;
    }

    /**
     * Returns the current GA population of roads. These roads are only defined
     * by their intersection points.
     *
     * @return Roads as (Eigen::MatrixXd)*
     */
    Eigen::MatrixXd* getCurrentRoads() {
	    return &this->currentRoadPopulation;
    }
    /**
     * Sets the current GA population of roads
     *
     * @param roads as (Eigen::MatrixXd)*
     */
    void setCurrentRoads(Eigen::MatrixXd* roads) {
	    this->currentRoadPopulation = *roads;
    }

    /**
     * Returns the best roads computed by the optimiser.
     *
     * @return Roads as std::vector< std::vector<RoadPtr> >*
     */
    std::vector< std::vector<RoadPtr> >* getBestRoads() {
	    return &this->bestRoads;
    }
    /**
     * Sets the best roads computed by the optimiser
     *
     * @param roads as std::vector< std::vector<RoadPtr> >*
     */
    void setBestRoads(std::vector< std::vector<RoadPtr> >* roads) {
	    this->bestRoads = *roads;
    }

    /**
     * Returns the different switching programs
     *
     * @return Program as std::vector<TrafficProgramPtr>
     */
    std::vector<TrafficProgramPtr>* getPrograms() {
	    return &this->programs;
    }
    /**
     * Sets the different switching programs
     *
     * @param programs as std::vector<TrafficProgramPtr>
     */
    void setPrograms(std::vector<TrafficProgramPtr>* program) {
	    this->programs = *program;
    }

    /**
     * Returns the miscellaneous inputs needed by the software
     *
     * @return Inputs as OtherInputsPtr
     */
    OtherInputsPtr getOtherInputs() {
	    return this->otherInputs;
    }
    /**
     * Sets the miscellaneous inputs needed by the software
     *
     * @param inputs as OtherInputsPtr
     */
    void setOtherInputs(OtherInputsPtr inputs) {
	    this->otherInputs.reset();
	    this->otherInputs = inputs;
    }

    /**
     * Returns the road design parameters
     *
     * @return Road design parameters as DesignParamsPtr
     */
    DesignParametersPtr getDesignParameters() {
	    return this->designParams;
    }
    /**
     * Sets the road design parameters
     *
     * @param params as DesignParamsPtr
     */
    void setDesignParams(DesignParametersPtr params) {
	    this->designParams.reset();
	    this->designParams = params;
    }

    /**
     * Returns the earthwork requirements used
     *
     * @return Earthworks as EarthworkCostsPtr
     */
    EarthworkCostsPtr getEarthworkCosts() {
	    return this->earthworks;
    }
    /**
     * Sets the earthwork requirements used
     *
     * @param earthworks as EarthworkCostsPtr
     */
    void setEarthworkCosts(EarthworkCostsPtr earthworks) {
	    this->earthworks.reset();
	    this->earthworks = earthworks;
    }

    /**
     * Returns the unit costs
     *
     * @return Unit costs as UnitCostsPtr
     */
    UnitCostsPtr getUnitCosts() {
	    return this->unitCosts;
    }
    /**
     * Sets the unit costs
     *
     * @param costs as UnitCostsPtr
     */
    void setUnitCosts(UnitCostsPtr costs) {
	    this->unitCosts.reset();
	    this->unitCosts = costs;
    }

    /**
     * Returns the variable parameters
     *
     * @return Variable parameters as VariableParametersPtr
     */
    VariableParametersPtr getVariableParams() {
	    return this->variableParams;
    }
    /**
     * Sets the variable parameters
     *
     * @param varParams as VariableParametersPtr
     */
    void setVariableParams(VariableParametersPtr varParams) {
	    this->variableParams.reset();
	    this->variableParams = varParams;
    }

    /**
     * Returns the species details
     *
     * @return Species as std::vector<SpeciesPtr>*
     */
    std::vector<SpeciesPtr>* getSpecies() {
	    return &this->species;
    }
    /**
     * Sets the species details
     *
     * @param species as std::vector<SpeciesPtr>*
     */
    void setSpecies(std::vector<SpeciesPtr>* species) {
	    this->species = *species;
    }

    /**
     * Returns the economic data
     *
     * @return Economic data as EconomicPtr
     */
    EconomicPtr getEconomic() {
	    return this->economic;
    }
    /**
     * Sets the economic data
     *
     * @param econ as EconomicPtr
     */
    void setEconomic(EconomicPtr econ) {
	    this->economic = econ;
    }

    /**
     * Returns the traffic data
     *
     * @return Traffic data as TrafficPtr
     */
    TrafficPtr getTraffic() {
	    return this->traffic;
    }
    /**
     * Sets the traffic data
     *
     * @param traffic as TrafficPtr
     */
    void setTraffic(TrafficPtr traffic) {
	    this->traffic = traffic;
    }

    /**
     * Returns the TrafficProgram over which the optimisation is run.
     *
     * @return TrafficProgram as TrafficProgramPtr
     */
    TrafficProgramPtr getTrafficProgram() {
        return this->trafficProgram;
    }
    /**
     * Sets the TrafficProgram over which the optimisation is run.
     *
     * @param tp as TrafficProgramPtr
     */
    void setTrafficProgram(TrafficProgramPtr tp) {
        this->trafficProgram = tp;
    }

    /**
     * Returns the region data
     *
     * @return Region data as RegionPtr
     */
    RegionPtr getRegion() {
	    return this->region;
    }
    /**
     * Sets the region data
     *
     * @param region as RegionPtr
     */
    void setRegion(RegionPtr region) {
	    this->region = region;
    }

    /**
     * Returns the GA mutation rate
     *
     * @return Mutation rate as double
     */
    double getMutationRate() {
	    return this->mutationRate;
    }
    /**
     * Sets the GA mutation rate
     *
     * @param rate as double
     */
    void setMutationRate(double rate) {
	    this->mutationRate = rate;
    }

    /**
     * Returns the number of optimisation runs to perform for population-based
     * optimisation.
     *
     * @return Number of runs as unsigned long
     */
    unsigned long getNoRuns() {
	    return this->noRuns;
    }
    /**
     * Sets the number of optimisation runs to perform.
     *
     * @param noRuns as unsigned long
     */
    void setNoRuns(unsigned long noRuns) {
	    this->noRuns = noRuns;
    }

    /**
     * Returns the GA crossover fraction
     *
     * @return Crossover fraction as double
     */
    double getCrossoverFrac() {
	    return this->crossoverFrac;
    }
    /**
     * Sets the GA crossover fraction
     *
     * @param frac as double
     */
    void setCrossoverFrac(double frac) {
	    this->crossoverFrac = frac;
    }

    /**
     * Returns the max number of GA generations
     *
     * @return Generations as unsigned long
     */
    unsigned long getMaxGens() {
	    return this->generations;
    }
    /**
     * Sets the max number of GA generations
     *
     * @params gens as unsigned long
     */
    void setMaxGens(unsigned long gens) {
	    this->generations = gens;
    }

    /**
     * Returns the GA population size
     *
     * @return Population size as unsigned long
     */
    unsigned long getGAPopSize() {
	    return this->populationSizeGA;
    }
    /**
     * Sets the GA population size
     *
     * @param size as unsigned long
     */
    void setPopSize(unsigned long size) {
	    this->populationSizeGA = size;
    }

    /**
     * Returns the optimiser stopping tolerance
     *
     * @return Stopping tolerance as double
     */
    double getStoppingTol() {
	    return this->stoppingTol;
    }
    /**
     * Sets the optimiser stopping tolerance
     *
     * @param tol as double
     */
    void setStoppingTol(double tol) {
	    this->stoppingTol = tol;
    }

    /**
     * Returns the confidence interval used for simulation results.
     *
     * This number (X) has different uses depending on the simulation model:
     * 1.   Constant full traffic flow - X is used to compute the end animal
     *      population that the road has a 95% chance of exceeding.
     * 2.   Stochastic dynamic programming - X is the probability that using
     *      control Y will result in the population exceeding the threshold.
     *
     * Alternately, X can be interpreted as the proportion of roads exceeding
     * the number we use to define the road (end population, profit, etc.)
     * when comparing it to other roads. That means that X is multi-purpose.
     * @return Confidence interval as double
     */
    double getConfidenceInterval() {
        return this->confInt;
    }
    /**
     * Sets the confidence interval used for simulation results.
     *
     * @param confidence as double
     */
    void setConfidence(double confidence) {
        this->confInt = confidence;
    }

    /**
     * Returns the confidence level used for simulations/sampling
     *
     * This is also used when computing required sample sizes.
     * @return Confidence level of results as double
     */
    double getConfidenceLevel() {
        return this->confLvl;
    }
    /**
     * Sets the confidence level used for simulations/sampling
     *
     * @param confidence as double
     */
    void setConfidenceLevel(double confidence) {
        this->confLvl = confidence;
    }

    /**
     * Returns the habitat grid resolution
     *
     * @return Grid resolution as unsigned long
     */
    unsigned long getGridRes() {
	    return this->habGridRes;
    }
    /**
     * Sets the habitat grid resolution
     *
     * @param res as double
     */
    void setGridRes(unsigned long res) {
	    this->habGridRes = res;
    }

    /**
     * Returns the solution scheme of the solver
     *
     * @return Solution scheme as std::string
     */
    std::string getSolutionScheme() {
	    return this->solutionScheme;
    }
    /**
     * Sets the solution scheme of the solver
     *
     * @param scheme as std::string
     */
    void setSolutionScheme(std::string scheme) {
	    this->solutionScheme = scheme;
    }

    // STATIC ROUTINES ////////////////////////////////////////////////////////

    // CALCULATION ROUTINES ///////////////////////////////////////////////////

    /**
     * Runs the algorithm to optimise the road with the possibility to enter
     * the animal habitat zones. Results are saved to output files.
     */
    void optimiseRoad();

    /**
     * Computes the habitat map for every species
     */
    void computeHabitatMaps();

///////////////////////////////////////////////////////////////////////////////
private:
    ExperimentalScenarioPtr scenario;               /**< Current experiment */
    Optimiser::Type type;                           /**< Type of ecological incorporation */
    Eigen::MatrixXd currentRoadPopulation;			/**< Current encoded population of roads */
    std::vector< std::vector<RoadPtr> > bestRoads;	/**< Best roads */
    std::vector<TrafficProgramPtr> programs;		/**< Operational programs */
    OtherInputsPtr otherInputs;						/**< Other inputs */
    DesignParametersPtr designParams;				/**< Design parameters */
    EarthworkCostsPtr earthworks;					/**< Earthwork requirements */
    VariableParametersPtr variableParams;			/**< Parameters to vary */
    std::vector<SpeciesPtr> species;				/**< Species studied */
    EconomicPtr economic;							/**< Economic parameters */
    TrafficPtr traffic;								/**< Traffic details */
    TrafficProgramPtr trafficProgram;               /**< Traffic program used */
    RegionPtr region;								/**< Region of interest */
    UnitCostsPtr unitCosts;                         /**< Unit Costs */
    double mutationRate;							/**< Mutation rate */
    double crossoverFrac;							/**< Crossover fraction */
    unsigned long generations;						/**< Generations required */
    unsigned long populationSizeGA;					/**< Population size for GA */
    double stoppingTol;								/**< Stopping tolerance */
    double confInt; 								/**< Required confidence interval */
    double confLvl;                                 /**< Desired confidence level (default = 95%) */
    unsigned long habGridRes;						/**< Habitat grid 1D resolution */
    unsigned long noRuns;							/**< Number of runs to perform */
    std::string solutionScheme;						/**< Solution scheme used (i.e. name of experiment) */
    OptimiserPtr me();                              /**< Creates a shared pointer from this */
};

#endif
