#ifndef OPTIMISER_H
#define OPTIMISER_H

class ExperimentalScenario;
typedef std::shared_ptr<ExperimentalScenario> ExperimentalScenarioPtr;

class Road;
typedef std::shared_ptr<Road> RoadPtr;

class SpeciesRoadPatches;
typedef std::shared_ptr<SpeciesRoadPatches> SpeciesRoadPatchesPtr;

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

class ThreadManager;
typedef std::shared_ptr<ThreadManager> ThreadManagerPtr;

class Optimiser;
typedef std::shared_ptr<Optimiser> OptimiserPtr;

class MonteCarloROV;
typedef std::shared_ptr<MonteCarloROV> MonteCarloROVPtr;

typedef std::shared_ptr<Gnuplot> GnuplotPtr;

/**
 * Class for managing the optimisation process
 */
class Optimiser : public std::enable_shared_from_this<Optimiser> {

public:
    // ENUMERATIONS ///////////////////////////////////////////////////////////
    typedef enum {
        NOPENALTY,      /**< No penalty at all */
        SIMPLEPENALTY,  /**< Penalty when building in certain areas */
        MTE,            /**< Set a minimum population to maintain per species */
        CONTROLLED      /**< Controlled animal population */
    } Type;

    typedef enum {
        ALGO1, /**< Regression Monte Carlo (Tsitsiklis and Van Roy) */
        ALGO2, /**< Regression Monte Carlo (Longstaff and Schwartz) */
        ALGO3, /**< Parametric Control (Guyon and Henry-Labordere) */
        ALGO4, /**< Regression Monte Carlo, State & Control (Kharroubi et al.) */
        ALGO5, /**< Regression Monte Carlo, State, Control & Recomputation */
        ALGO6, /**< Regression Monte Carlo, State, Control, Recomputation & Switching (Langrene et al.) */
        ALGO7  /**< Regression Monte Carlo with targeted end population (Zhang et al.) */
    } ROVType;

    typedef enum {
        COMPUTATION_FAILED,
        COMPUTATION_SUCCESS,
    } ComputationStatus;

    // CONSTRUCTORS AND DESTRUCTORS ///////////////////////////////////////////

    /**
     * Constructor I
     *
     * Constructs an %Optimiser object with default values.
     */
    Optimiser();

    /**
     * Constructor II
     *
     * Constructs an %Optimiser object with assigned values
     *
     * @param programs as const std::vector<TrafficProgramPtr>&
     * @param oInputs as OtherInputsPtr
     * @param desParams as DesignParametersPtr
     * @param earthworks as EarthworkCostsPtr
     * @param unitCosts as UnitCostsPtr
     * @param varParams as VariableParametersPtr
     * @param species as const std::vector<SpeciesPtr>&
     * @param economic as EconomicPtr
     * @param traffic as TrafficPtr
     * @param region as RegionPtr
     * @param mr as double
     * @param cf as unsigned long
     * @param gens as unsigned long
     * @param sg as unsigned long
     * @param popSize as unsigned long
     * @param stopTol as double
     * @param confInt as double
     * @param confLvl as double
     * @param habGridRes as unsigned long
     * @param solScheme as std::string
     * @param noRuns as unsigned long
     * @param type as Optimiser::Type
     * @param threader as ThreadManagerPtr
     * @param elite as double
     * @param msr as double
     * @param gpu as bool (default = false)
     * @param method as Optimiser:: (default = Optimiser::ALGO5)
     */
    Optimiser(double mr, double cf, unsigned long gens, unsigned long
            popSize, double stopTol, double confInt, double confLvl, unsigned
            long habGridRes, unsigned long surrDimRes, std::string solScheme,
            unsigned long noRuns, Optimiser::Type type, unsigned long sg,
            double msr, bool gpu=false, Optimiser::ROVType method =
            Optimiser::ALGO5);
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
        this->scenario.reset();
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
     * All roads have the same start and end points (in three dimensions) but
     * the optimisation algorithm only modifies the intervening points.
     *
     * @return Roads as const Eigen::MatrixXd&
     */
    const Eigen::MatrixXd& getCurrentRoads() {
        return this->currentRoadPopulation;
    }
    /**
     * Sets the current GA population of roads
     *
     * @param roads as const Eigen::MatrixXd&
     */
    void setCurrentRoads(const Eigen::MatrixXd& roads) {
        this->currentRoadPopulation = roads;
    }

    /**
     * Returns the best roads computed by the optimiser.
     *
     * @return Roads as const std::vector< std::vector<RoadPtr> >&
     */
    const std::vector< std::vector<RoadPtr> >& getBestRoads() {
        return this->bestRoads;
    }
    /**
     * Sets the best roads computed by the optimiser
     *
     * @param roads as const std::vector< std::vector<RoadPtr> >&
     */
    void setBestRoads(const std::vector< std::vector<RoadPtr> >& roads) {
        this->bestRoads = roads;
    }

    /**
     * Returns the different switching programs
     *
     * @return Program as const std::vector<TrafficProgramPtr>&
     */
    const std::vector<TrafficProgramPtr>& getPrograms() {
        return this->programs;
    }
    /**
     * Sets the different switching programs
     *
     * @param programs as const std::vector<TrafficProgramPtr>&
     */
    void setPrograms(const std::vector<TrafficProgramPtr>& program) {
        this->programs = program;
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
     * Returns the current generation in the optimisation process
     *
     * @return Current generation as double
     */
    double getGeneration() {
        return this->generation;
    }
    /**
     * Sets the current generation in the optimisation process
     *
     * @param gen as double
     */
    void setGeneration(double gen) {
        this->generation = gen;
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
     * @return Species as const std::vector<SpeciesPtr>&
     */
    const std::vector<SpeciesPtr>& getSpecies() {
        return this->species;
    }
    /**
     * Sets the species details
     *
     * @param species as const std::vector<SpeciesPtr>&
     */
    void setSpecies(const std::vector<SpeciesPtr>& species) {
        this->species = species;
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
        this->economic.reset();
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
        this->traffic.reset();
        this->traffic = traffic;
    }

//    /**
//     * Returns the TrafficProgram over which the optimisation is run.
//     *
//     * @return TrafficProgram as TrafficProgramPtr
//     */
//    TrafficProgramPtr getTrafficProgram() {
//        return this->trafficProgram;
//    }
//    /**
//     * Sets the TrafficProgram over which the optimisation is run.
//     *
//     * @param tp as TrafficProgramPtr
//     */
//    void setTrafficProgram(TrafficProgramPtr tp) {
//        this->trafficProgram.reset();
//        this->trafficProgram = tp;
//    }

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
        this->region.reset();
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
     * Returns the max number of generations to continue without improvement
     *
     * @return Number of stall generations permitted as unsigned long
     */
    unsigned long getStallGens() {
        return this->stallGenerations;
    }
    /**
     * Sets the max number of generations to continue without improvement
     *
     * @param sg as unsigned long
     */
    void setStallGens(unsigned long sg) {
        this->stallGenerations = sg;
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
     * Returns the surrogate resolution in each dimension
     *
     * @return Resolution as unsigned long
     */
    unsigned long getSurrDimRes() {
        return this->surrDimRes;
    }
    /**
     * Sets the surrogate resolution in each dimension
     *
     * @param res as unsigned long
     */
    void setSurrDimRes(unsigned long res) {
        this->surrDimRes = res;
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

    /**
     * Returns the Thread Manager
     *
     * @return ThreadManager as ThreadManagerPtr
     */
    ThreadManagerPtr getThreadManager() {
        return this->threader;
    }
    /**
     * Sets the ThreadManager
     *
     * @param threader as ThreadManagerPtr
     */
    void setThreadManager(ThreadManagerPtr threader) {
        this->threader.reset();
        this->threader = threader;
    }

    /**
     * Returns the proportion of the population to retain as elite
     *
     * @return Elite individuals proportion as double
     */
    double getEliteIndividuals() {
        return this->eliteIndividuals;
    }
    /**
     * Sets the proportion of the population to retain as elite
     *
     * @param e as double
     */
    void setEliteIndividuals(double e) {
        this->eliteIndividuals = e;
    }

    /**
     * Returns the maximum rate at which to extract samples from a generation
     *
     * Returns the maximum rate at which to extract samples from a generation
     * so as to determine the surrogate model for improving the performance of
     * the GA.
     *
     * @return Max sample rate as double
     */
    double getMaxSampleRate() {
        return this->maxSampleRate;
    }
    /**
     * Sets the sampling rate for building surrogate models
     *
     * @param msr as double
     */
    void setMaxSampleRate(double msr) {
        this->maxSampleRate = msr;
    }

    /**
     * Returns the handle to the output interface for Gnuplot (standard)
     *
     * @return Handle as GnuplotPtr
     */
    GnuplotPtr getPlotHandle() {
        return this->plothandle;
    }
    /**
     * Returns the handle to the output interface for Gnuplot (surrogates)
     *
     * @return Handle as GnuplotPtr
     */
    GnuplotPtr getSurrPlotHandle() {
        return this->surrPlotHandle;
    }

    /**
     * Returns a Boolean of whether we are using a GPU or not
     *
     * @return Whether we are using a GPU as bool
     */
    bool getGPU() {
        return this->gpu;
    }
    /**
     * Sets a Boolean of whether we are using a GPU or not
     *
     * @param gpu as bool
     */
    void setGPU(bool gpu) {
        this->gpu = gpu;
    }

    /**
     * Returns the ROV method used in the optimisation
     *
     * @return Method as Optimiser::ROVType
     */
    Optimiser::ROVType getROVMethod() {
        return this->method;
    }
    /**
     * Sets the ROV method used in the optimisation
     *
     * @param method as Optimiser::ROVType
     */
    void setROVMethod(Optimiser::ROVType method) {
        this->method = method;
    }

    // STATIC ROUTINES ////////////////////////////////////////////////////////

    // CALCULATION ROUTINES ///////////////////////////////////////////////////

    /**
     * Initialises the storage containers in the object (bestRoads etc.)
     *
     * @note This function must be called once all object attributes such as
     * VarParams have been assigned non-null pointers
     */
    virtual void initialiseStorage();

    /**
     * Runs the algorithm to optimise the road with the possibility to enter
     * the animal habitat zones. Results are saved to output files. If the
     * argument "true" is passed to the function, the following plots are
     * generated as the optimisation proceeds. Otherwise, no plots are
     * generated:
     * 1. 3D plot of best road on terrain
     * 2. 2D plot of best road with respect to habitat
     * 3. Surrogate function
     * 4. Best results history
     *
     * This routine contains code to simply plot the result. The actual
     * optimisation procedures are contained in derived classes.
     *
     * @param plot as bool
     */
    virtual void optimise(bool plot = false);

    /**
     * Computes the habitat map for every species
     */
    void computeHabitatMaps();

    /**
     * Computes the expected present value of a constant unit of use of every
     * uncertain parameter over the design horizon.
     */
    void computeExpPv();

    /**
     * Returns a reference to the surrogate function so that it may be used
     *
     * @return Surrogate function as std::vector<std::vector<std::vector<alglib::spline1dinterpolant>>>&
     */
    std::vector<std::vector<std::vector<alglib::spline1dinterpolant>>>& getSurrogate() {
        return this->surrogate;
    }
    /**
     * Sets the surrogate function so that it may be called at a later stage
     *
     * The surrogate function accepts a pointer to a road object to compute its
     * operating cost (based on a learned function), which is returned as a
     * double.
     *
     * @param surrogate as std::vector<std::vector<std::vector<alglib::spline1dinterpolant>>>&
     */
    void setSurrogate(std::vector<std::vector<std::vector<
            alglib::spline1dinterpolant>>>& surrogate) {
        this->surrogate = surrogate;
    }

    /**
     * Returns a reference to the surrogate functions so that they may be used
     *
     * The surrogate model is stored as a fixed-resolution map of predictor
     * points (res*noDims) with corresponding regressed values (res^noDims)
     *
     * The first res x noDims values are the fixed predictor points that can be
     * interpolated. They are sorted by each dimension so that all the 'res'
     * values along each dimension are listed first followed by the 'res'
     * values for the next dimension, and so on. The next 'res*noDims' values
     * are the corresponding predicted values. The way to index is as follows:
     *
     * 1. Value of predictor ii in dimension jj
     *
     *    double value = this->surrogateROV[jj*res + ii]
     *
     * 2. Value of corresponding regressed value for coordinate I[jj_0, jj_1,
     *    ..., jj_(noDims-1)]
     *
     *    int idx = res*noDims;
     *
     *    for (int ii = 0; ii < noDims; ii++) {
     *       idx += I[ii]*res^(noDims - ii - 1);
     *    }
     *
     *    double value = this->surrogateROV[idx];
     *
     * @return Surrogates as std::vector<std::vector<Eigen::VectorXd>>&
     */
    std::vector<std::vector<Eigen::VectorXd>>& getSurrogateROV() {
        return this->surrogateROV;
    }
    /**
     * Sets the surrogate function so that it may be called at a later stage
     *
     * @param surr as std::vector<std::vector<std::vector<double>>>&
     */
    void setSurrogateROV(std::vector<std::vector<Eigen::VectorXd>>& surr) {
        this->surrogateROV = surr;
    }

    /**
     * Evaluates the surrogate model for a given road in the fixed traffic flow
     * case.
     *
     * For each species, a cubic spline of the BSpline class is evaluated
     *
     * @param (input) road as RoadPtr
     * @param (output) pops and as Eigen::VectorXd&
     * @param (output) popsSD as Eigen::VectorXd&
     */
    void evaluateSurrogateModelMTE(RoadPtr road, Eigen::VectorXd &pops,
            Eigen::VectorXd &popsSD);
    /**
     * Evaluates the surrogate model for a given road in the fixed traffic flow
     * case.
     *
     * @param (input) road as RoadPtr
     * @param (output) value as double&
     * @param (output) valueSD as double&
     * @param (output) use as double&
     * @param (output) use standard deviation as double&
     */
    void evaluateSurrogateModelROVCR(RoadPtr road, double value, double
            valueSD);

///////////////////////////////////////////////////////////////////////////////
protected:
    std::vector<std::vector<std::vector<alglib::spline1dinterpolant>>> surrogate;   /**< MTE Surrogate model for evaluating road (for each run) stored as a collection of splines (To be deprecated) */
    std::vector<std::vector<Eigen::VectorXd>> surrogateROV;                         /**< ROV Surrogate model */
    ExperimentalScenarioPtr scenario;                                               /**< Current experiment */
    Optimiser::Type type;                                                           /**< Type of ecological incorporation */
    Eigen::MatrixXd currentRoadPopulation;                                          /**< Current encoded population of roads */
    std::vector< std::vector<RoadPtr> > bestRoads;                                  /**< Best roads */
    std::vector<TrafficProgramPtr> programs;                                        /**< Operational programs */
    OtherInputsPtr otherInputs;                                                     /**< Other inputs */
    DesignParametersPtr designParams;                                               /**< Design parameters */
    EarthworkCostsPtr earthworks;                                                   /**< Earthwork requirements */
    VariableParametersPtr variableParams;                                           /**< Parameters to vary */
    std::vector<SpeciesPtr> species;                                                /**< Species studied */
    EconomicPtr economic;                                                           /**< Economic parameters */
    TrafficPtr traffic;                                                             /**< Traffic details */
    //TrafficProgramPtr trafficProgram;                                             /**< Traffic program used */
    RegionPtr region;                                                               /**< Region of interest */
    UnitCostsPtr unitCosts;                                                         /**< Unit Costs */
    unsigned long generation;                                                       /**< Current generation in optimisation process */
    unsigned long stallGen;                                                         /**< Number of sequential stall generations so far */
    double mutationRate;                                                            /**< Mutation rate */
    double crossoverFrac;                                                           /**< Crossover fraction */
    unsigned long generations;                                                      /**< Generations required */
    unsigned long stallGenerations;                                                 /**< Maximum number of stall generations at which to stop the algorithm */
    unsigned long populationSizeGA;                                                 /**< Population size for GA */
    double stoppingTol;                                                             /**< Stopping tolerance */
    double confInt;                                                                 /**< Required confidence interval */
    double confLvl;                                                                 /**< Desired confidence level (default = 95%) */
    unsigned long habGridRes;                                                       /**< Habitat grid 1D resolution */
    unsigned long surrDimRes;                                                       /**< Resolution along each dimension of the surrogate */
    unsigned long noRuns;                                                           /**< Number of runs to perform */
    double eliteIndividuals;                                                        /**< Proportion of elite individuals to retain each generation */
    double maxSampleRate;                                                           /**< Maximum rate at which to perform sampling for surrogate models */
    std::string solutionScheme;                                                     /**< Solution scheme used (i.e. name of experiment) */
    ThreadManagerPtr threader;                                                      /**< Thread manager used for multithreading computations */
    OptimiserPtr me();                                                              /**< Creates a shared pointer from this */
    GnuplotPtr plothandle;                                                          /**< Pipe for plotting results */
    GnuplotPtr surrPlotHandle;                                                      /**< Pipe for plotting surrogate model results */
    bool gpu;                                                                       /**< If we are using GPUs to assist computing */
    Optimiser::ROVType method;                                                      /**< ROV algorithm */

    // Sharing of derived from base
    template <typename Derived>
    std::shared_ptr<Derived> meDerived()
    {
        return std::static_pointer_cast<Derived>(shared_from_this());
    }
};

#endif
