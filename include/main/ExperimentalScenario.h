#ifndef EXPERIMENTALSCENARIO_H
#define EXPERIMENTALSCENARIO_H

class Optimiser;
typedef std::shared_ptr<Optimiser> OptimiserPtr;

class ExperimentalScenario;
typedef std::shared_ptr<ExperimentalScenario> ExperimentalScenarioPtr;

class ExperimentalScenario :
        public std::enable_shared_from_this<ExperimentalScenario> {

public:
    // CONSTRUCTORS AND DESTRUCTORS

    /**
     * Constructor I
     *
     * Constructs an %ExperimentalScenario object with default values
     */
    ExperimentalScenario(OptimiserPtr optimiser);
    /**
     * Constructor II
     *
     * Constructs an %ExperimentalScenario object with assigned values
     */
    ExperimentalScenario(OptimiserPtr optimiser, int program, int popLevel,
            int habPrefSD, int lambdaSD, int rangingCoeffSD, int animalBridge,
            int popGR, int fuel, int commodity);
    /**
     * Destructor
     */
    ~ExperimentalScenario();

    // ACCESSORS///////////////////////////////////////////////////////////////

    /**
     * Returns the Optimiser
     *
     * @return Optimiser as OptimiserPtr
     */
    OptimiserPtr getOptimiser() {
        return this->optimiser;
    }
    /**
     * Sets the Optimiser
     *
     * @param optimiser as OptimiserPtr
     */
    void setOptimiser(OptimiserPtr optimiser) {
        this->optimiser.reset();
        this->optimiser = optimiser;
    }

    /**
     * Returns the program index
     *
     * @return Program index as int
     */
    int getProgram() {
        return this->program;
    }
    /**
     * Sets the program index
     *
     * @param program as int
     */
    void setProgram(int program) {
        this->program = program;
    }

    /**
     * Returns the population level index
     *
     * @return Population level index as int
     */
    int getPopLevel() {
        return this->popLevel;
    }
    /**
     * Sets the population level index
     *
     * @param popLevel as int
     */
    void setPopLevel(int popLevel) {
        this->popLevel = popLevel;
    }

    /**
     * Returns the habitat preference index
     *
     * @return HabPref index as int
     */
    int getHabPref() {
        return this->habPref;
    }
    /**
     * Sets the habitat preference index
     *
     * @param habPref as int
     */
    void setHabPref(int habPref) {
        this->habPref = habPref;
    }

    /**
     * Returns the lambda index as int
     *
     * @return Lamnda index as int
     */
    int getLambda() {
        return this->lambda;
    }
    /**
     * Sets the lambda index as int
     *
     * @param lambda as int
     */
    void setLambdaS(int lambda) {
        this->lambda = lambda;
    }

    /**
     * Returns the ranging coefficient index as int
     *
     * @return Ranging coefficient index as int
     */
    int getRangingCoeff() {
        return this->rangingCoeff;
    }
    /**
     * Sets the ranging coefficient index as int
     *
     * @param ranging as int
     */
    void setRangingCoeff(int ranging) {
        this->rangingCoeff = ranging;
    }

    /**
     * Returns the animal bridge index as int
     *
     * @return Animal bridge index as int
     */
    int getAnimalBridge() {
        return this->animalBridge;
    }
    /**
     * Sets the animal bridge index as int
     *
     * @param bridge as int
     */
    void setAnimalBridge(int bridge) {
        this->animalBridge = bridge;
    }

    /**
     * Returns the population growth rate SD multiplier index
     *
     * @return Population growth rate SD multiplier index as int
     */
    int getPopGR () {
        return this->popGR;
    }
    /**
     * Sets the population growth rate SD multiplier index
     *
     * @param popGR as int
     */
    void setPopGR(int popGR) {
        this->popGR = popGR;
    }

    /**
     * Returns the fuel price SD multiplier index as int
     *
     * @return Fuel price SD multiplier index as int
     */
    int getFuel() {
        return this->fuel;
    }
    /**
     * Sets the fuel price SD multiplier index as int
     *
     * @param fuel as int
     */
    void setFuel(int fuel) {
        this->fuel = fuel;
    }

    /**
     * Returns the commodity price SD multiplier index as int
     *
     * @return Commodity price SD multiplier index as int
     */
    int getCommodity() {
        return this->commodity;
    }
    /**
     * Sets the commodity price SD multiplier index as int
     *
     * @param commodity as int
     */
    void setCommodity(int commodity) {
        this->commodity = commodity;
    }

    // STATIC ROUTINES ////////////////////////////////////////////////////////

    // CALCULATION ROUTINES ///////////////////////////////////////////////////

///////////////////////////////////////////////////////////////////////////////
private:
    OptimiserPtr optimiser;     /**< Calling Optimiser */
    int program;                /**< Index of Program used */
    int popLevel;               /**< Index of population level used */
    int habPref;                /**< Index of habitat preference used */
    int lambda;                 /**< Index of lambda used */
    int rangingCoeff;           /**< Index of ranging coefficient used */
    int animalBridge;           /**< Index of animal bridge test used */
    int popGR;                  /**< Index of population growth uncertainty used (0 = no uncertainty) */
    int fuel;                   /**< Index of fuel price uncertainty multiplier used (0 = no uncertainty) */
    int commodity;              /**< Index of commodity price uncertainty multiplier used (0 = no uncertainty) */
};

#endif
