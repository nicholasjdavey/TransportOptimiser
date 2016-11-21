#ifndef SPECIES_H
#define SPECIES_H

class Optimiser;
typedef std::shared_ptr<Optimiser> OptimiserPtr;

class Region;
typedef std::shared_ptr<Region> RegionPtr;

class Road;
typedef std::shared_ptr<Road> RoadPtr;

class SpeciesRoadPatches;
typedef std::shared_ptr<SpeciesRoadPatches> SpeciesRoadPatchesPtr;

class Uncertainty;
typedef std::shared_ptr<Uncertainty> UncertaintyPtr;

class Species;
typedef std::shared_ptr<Species> SpeciesPtr;

class HabitatType;
typedef std::shared_ptr<HabitatType> HabitatTypePtr;

class HabitatPatch;
typedef std::shared_ptr<HabitatPatch> HabitatPatchPtr;

/**
 * Class for managing %Species objects
 */
class Species : public std::enable_shared_from_this<Species> {

public:
    // CONSTRUCTORS AND DESTRUCTORS ////////////////////////////////////////////

    /**
     * Constructor I
     *
     * Constructs a %Species object with default parent values.
     */
    Species(std::string nm, bool sex, double lm, double lsd, double rcm,
            double rcsd, double grm, double grsd, double lenm, double lensd,
            double spm, double spsd, double cpa, bool active,
            std::vector<HabitatTypePtr>& habitat);

    /**
     * Constructor II
     *
     * Constructs a %Species object with assigned values.
     */
    Species(std::string nm, bool sex, double lm, double lsd, double rcm,
            double rcsd, double grm, double grsd, double lenm, double lensd,
            double spm, double spsd, double cpa, bool active,
            std::vector<HabitatTypePtr> &habitat, double current);

    /**
     * Destructor
     */
    ~Species();

    // ACCESSORS ///////////////////////////////////////////////////////////////

    /**
     * Returns the Species name
     *
     * @return Name as std::string
     */
    std::string getName() {
        return this->name;
    }
    /**
     * Sets the Species name
     *
     * @param nm as std::string
     */
    void setName(std::string nm) {
        this->name = nm;
    }

    /**
     * Returns whether the species is considered in the design process
     *
     * @return Active as bool
     */
    bool getActive() {
        return this->active;
    }
    /**
     * Sets whether the species is considered in the design process
     *
     * @param ac as bool
     */
    void setActive(bool ac) {
        this->active = ac;
    }

    /**
     * Returns the sex
     *
     * @return Sex as bool (1=female, 0=male)
     */
    bool getSex() {
        return this->sex;
    }
    /**
     * Sets the sex
     *
     * @param sex as bool (1=female, 0=male)
     */
    void setSex(bool sex) {
        this->sex = sex;
    }

    /**
     * Returns the mean animal movement propensity
     *
     * @return Lambda as double
     */
    double getLambdaMean() {
        return this->lambdaMean;
    }
    /**
     * Sets the mean animal movement propensity
     *
     * @param lm as double
     */
    void setLambdaMean(double lm) {
        this->lambdaMean = lm;
    }

    /**
     * Returns the animal movement propensity standard deviation
     *
     * @return LambdaSD as double
     */
    double getLambdaSD() {
        return this->lambdaSD;
    }
    /**
     * Sets the animal movement propensity standard deviation
     *
     * @param lsd as double
     */
    void setLambdaSD(double lsd) {
        this->lambdaSD = lsd;
    }

    /**
     * Returns the ranging coefficient mean
     *
     * @return Ranging coefficient mean as double
     */
    double getRangingCoeffMean() {
        return this->rangingCoeffMean;
    }
    /**
     * Sets the ranging coefficient mean
     *
     * @param rcm as double
     */
    void setRangingCoeffMean(double rcm) {
        this->rangingCoeffMean = rcm;
    }

    /**
     * Returns the ranging coefficient standard deviation
     *
     * @return Ranging coefficient standard deviation as double
     */
    double getRangingCoeffSD() {
        return this->rangingCoeffSD;
    }
    /**
     * Sets the ranging coefficient standard deviation
     *
     * @param rcsd as double
     */
    void set(double rcsd) {
        this->rangingCoeffSD = rcsd;
    }

    /**
     * Returns the mean population growth rate parameter (p.a.)
     *
     * @return Mean growth rate as double
     */
    double getGrowthRateMean() {
        return this->growthRateMean;
    }
    /**
     * Sets the mean population growth rate parameter (p.a.)
     *
     * @param grm as double
     */
    void setGrowthRateMean(double grm) {
        this->growthRateMean = grm;
    }

    /**
     * Returns the growth rate standard deviation (p.a.)
     *
     * @return Growth rate standard deviation as double
     */
    double getGrowthRateSD() {
        return this->growthRateSD;
    }
    /**
     * Sets the growth rate standard deviation (p.a.)
     *
     * @param grsd as double
     */
    void setGrowthRateSD(double grsd) {
        this->growthRateSD = grsd;
    }

    /**
     * Returns the mean animal lenth (m)
     *
     * @return Mean animal length as double
     */
    double getLengthMean() {
        return this->lengthMean;
    }
    /**
     * Sets the mean animal length (m)
     *
     * @param lenm as double
     */
    void setLengthMean(double lenm) {
        this->lengthMean = lenm;
    }

    /**
     * Returns the animal length standard deviation (m)
     *
     * @return Animal length standard deviation as double
     */
    double getLengthSD() {
        return this->lengthSD;
    }
    /**
     * Sets the animal length standard deviation (m)
     *
     * @param lensd as double
     */
    void setLengthSD(double lensd) {
        this->lengthSD = lensd;
    }

    /**
     * Returns the mean crossing speed (m/s)
     *
     * @return Mean crossing speed as double
     */
    double getSpeedMean() {
        return this->speedMean;
    }
    /**
     * Sets the mean crossing speed (m/s)
     *
     * @param sm as double
     */
    void setSpeedMean(double sm) {
        this->speedMean = sm;
    }

    /**
     * Returns the crossing speed standard deviation (m/)
     *
     * @return Crossing speed standard deviation
     */
    double getSpeedSD() {
        return this->speedSD;
    }
    /**
     * Sets the crossing speed standard deviation (m/s)
     *
     * @param spsd as double
     */
    void setSpeedSD(double spsd) {
        this->speedSD = spsd;
    }

    /**
     * Returns the cost per animal below required threshold ($)
     *
     * @return Cost as double
     */
    double getCostPerAnimal() {
        return this->costPerAnimal;
    }
    /**
     * Sets the cost per animal below required threshold ($)
     *
     * @param cost as double
     */
    void setCostPerAnimal(double cost) {
        this->costPerAnimal = cost;
    }

    /**
     * Returns the habitat type of each grid location
     *
     * @return Habitat types as const std::vector<HabitatTypePtr>&
     */
    const std::vector<HabitatTypePtr>& getHabitatTypes() {
        return this->habitat;
    }
    /**
     * Sets the habitat type of each grid location
     *
     * @param habitat as const std::vector<HabitatTypePtr>&
     */
    void setHabitat(const std::vector<HabitatTypePtr>& habitat) {
        this->habitat = habitat;
    }

    /**
     * Returns the map of habitat types for this species
     *
     * @return HabitatType map as const Eigen::MatrixXi&
     */
    const Eigen::MatrixXi& getHabitatMap() {
        return this->habitatMap;
    }
    /**
     * Sets the map of habitat types for this species
     *
     * @param habMap as const Eigen::MatrixXi&
     */
    void setHabitatMap(const Eigen::MatrixXi& habMap) {
        this->habitatMap = habMap;
    }

    /**
     * Returns the population map of the species
     *
     * @return Population map as const Eigen::MatrixXd&
     */
    const Eigen::MatrixXd& getPopulationMap() {
        return this->populationMap;
    }
    /**
     * Sets the population map of the species
     *
     * @param popMap as const Eigen::MatrixXd&
     */
    void setPopulationMap(const Eigen::MatrixXd& popMap) {
        this->populationMap = popMap;
    }

    /**
     * Returns the target population threshold (as a faction of initial)
     *
     * @return Threshold as double
     */
    double getThreshold() {
        return this->threshold;
    }
    /**
     * Sets the target population threshold
     *
     * @param threshold as double
     */
    void setThreshold(double threshold) {
        this->threshold = threshold;
    }

    // STATIC ROUTINES /////////////////////////////////////////////////////////

    // CALCULATION ROUTINES ////////////////////////////////////////////////////

    /**
     * Builds the habitat map using input vegetation data.
     *
     * This routine is only run once at the beginning of the analysis.
     */
    void generateHabitatMap(OptimiserPtr optimiser);

////////////////////////////////////////////////////////////////////////////////
private:
    std::string name;                   /**< Species name */
    bool active;                        /**< Whether the species is used in the simulation */
    bool sex;                           /**< 1 = female, 0 = male */
    double lambdaMean;                  /**< Mean movement propensity */
    double lambdaSD;                    /**< Movement propensity standard deviation */
    double rangingCoeffMean;            /**< Ranging coefficient mean */
    double rangingCoeffSD;              /**< Ranging coefficient standard deviation */
    double growthRateMean;              /**< Mean growth rate (%p.a.) */
    double growthRateSD;                /**< Growth rate (%p.a.) standard devation */
    double lengthMean;                  /**< Mean species length */
    double lengthSD;                    /**< Species length standard deviation */
    double speedMean;                   /**< Mean species road crossing speed */
    double speedSD;                     /**< Species road crossing speed standard deviation */
    double costPerAnimal;               /**< Cost per animal below threshold ($) */
    std::vector<HabitatTypePtr> habitat;/**< Habitat type */
    Eigen::MatrixXi habitatMap;         /**< Breakdown of region into the four base habitat types */
    Eigen::MatrixXd populationMap;      /**< Population of animals in each cell */
    double threshold;                   /**< Target threshold as proportion of initial population */
    SpeciesPtr me();                    /**< Enables sharing from this */
};

#endif
