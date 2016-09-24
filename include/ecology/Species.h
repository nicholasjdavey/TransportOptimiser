#ifndef SPECIES_H
#define SPECIES_H

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
class Species : public Uncertainty, 
		public std::enable_shared_from_this<Species> {

public:
	// CONSTRUCTORS AND DESTRUCTORS ////////////////////////////////////////////
	
	/**
	 * Constructor I
	 *
	 * Constructs a %Species object with default parent values.
	 */
    Species(std::string* nm, bool sex, double lm, double lsd, double rcm,
            double rcsd, double grm, double grsd, double lenm, double lensd,
            double spm, double spsd, double cpa, bool active,
            std::vector<HabitatTypePtr> *habitat);

	/**
	 * Constructor II
	 *
	 * Constructs a %Species object with assigned values.
	 */
    Species(std::string* nm, bool sex, double lm, double lsd, double rcm,
            double rcsd, double grm, double grsd, double lenm, double lensd,
            double spm, double spsd, double cpa, bool active,
            std::vector<HabitatTypePtr> *habitat, double current,
            double meanP, double mean, double stdDev, double rev);

	/**
	 * Destructor
	 */
	~Species();

	// ACCESSORS ///////////////////////////////////////////////////////////////

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
     * @return Habitat types as std::vector<HabitatTypePtr>*
	 */
    std::vector<HabitatTypePtr>* getHabitatTypes() {
		return &this->habitat;
	}
	/**
	 * Sets the habitat type of each grid location
	 *
     * @param habitat as std::vector<HabitatTypePtr>*
	 */
    void setHabitat(std::vector<HabitatTypePtr>* habitat) {
		this->habitat = *habitat;
	}

	/**
	 * Returns the habitat patches for simulations
	 *
     * @return Habitat patches as std::vector<HabitatPatchPtr>*
	 */
    std::vector<HabitatPatchPtr>* getHabPatches() {
		return &this->habPatch;
	}
	/**
	 * Sets the habitat patches during simulations
	 *
     * @param habp as std::vector<HabitatPatchPtr>*
	 */
    void setHabPatches(std::vector<HabitatPatchPtr>* habp) {
		this->habPatch = *habp;
    }

    /**
     * Returns the map of habitat types for this species
     *
     * @return HabitatType map as Eigen::MatrixXi*
     */
    Eigen::MatrixXi* getHabitatMap() {
        return &this->habitatMap;
    }
    /**
     * Sets the map of habitat types for this species
     *
     * @param habMap as Eigen::MatrixXi*
     */
    void setHabitatMap(Eigen::MatrixXi* habMap) {
        this->habitatMap = *habMap;
    }

	// STATIC ROUTINES /////////////////////////////////////////////////////////

	// CALCULATION ROUTINES ////////////////////////////////////////////////////

////////////////////////////////////////////////////////////////////////////////
private:
        bool sex;						/**< 1 = female, 0 = male */
        double lambdaMean;					/**< Mean movement propensity */
        double lambdaSD;					/**< Movement propensity standard deviation */
        double rangingCoeffMean;				/**< Ranging coefficient mean */
        double rangingCoeffSD;					/**< Ranging coefficient standard deviation */
        double growthRateMean;					/**< Mean growth rate parameters */
        double growthRateSD;					/**< Growth rate standard devation */
        double lengthMean;					/**< Mean species length */
        double lengthSD;					/**< Species length standard deviation */
        double speedMean;					/**< Mean species road crossing speed */
        double speedSD;						/**< Species road crossing speed standard deviation */
        double costPerAnimal;					/**< Cost per animal below threshold ($) */
        std::vector<HabitatTypePtr> habitat;	/**< Habitat type */
        std::vector<HabitatPatchPtr> habPatch;   /**< Habitat patches (for simulations) */
        Eigen::MatrixXi habitatMap;         /**< Breakdown of region into the four base habitat types */
};

#endif
