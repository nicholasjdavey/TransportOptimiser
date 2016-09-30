#ifndef UNCERTAINTY_H
#define UNCERTAINTY_H

class Uncertainty;
typedef std::shared_ptr<Uncertainty> UncertaintyPtr;

/**
 * Class for managing %Uncertainty objects
 */
class Uncertainty {

public:
	// CONSTRUCTORS AND DESTRUCTORS ///////////////////////////////////////////

	/**
	 * Constructor I
	 *
	 * Constructs an %Uncertainty object with default values
	 */
	Uncertainty();

	/**
	 * Constructor II
	 *
	 * Constructs an %Uncertainty object with assigned values
	 */
	Uncertainty(std::string nm, double mp, double sd, double rev, bool active);

	/**
	 * Destructor
	 */
	~Uncertainty();

	// ACCESSORS //////////////////////////////////////////////////////////////
	
	/**
	 * Returns the name
	 *
	 * @return Name as std::string
	 */
	std::string getName() {
		return this->name;
	}
	/**
	 * Sets the name
	 *
	 * @param nm as std::string
	 */
	void setName(std::string nm) {
		this->name = nm;
	}

	/**
	 * Returns the current level of the uncertainty
	 *
	 * @return Current as double
	 */
	double getCurrent() {
		return this->current;
	}
	/**
	 * Sets the current level of the uncertainty
	 *
	 * @param curr as double
	 */
	void setCurrent(double curr) {
		this->current = curr;
	}

	/**
	 * Returns the long run mean
	 *
	 * @return Long run mean as double
	 */
	double getMean() {
		return this->meanP;
	}
	/**
	 * Sets the long run mean
	 *
	 * @param mean as double
	 */
	void setMean(double mean) {
		this->meanP = mean;
	}

	/**
	 * Returns standard deviation for noise
	 *
	 * @return Standard deviation as double
	 */
	double getNoiseSD() {
		return this->standardDev;
	}
	/**
	 * Sets the standard deviation for noise
	 *
	 * @param sd as double
	 */
	void setNoiseSD(double sd) {
		this->standardDev = sd;
	}

	/**
	 * Returns the strength of mean reversion
	 *
	 * @return Mean reversion strength as double
	 */
	double getMRStrength() {
		return this->reversion;
	}
	/**
	 * Sets the strength of mean reversion
	 *
	 * @param mrs as double
	 */
	void setMRStrength(double mrs) {
		this->reversion = mrs;
	}

	/**
	 * Returns whether the commodity is active
	 *
	 * @return Active status as bool
	 */
	bool getStatus() {
		return this->active;
	}
	/**
	 * Sets whether the commodity is active
	 *
	 * @param status as bool
	 */
	void setStatus(bool status) {
		this->active = status;
	}


	// STATIC ROUTINES ////////////////////////////////////////////////////////

	// CALCULATION ROUTINES ///////////////////////////////////////////////////

private:
	std::string name;			/**< Name of the product */
	double current;				/**< Current level of uncertainty */
	double meanP;				/**< Long-run mean */
	double standardDev;			/**< Standard deviation */
	double reversion;			/**< Strength of mean reversion */
	bool active;				/**< Used in problem? */
};

#endif
