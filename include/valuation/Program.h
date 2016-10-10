#ifndef PROGRAM_H
#define PROGRAM_H

class Program;
typedef std::shared_ptr<Program> ProgramPtr;

/**
 * Class for managing policy programs for ROV.
 */
class Program : public std::enable_shared_from_this<Program> {

public:
	// CONSTRUCTORS AND DESTRUCTORS ///////////////////////////////////////////

	/**
	 * Constructor Constructs a %Program object with default values.
	 */
    Program(const Eigen::VectorXd& flowRates, const Eigen::MatrixXd&
            switching);

	/**
	 * Destructor
	 */
	~Program();

	// ACCESSORS //////////////////////////////////////////////////////////////
	/**
	 * Returns the program index
	 *
	 * @return Index as unsigned long
	 */
	unsigned long getNumber() {
		return this->number;
	}

	/**
	 * Returns the vector of flow rates
	 *
     * @return Flow rates as const Eigen::VectorXd&
	 */
    const Eigen::VectorXd& getFlowRates() {
        return this->flowRates;
	}
	/**
	 * Sets the vector of flow rates
	 *
     * @param rates as const Eigen::VectorXd&
	 */
    void setFlowRates(const Eigen::VectorXd& rates) {
        this->flowRates = rates;
	}

	/**
	 * Returns the matrix of switching costs
	 *
     * @return Switching matrix as const Eigen::MatrixXd&
	 */
    const Eigen::MatrixXd& getSwitchingCosts() {
        return this->switching;
	}
	/**
	 * Sets the matrix of switching costs
	 *
     * @param costs as const Eigen::MatrixXd&
	 */
    void setSwitchingCosts(const Eigen::MatrixXd& costs) {
        this->switching = costs;
	}

	/**
	 * Returns the currently-selected option
	 *
	 * @return Selected option as unsigned long
	 */
	unsigned long getSelected() {
		return this->selected;
    }
	/**
	 * Sets the currently-selected option
	 *
	 * @param sel as unsigned long
	 */
	void setSelected(unsigned long sel) {
		this->selected = sel;
	}

	// STATIC ROUTINES ////////////////////////////////////////////////////////

	// CALCULATION ROUTINES ///////////////////////////////////////////////////

private:
	unsigned long number;			/**< Program identifier */
	Eigen::VectorXd flowRates;		/**< Flow rate options associated with program */
    Eigen::MatrixXd switching;		/**< Switching costs between controls */
	unsigned long selected;			/**< Currently selected option */
	static unsigned long programs;	/**< Number of programs */
};

#endif
