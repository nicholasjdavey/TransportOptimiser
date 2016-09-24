#ifndef STATE_H
#define STATE_H

class Program;
typedef std::shared_ptr<Program> ProgramPtr;

class Uncertainty;
typedef std::shared_ptr<Uncertainty> UncertaintyPtr;

/**
 * Class for managing the simulator state
 */
class State : public std::enable_shared_from_this<State> {

public:
	// CONSTRUCTORS AND DESTRUCTORS //////////////////////////////////////////////

	/**
	 * Constructor
	 *
	 * Constructs a %State object using the problem variables
	 */
	State(double t, ProgramPtr p, std::vector<UncertaintyPtr>* ex,
			std::vector<UncertaintyPtr>* en);

	/**
	 * Destructor
	 */
	~State();

	// ACCESSORS /////////////////////////////////////////////////////////////////

	/**
	 * Returns the current time in the state
	 *
	 * @return Time as double
	 */
	double getTime() {
		return this->time;
	}
	/**
	 * Sets the current time in the state
	 *
	 * @param t as double
	 */
	void setTime(double t) {
		this->time = t;
	}

	/**
	 * Returns the control program
	 *
	 * @return Control program as ProgramPtr
	 */
	ProgramPtr getProgram() {
		return this->program;
	}
	/**
	 * Sets the control program
	 *
	 * @param program as ProgramPtr
	 */
	void setProgram(ProgramPtr program) {
		this->program = program;
	}

	/**
	 * Returns the exogenous uncertainties in the state
	 *
	 * @return Exogenous uncertainties as std::vector<UncertaintyPtr>*
	 */
	std::vector<UncertaintyPtr>* getExogenousUncertainties() {
		return &this->exogenousUncertainties;
	}
	/**
	 * Sets the exogenous uncertainties in the state
	 *
	 * @param ex as std::vector<UncertaintyPtr>*
	 */
	void setExogenousUncertainties(std::vector<UncertaintyPtr>* ex) {
		this->exogenousUncertainties = *ex;
	}

	/**
	 * Returns the endogenous uncertainties in the state
	 *
	 * @return Endogenous uncertainties as std::vector<UncertaintyPtr>*
	 */
	std::vector<UncertaintyPtr>* getEndogenousUncertainties() {
		return &this->endogenousUncertainties;
	}
	/**
	 * Sets the endogenous uncertainties in the state
	 *
	 * @param en as std::vector<UncertaintyPtr>*
	 */
	void setEndogenousUncertainties(std::vector<UncertaintyPtr>* en ) {
		this->endogenousUncertainties = *en;
	}

	// STATIC ROUTINES ///////////////////////////////////////////////////////////

	// CALCULATION ROUTINES //////////////////////////////////////////////////////

private:
	double time;										/**< Current time step (days) */
	ProgramPtr program;									/**< Control policy */
	std::vector<UncertaintyPtr> exogenousUncertainties;	/**< Exogenous uncertainties */
	std::vector<UncertaintyPtr> endogenousUncertainties;/**< Endogenous uncertainties */
};

#endif
