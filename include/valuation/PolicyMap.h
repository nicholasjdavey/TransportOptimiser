#ifndef POLICYMAP_H
#define POLICYMAP_H

class Program;
typedef std::shared_ptr<Program> ProgramPtr;

class PolicyMapYear;
typedef std::shared_ptr<PolicyMapYear> PolicyMapYearPtr;

/**
 * Class for managing %PolicyMap objects.
 */
class PolicyMap : public std::enable_shared_from_this<PolicyMap> {

public:
    // CONSTRUCTORS AND DESTRUCTORS ///////////////////////////////////////////

    /**
     * Constructor I
     *
     * Constructs a %PolicyMap object by passing the program and number of years.
     */
    PolicyMap(ProgramPtr program, unsigned long noYears);

    /**
     * Constructor II
     *
     * Constructs a %PolicyMap object by passing attribute values.
     */
    PolicyMap(ProgramPtr program, const std::vector<PolicyMapYearPtr>& years);

    /**
     * Destructor
     */
    ~PolicyMap();

    // ACCESSORS //////////////////////////////////////////////////////////////

    /**
     * Returns the the Program used
     *
     * @return Program as ProgramPtr
     */
    ProgramPtr getProgram() {
            return this->program;
    }
    /**
     * Sets the Program used
     *
     * @param prog as ProgramPtr
     */
    void setProgram(ProgramPtr prog) {
    this->program.reset();
            this->program = prog;
    }

    /**
     * Returns the policy map by year
     *
     * @return Policy map as const std::vector<PolicyMapYearPtr>&
     */
    const std::vector<PolicyMapYearPtr>& getPolicyMapYear() {
        return this->yearlyMaps;
    }
    /**
     * Sets the policy map by year
     *
     * @param pmy as const std::vector<PolicyMapYearPtr>&
     */
    void setPolicyMapYear(const std::vector<PolicyMapYearPtr>& pmy) {
        this->yearlyMaps = pmy;
    }

    // STATIC ROUTINES ////////////////////////////////////////////////////////

    // CALCULATION ROUTINES ///////////////////////////////////////////////////

private:
    ProgramPtr program;                         /**< The program used for the policy map */
    std::vector<PolicyMapYearPtr> yearlyMaps;	/**< Vector of policy maps by year */
};

#endif
