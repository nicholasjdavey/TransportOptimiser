#include "../transportbase.h"

PolicyMap::PolicyMap(ProgramPtr program, unsigned long noYears, unsigned long
        noPaths, unsigned long noDims) {
    this->program = program;

    std::vector<PolicyMapYearPtr> years(noYears);

    for(unsigned long ii = 0; ii<noYears; ii++) {
        PolicyMapYearPtr year(new PolicyMapYear(noPaths,noDims));
        years[ii] = year;
    }

    this->setPolicyMapYear(years);
}

PolicyMap::PolicyMap(ProgramPtr program, const std::vector<PolicyMapYearPtr>
        &years) {
    this->program = program;
    this->yearlyMaps = years;
}

PolicyMap::~PolicyMap() {

}
