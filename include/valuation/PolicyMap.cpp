#include "../transportbase.h"

PolicyMap::PolicyMap(ProgramPtr program, unsigned long noYears) {
	this->program = program;

	std::vector<PolicyMapYearPtr> years(noYears);

	for(unsigned long ii = 0; ii<noYears; ii++) {
		PolicyMapYearPtr year(new PolicyMapYear());
		years[ii] = year;
	}
}

PolicyMap::PolicyMap(ProgramPtr program, std::vector<PolicyMapYearPtr>* years) {
	this->program = program;
	this->yearlyMaps = *years;
}
