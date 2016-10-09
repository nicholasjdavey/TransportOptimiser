#include "../transportbase.h"

Traffic::Traffic() {

	this->peakProportion = 0.5;
	this->directionality = 0.5;
	this->peakHours = 6;
	this->growthRate = 0;
}

Traffic::Traffic(const std::vector<VehiclePtr> &vehicles, double peakProp,
        double d, double peak, double gr) {

    this->vehicles = vehicles;
	this->peakProportion = peakProp;
	this->directionality = d;
	this->peakHours = peak;
	this->growthRate = gr;
}

Traffic::~Traffic() {
}
