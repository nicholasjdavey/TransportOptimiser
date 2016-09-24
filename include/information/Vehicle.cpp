#include "../transportbase.h"

Vehicle::Vehicle(FuelPtr fuel, std::string nm, double width, double length, 
		double trafficProp, double load, double a, double agr, double av,
		double avsq, double travel) {

	this->fuel = fuel;
	this->name = nm;
	this->averageWidth = width;
	this->averageLength = length;
	this->trafficProportion = trafficProp;
	this->maxLoad = load;
	this->aConst = a;
	this->agr = agr;
	this->av = av;
	this->avsq = avsq;
	this->travelPerHrCost = travel;
}
