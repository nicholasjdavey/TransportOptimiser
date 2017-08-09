#include "../transportbase.h"

Vehicle::Vehicle() {
    // Default values
    this->name = "";
    this->averageWidth = 0;
    this->averageLength = 0;
    this->trafficProportion = 0;
    this->maxLoad = 0;
    this->aConst = 0;
    this->agr = 0;
    this->av = 0;
    this->avsq = 0;
    this->travelPerHrCost = 0;
}

Vehicle::Vehicle(CommodityPtr fuel, std::string nm, double width, double length,
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

Vehicle::~Vehicle() {

}
