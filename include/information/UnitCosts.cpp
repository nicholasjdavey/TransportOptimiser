#include "../transportbase.h"

UnitCosts::UnitCosts() {
    // Default values
    this->perAccident = 0;
    this->airPollution = 0;
    this->noisePollution = 0;
    this->waterPollution = 0;
    this->oilExtractDistUse = 0;
    this->landUse = 0;
    this->solidChemWaste = 0;
}

UnitCosts::UnitCosts(double acc, double air, double noise, double water, double oil,
		double land, double chem) {

    this->perAccident = acc;
    this->airPollution = air;
    this->noisePollution = noise;
    this->waterPollution = water;
    this->oilExtractDistUse = oil;
    this->landUse = land;
    this->solidChemWaste = chem;
}

UnitCosts::~UnitCosts() {

}
