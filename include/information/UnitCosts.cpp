#include "../transportbase.h"

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
