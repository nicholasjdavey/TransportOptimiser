#include "../transportbase.h"

Attributes::Attributes(RoadPtr road) {
	this->road = road;
	this->initAAR = 0.0;
	this->endPopMTE = 0.0;
	this->unitVarCosts = 0.0;
	this->unitVarRevenue = 0.0;
	this->length = 0.0;
	this->varProfitIC = 0.0;
	this->totalValueMTE = 0.0;
	this->totalValueROV = 0.0;
}

Attributes::Attributes(double iar, double endpop, double uvc, double uvr,
		double length, double vpic, double tvmte, double tvrov,
		RoadPtr road) {
	this->road = road;
	this->initAAR = iar;
	this->endPopMTE = endpop;
	this->unitVarCosts = uvc;
	this->unitVarRevenue = uvr;
	this->length = length;
	this->varProfitIC = vpic;
	this->totalValueMTE = tvmte;
	this->totalValueROV = tvrov;
}

Attributes::~Attributes() {
}
