#include "../transportbase.h"

Attributes::Attributes(RoadPtr road) {
    this->road = road;
    this->unitVarCosts = 0.0;
    this->unitVarRevenue = 0.0;
    this->length = 0.0;
    this->varProfitIC = 0.0;
    this->totalValueMean = 0.0;
    this->totalValueSD = 0.0;
    this->totalUtilisationROV = 0.0;
    this->totalUtilisationROVSD = 0.0;
}

Attributes::Attributes(double uvc, double uvr, double length, double vpic,
        double tvm, double tvsd, double turov, double turovsd, RoadPtr road) {
    this->road = road;
    this->unitVarCosts = uvc;
    this->unitVarRevenue = uvr;
    this->length = length;
    this->varProfitIC = vpic;
    this->totalValueMean = tvm;
    this->totalValueSD = tvsd;
    this->totalUtilisationROV = turov;
    this->totalUtilisationROVSD = turovsd;
}

Attributes::~Attributes() {
}
