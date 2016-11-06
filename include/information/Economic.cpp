#include "../transportbase.h"

Economic::Economic() {
    this->reqRate = 0;
    this->nYears = 0;
}

Economic::Economic(const std::vector<CommodityPtr> &commodities,
        const std::vector<CommodityPtr> &fuels, double rr, double ny) {

    this->reqRate = rr;
    this->nYears = ny;
    this->commodities = commodities;
    this->fuels = fuels;
}

Economic::~Economic() {
}
