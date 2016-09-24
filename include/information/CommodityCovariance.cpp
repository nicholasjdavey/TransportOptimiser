#include "../transportbase.h"

CommodityCovariance::CommodityCovariance(CommodityPtr com1, CommodityPtr com2,
		double cov) {
	this->commodity1 = com1;
	this->commodity2 = com2;
	this->covariance = cov;
}

CommodityCovariance::~CommodityCovariance() {
}
