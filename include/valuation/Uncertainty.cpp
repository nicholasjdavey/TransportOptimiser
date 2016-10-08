#include "../transportbase.h"

Uncertainty::Uncertainty() {

	this->name = "";
	this->current = 0;
	this->meanP = 0;
	this->standardDev = 0;
	this->reversion = 0;
	this->active = false;
}

Uncertainty::Uncertainty(std::string nm, double mp, double sd, double rev,
		bool active) {

	this->name = nm;
	this->current = 0;
	this->meanP = mp;
	this->standardDev = sd;
	this->reversion = rev;
	this->active = active;
}

Uncertainty::~Uncertainty() {
}

void Uncertainty::computeExpPV() {

}
