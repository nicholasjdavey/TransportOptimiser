#include "../transportbase.h"

HabitatPatch::HabitatPatch() {
	
	HabitatTypePtr hab(new HabitatType());
	hab->setType(HabitatType::OTHER);

	this->type = hab;
	this->area = 0;
	this->centroidX = 0.0;
	this->centroidY = 0.0;
	this->capacity = 0.0;
	this->growthRate = 0.0;
	this->population = 0.0;
	this->aar = 0.0;
}

HabitatPatch::HabitatPatch(HabitatTypePtr typ, double area, double cx,
		double cy, double cap, double gr, double pop, double aar) {
	this->type = typ;
	this->area = area;
	this->centroidX = cx;
	this->centroidY = cy;
	this->capacity = cap;
	this->growthRate = gr;
	this->population = pop;
	this->aar = aar;
}

HabitatPatch::~HabitatPatch() {
}
