#include "../transportbase.h"

OtherInputs::OtherInputs() {
    // Default values
    this->inputDataFile = "";
    this->outputResultsFile = "";
    this->inputTerrainFile = "";
    this->existingRoadsFile = "";
    this->minLat = 0;
    this->maxLat = 0;
    this->minLon = 0;
    this->maxLon = 0;
    this->latPoints = 0;
    this->lonPoints = 0;
    this->habGridRes = 0;
    this->noPaths = 0;
    this->dimRes = 0;
}

OtherInputs::OtherInputs(std::string& idf, std::string& orf, std::string& itf,
        std::string& erf, double minLat, double maxLat, double minLon,
        double maxLon, unsigned long latPoints, unsigned long lonPoints,
        unsigned long habGridRes, unsigned long noPaths, unsigned long
        dimRes) {

    // Initialise object
    this->inputDataFile = idf;
    this->outputResultsFile = orf;
    this->inputTerrainFile = itf;
    this->existingRoadsFile = erf;
    this->minLat = minLat;
    this->maxLat = maxLat;
    this->minLon = minLon;
    this->maxLon = maxLon;
    this->latPoints = latPoints;
    this->lonPoints = lonPoints;
    this->habGridRes = habGridRes;
    this->noPaths = noPaths;
    this->dimRes = dimRes;
}

OtherInputs::~OtherInputs() {

}
