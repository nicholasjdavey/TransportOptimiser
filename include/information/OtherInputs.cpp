#include "../transportbase.h"

OtherInputs::OtherInputs(std::string& idf, std::string& orf, std::string& itf,
        std::string& erf, double minLat, double maxLat, double minLon,
        double maxLon, unsigned long latPoints, unsigned long lonPoints,
        unsigned long habGridRes, unsigned long noPaths) {

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
}

OtherInputs::~OtherInputs() {

}
