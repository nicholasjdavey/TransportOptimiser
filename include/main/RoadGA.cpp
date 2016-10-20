#include "../transportbase.h"

RoadGA::RoadGA() : Optimiser() {

}

RoadGA::RoadGA(const std::vector<TrafficProgramPtr>& programs, OtherInputsPtr
    oInputs, DesignParametersPtr desParams, EarthworkCostsPtr earthworks,
    UnitCostsPtr unitCosts, VariableParametersPtr varParams, const
    std::vector<SpeciesPtr>& species, EconomicPtr economic, TrafficPtr traffic,
    RegionPtr region, double mr, unsigned long cf, unsigned long gens, unsigned
    long popSize, double stopTol, double confInt, double confLvl, unsigned long
    habGridRes, std::string solScheme, unsigned long noRuns,
    Optimiser::Type type) :
    Optimiser(programs, oInputs, desParams, earthworks, unitCosts, varParams,
            species, economic, traffic, region, mr, cf, gens, popSize, stopTol,
            confInt, confLvl, habGridRes, solScheme, noRuns, type) {

}

void RoadGA::creation() {
    unsigned long individualsPerPop = this->populationSizeGA/5;

    double sx = this->designParams->getStartX();
    double sy = this->designParams->getStartY();
    double ex = this->designParams->getEndX();
    double ey = this->designParams->getEndY();

    double gmax = this->designParams->getMaxGrade();
    double eHigh = this->designParams->getMaxSE()/100;
    double velDes = this->designParams->getDesignVelocity();

    // Region limits. We ignore the two outermost cell boundaries
    double minLon = (this->getRegion()->getX())(3,1);
    double maxLon = (this->getRegion()->getX())(this->getRegion()->getX().
            rows()-2,1);
    double minLat = (this->getRegion()->getY())(1,3);
    double maxLat = (this->getRegion()->getY())(1,this->getRegion()->getY().
            cols()-2);

    if (sx > maxLon || sx < minLon || ex > maxLon || ex < minLon ||
            sy > maxLat || sy < minLat || ey > maxLat || ey < minLat) {
        std::cerr << "One or more of the end points lie outside the region of interest"
                  << std::endl;
    }

    // The length of the genome is essentially the points of intersection.
    // However, the length of the individuals used as an input includes the
    // start and end points. Furthermore, these are broken down into three
    // coordinates for each point. We adjust to extract the real genome length
    // from this information.
    unsigned long genomeLength = this->designParams->getIntersectionPoints();

    // Create population using the initial population routine devised by
    // Jong et al. (2003)

    // First compute the elevations of the start and end points and place them
    // as the start and end points of the intersection points vectors that
    // represent the population roads
    this->currentRoadPopulation;
}

void RoadGA::crossover() {}

void RoadGA::mutation() {}

void RoadGA::optimise() {}

void RoadGA::output() {}

void RoadGA::computeSurrogate() {}
