#include "../include/transportbase.h"

int main(int argc, char **argv) {

    // INITIALISE THE OPTIMISER OBJECT FIRST
    // We will later add the attributes to the Optimiser Object
    std::string solScheme = "GA";

    RoadGAPtr roadGA(new RoadGA());

    RoadGAPtr roadGA(new RoadGA(programs,otherInputs,desParams,earthwork,
            unitCosts,varParams,species,economic,traffic,region,0.4,0.55,500,
            400,0.1,0.95,0.95,10,solScheme,5,Optimiser::MTE,0.5,1,50,0.05,50,
            3,10,RoadGA::TOURNAMENT,RoadGA::RANK,0.4,0.8,5));

    // Initialise the input classes
    // SPECIES
    Eigen::VectorXi habVec(1);
    habVec << 1;
    HabitatTypePtr habType(new HabitatType(HabitatType::PRIMARY, 0.1, habVec,
            2.5714e-5,0,1000000));
    std::vector<HabitatTypePtr> habTyps(1);
    habTyps[0] = habType;

    std::string nm = "species1";

    SpeciesPtr animal(new Species(nm,false,2.52e-3,0.0928e-3,-2.52e-3,0.1014e-3,
            1.4,0.5,0.7,0.012,2.78,1.39,0.1,true,habTyps));

    std::vector<SpeciesPtr> species(1);
    species[0] = animal;

    // DESIGN PARAMETERS
    DesignParametersPtr desParams(new DesignParameters(100,100,100,4900,4900,
            5,8,15,2.5,3.4,8,20,M_PI/4,M_PI/6,0,0,0,0,0,0,100,3.621,
            0.355,0.1562,2.4282,4.331,0.1704,false));

    // COMMODITIES
    std::string dieselName = "diesel";
    std::string petrolName = "petrol";
    std::string commodityName = "ore";

    CommodityPtr diesel(roadGA,dieselName,1.2,0.01,0.01,true);
    CommodityPtr petrol(roadGA,dieselName,1.05,0.01,0.01,true);

    // VEHICLES
    std::string smallCarName = "small";
    std::string mediumVehName = "medium";
    std::string largeVehicleName = "large";
    VehiclePtr small(new Vehicle(petrol,smallCarName,1.7,2.5,0.6,0.5,171.3978,9.8565,2.5314,0.0219,13.23));
    VehiclePtr medium(new Vehicle(petrol,mediumVehName,2,5,0.3,5,620.5076,30.9489,9.0353,0.061,30.87));
    VehiclePtr large(new Vehicle(diesel,largeVehicleName,2.5,10,0.1,20,1198.906,70.226,14.4698,0.081,35.28));

    std::vector<VehiclePtr> vehicles(3);
    vehicles[0] = small;
    vehicles[1] = medium;
    vehicles[2] = large;

    // TRAFFIC INFORMATION
    TrafficPtr traffic(new Traffic(vehicles,0.5,0.5,6,0));

    // TRAFFIC PROGRAMS
    // Only one program with three flow rates
    Eigen::VectorXd flows(3);
    flows << 0,75,150;
    Eigen::MatrixXd switching(flows.size(),flows.size());
    switching << 0, 0, 0, 0, 0, 0, 0, 0, 0;

    TrafficProgramPtr prog1(new TraficProgram(false, flows,switching));
    std::vector<TrafficProgramPtr> programs(1);
    programs[1] = prog1;

    // OTHER INPUTS
    OtherInputsPtr otherInputs(new OtherInputs());

    // EARTHWORK COSTS
    EarthworkCostsPtr earthwork(new EarthworkCosts());

    // UNIT COSTS
    UnitCostsPtr unitCosts(new UnitCosts());

    // VARIABLE PARAMETERS
    VariableParametersPtr varParams(new VariableParameters());

    // ECONOMIC
    EconomicPtr economic(new Economic());

    // TRAFFIC
    TrafficPtr traffic(new Traffic());

    // REGION
    RegionPtr region(new Region());
    opt->setDesignParams(desParams);

    // ADD THE COMPONENTS TO THE OPTIMISER OBJECT
    roadGA->setPrograms(programs);
    roadGA->setOtherInputs(otherInputs);
    roadGA->setDesignParams(desParams);
    roadGA->setEarthworkCosts(earthwork);
    roadGA->setUnitCosts(unitCosts);
    roadGA->setVariableParams(varParams);
    roadGA->setSpecies(species);
    roadGA->setEconomic(economic);
    roadGA->setTraffic(traffic);
    roadGA->setRegion(region);

    // ADD A TEST ROAD
    Eigen::RowVectorXd genome(30);

    genome << 0,0,0,2000,0,50,4000,2000,100,4000,6000,0,1000,7000,0,2000,8000,-50,5000,7000,-150,7000,8000,-50,8000,4000,0,8500,3000,0;

    RoadPtr trialRoad(new Road(opt,genome));
}
