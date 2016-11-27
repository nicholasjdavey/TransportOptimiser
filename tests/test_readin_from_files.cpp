#include "../include/transportbase.h"

int main(int argc, char **argv) {

    // INITIALISE THE OPTIMISER OBJECT FIRST
    // We will later add the attributes to the Optimiser Object
    std::string solScheme = "GA";

    RoadGAPtr roadGA(new RoadGA(0.4,0.55,500,400,0.1,0.95,0.95,10,solScheme,5,
            Optimiser::MTE,1.0,50,0.05,50,3,10,RoadGA::TOURNAMENT,
            RoadGA::RANK,0.4,0.8,5));

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

    CommodityPtr diesel(new Commodity(roadGA,dieselName,1.2,0.01,0.01,true));
    diesel->setCurrent(1.2);
    CommodityPtr petrol(new Commodity(roadGA,petrolName,1.05,0.01,0.01,true));
    petrol->setCurrent(1.05);
    CommodityPtr ore(new Commodity(roadGA,commodityName,100,0.1,0.01,true));
    ore->setCurrent(120);

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

    TrafficProgramPtr prog1(new TrafficProgram(false,traffic,flows,switching));
    std::vector<TrafficProgramPtr> programs(1);
    programs[0] = prog1;

    // OTHER INPUTS
    std::string idf = "Input Data/input_data_file.csv";
    std::string orf = "Output Data/output_data_file.csv";
    std::string itf = "Input Data/input_terrain_file.csv";
    std::string erf = "Input Data/existing_roads_file.csv";
    OtherInputsPtr otherInputs(new OtherInputs(idf,orf,itf,erf,0,1,0,1,1000,
            20));

    // EARTHWORK COSTS
    Eigen::VectorXd cd(6);
    cd << 0,1.5,3.0,4.5,6.0,7.5;
    Eigen::VectorXd cc(6);
    cc << 40.0,57.6,72.8,100.0,120.0,200.0;
    EarthworkCostsPtr earthwork(new EarthworkCosts(cd,cc,40/1.623));

    // UNIT COSTS
    UnitCostsPtr unitCosts(new UnitCosts(4852,1.42*2.55,1.42*0.125,1.42*0.11,
            1.42*1.71,1.42*3.05,1.42*0.12));

    // VARIABLE PARAMETERS
    Eigen::VectorXd popLevels(6);
    popLevels << 50,60,70,80,90,100;
    Eigen::VectorXi bridge(1);
    bridge << 0;
    Eigen::VectorXd hp(1);
    hp << 100;
    Eigen::VectorXd l(3);
    l << -1,0,1;
    Eigen::VectorXd beta(1);
    beta << 0;
    Eigen::VectorXd pgr(3);
    pgr << 50,100,150;
    Eigen::VectorXd pgrsd(3);
    pgrsd << 1,2,3;
    Eigen::VectorXd c(1);
    c << 100;
    Eigen::VectorXd csd(3);
    csd << 1,2,3;
    VariableParametersPtr varParams(new VariableParameters(popLevels,bridge,
            hp,l,beta,pgr,pgrsd,c,csd));

    // ECONOMIC
    EconomicPtr economic(new Economic());

    // REGION
    std::string regionData = "Input Data/region.csv";
    RegionPtr region(new Region(regionData,true));

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
    // Now that we have all of the contained objects within the roadGA object,
    // we can initialise the storage elements
    roadGA->initialiseStorage();

    // ADD A TEST ROAD
    Eigen::RowVectorXd genome(30);

    genome << 0,0,0,2000,0,50,4000,2000,100,4000,6000,0,1000,7000,0,2000,8000,-50,5000,7000,-150,7000,8000,-50,8000,4000,0,8500,3000,0;

    RoadPtr trialRoad(new Road(roadGA,genome));
    std::cout << "Read in test success" << std::endl;
}
