#include "../include/transportbase.h"

int main(int argc, char **argv) {

    // INITIALISE THE OPTIMISER OBJECT FIRST
    // We will later add the attributes to the Optimiser Object
    std::string solScheme = "GA";

    RoadGAPtr roadGA(new RoadGA(0.4,0.55,500,400,0.1,0.95,0.95,10,solScheme,5,
            Optimiser::SIMPLEPENALTY,1.0,50,0.05,50,3,10,RoadGA::TOURNAMENT,
            RoadGA::RANK,0.4,0.8,5));

    // Initialise the input classes
    // REGION
    std::string regionData = "Input Data/region.csv";
    RegionPtr region(new Region(regionData,true));

    // Create X and Y matrices
    Eigen::VectorXd x = Eigen::VectorXd::LinSpaced(201,-1000,15250);
    Eigen::RowVectorXd y = Eigen::RowVectorXd::LinSpaced(201,-1000,17000);

    Eigen::MatrixXd X = x.replicate<1,201>();
    Eigen::MatrixXd Y = y.replicate<201,1>();

    // Read in region Z coordinates
    std::ifstream regionFile;

    regionFile.open("Input_Data/Inputs/Regions/Region_1/zcoordsMatrix2.csv", std::ifstream::in);
    Eigen::MatrixXd Z(201,201);

    for (int ii = 0; ii < 201; ii++) {
        std::string line;
        std::getline(regionFile,line,'\n');
        std::stringstream ss(line);
        for (int jj = 0; jj < 201; jj++) {
            std::string substr;
            std::getline(ss,substr,',');
            std::stringstream subss(substr);
            double val;
            if (substr != "NaN") {
                subss >> val;
                Z(ii,jj) = val;
            } else {
                Z(ii,jj) = std::numeric_limits<double>::quiet_NaN();
            }
        }
    }
    regionFile.close();

    std::ifstream habFile;

    habFile.open("Input_Data/Inputs/Regions/Region_1/habMatrix2.csv");
    Eigen::MatrixXi V(201,201);

    for (int ii = 0; ii < 201; ii++) {
        std::string line;
        std::getline(habFile,line,'\n');
        std::stringstream ss(line);
        for (int jj = 0; jj < 201; jj++) {
            std::string substr;
            std::getline(ss,substr,',');
            std::stringstream subss(substr);
            double val;
            if (substr != "NaN") {
                subss >> val;
                V(ii,jj) = val;
            } else {
                V(ii,jj) = std::numeric_limits<int>::quiet_NaN();
            }
        }
    }
    habFile.close();

    region->setX(X);
    region->setY(Y);
    region->setZ(Z);
    region->setVegetation(V);

    Eigen::MatrixXd acq = Eigen::MatrixXd::Constant(201,201,500);
    region->setAcquisitionCost(acq);
    Eigen::MatrixXd stab = Eigen::MatrixXd::Constant(201,201,50);
    region->setSoilStabilisationCost(stab);

    // SPECIES
    // Habitat types. Always ordered as:
    // 0. PRIMARY
    // 1. MARGINAL
    // 2. OTHER
    // 3. CLEAR
    // 4. ROAD
    std::vector<HabitatTypePtr> habTypes(5);
    // PRIMARY/SECONDARY
    Eigen::VectorXi primaryVegSpec1(1);
    primaryVegSpec1 << 4;
    HabitatTypePtr primary(new HabitatType(HabitatType::PRIMARY,0.005,
            primaryVegSpec1,0,0,1e8));
    habTypes[0] = primary;
    // MARGINAL
    Eigen::VectorXi marginalVegSpec1(1);
    marginalVegSpec1 << 3;
    HabitatTypePtr marginal(new HabitatType(HabitatType::MARGINAL,0.0025,
            marginalVegSpec1,-0.262,0.073,5e7));
    habTypes[1] = marginal;
    // OTHER
    Eigen::VectorXi otherVegSpec1(1);
    otherVegSpec1 << 2;
    HabitatTypePtr other(new HabitatType(HabitatType::OTHER,0.001,
            otherVegSpec1,-0.396,0.120,5e6));
    habTypes[2] = other;
    // CLEAR
    Eigen::VectorXi clearVegSpec1(1);
    clearVegSpec1 << 1;
    HabitatTypePtr clear(new HabitatType(HabitatType::CLEAR,0.0000,
            clearVegSpec1,-0.373,0.175,0));
    habTypes[3] = clear;
    // ROAD
    Eigen::VectorXi roadVegSpec1(1);
    roadVegSpec1 << 0;
    HabitatTypePtr road(new HabitatType(HabitatType::ROAD,0.000,
            roadVegSpec1,-1000,0,0));
    habTypes[4] = road;

    std::string nm = "species1";

    SpeciesPtr animal(new Species(nm,false,2.52e-3,0.0928e-3,-2.52e-3,0.1014e-3,
            1.4,0.5,0.7,0.012,2.78,1.39,0.1,true,1000,habTypes));

    std::vector<SpeciesPtr> species(1);
    species[0] = animal;

    // Habitat map and initial populaiton

    // DESIGN PARAMETERS
    DesignParametersPtr desParams(new DesignParameters(100,0,0,8500,3000,
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

    // Initialise species habitat maps and initial populations
    animal->generateHabitatMap(roadGA);

    // Build the initial population map if it doesn't already exist
    std::ifstream initPopFile;
    initPopFile.open("Input_Data/Inputs/Regions/Region_1/initPopFile.csv");

    if (initPopFile.good()) {
        Eigen::MatrixXd IP(Z.rows(),Z.cols());
        for (int ii = 0; ii < Z.rows(); ii++) {
            std::string line;
            std::getline(habFile,line,'\n');
            std::stringstream ss(line);
            for (int jj = 0; jj < Z.cols(); jj++) {
                std::string substr;
                std::getline(ss,substr,',');
                std::stringstream subss(substr);
                double val;
                if (substr != "NaN") {
                    subss >> val;
                    IP(ii,jj) = val;
                } else {
                    IP(ii,jj) = std::numeric_limits<int>::quiet_NaN();
                }
            }
        }
        initPopFile.close();
        animal->setPopulationMap(IP);

    } else {
        initPopFile.close();
        std::ofstream initPopFile;
        initPopFile.open("Input_Data/Inputs/Regions/Region_1/initPopFile.csv",
                std::ios::out);

        animal->initialisePopulationMap(roadGA);

        for (int ii = 0; ii < animal->getHabitatMap().rows(); ii++) {
            for (int jj = 0; jj < animal->getHabitatMap().cols(); jj++) {
                initPopFile << animal->getHabitatMap()(ii,jj);

                if (jj < (animal->getHabitatMap().cols() - 1)) {
                    initPopFile << ",";
                }
            }

            if (ii < (animal->getHabitatMap().rows()-1)) {
                initPopFile << std::endl;
            }
        }
    }
    initPopFile.close();

    // ADD A TEST ROAD
    Eigen::RowVectorXd genome(30);

    genome << 0,0,0,2000,0,50,4000,2000,100,4000,6000,0,1000,7000,0,2000,8000,-50,5000,7000,-150,7000,8000,-50,8000,4000,0,8500,3000,0;

    RoadPtr trialRoad(new Road(roadGA,genome));
    std::cout << "Read in test success" << std::endl;

    trialRoad->designRoad();
    trialRoad->evaluateRoad(false);
}
