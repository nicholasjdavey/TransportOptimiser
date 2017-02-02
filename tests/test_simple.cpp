#include "../include/transportbase.h"

void handler(int sig) {
  void *array[10];
  size_t size;

  // get void*'s for all entries on the stack
  size = backtrace(array, 10);

  // print out all the frames to stderr
  fprintf(stderr, "Error: signal %d:\n", sig);
  backtrace_symbols_fd(array, size, STDERR_FILENO);
  exit(1);
}

int main(int argc, char **argv) {
    signal(SIGSEGV, handler);

    // INITIALISE THE OPTIMISER OBJECT FIRST
    // We will later add the attributes to the Optimiser Object
    std::string solScheme = "GA";

    RoadGAPtr roadGA(new RoadGA(0.6,0.375,500,50,1e-4,0.95,0.95,10,solScheme,5,
            Optimiser::MTE,1.0,15,0.05,10,3,10,RoadGA::TOURNAMENT,
            RoadGA::RANK,0.4,0.65,5,0.1,true));

    // SET THREADER
//    ThreadManagerPtr threader(new ThreadManager(8));
//    roadGA->setThreadManager(threader);

    // Initialise the input classes
    // REGION
    std::string regionData = "Input Data/region.csv";
    RegionPtr region(new Region(regionData,true));

    // Create X and Y matrices
    //Eigen::VectorXd x = Eigen::VectorXd::LinSpaced(201,-1000,15250);
    //Eigen::RowVectorXd y = Eigen::RowVectorXd::LinSpaced(201,-1000,17000);
    Eigen::VectorXd x = Eigen::VectorXd::LinSpaced(201,-500,9500);
    Eigen::RowVectorXd y = Eigen::RowVectorXd::LinSpaced(201,-500,9500);

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

    // Set column-major cell indices for region
    Eigen::MatrixXi idx(X.rows(),X.cols());
    for (int ii = 0; ii < X.cols(); ii++) {
        for (int jj = 0; jj < X.rows(); jj++) {
            idx(jj,ii) = jj + ii*X.rows();
        }
    }

    region->setCellIdx(idx);
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
    HabitatTypePtr primary(new HabitatType(HabitatType::PRIMARY,0.0005,
            primaryVegSpec1,0,0,1e8));
    habTypes[0] = primary;
    // MARGINAL
    Eigen::VectorXi marginalVegSpec1(1);
    marginalVegSpec1 << 3;
    HabitatTypePtr marginal(new HabitatType(HabitatType::MARGINAL,0.00025,
            marginalVegSpec1,-0.262,0.073,5e7));
    habTypes[1] = marginal;
    // OTHER
    Eigen::VectorXi otherVegSpec1(1);
    otherVegSpec1 << 2;
    HabitatTypePtr other(new HabitatType(HabitatType::OTHER,0.0001,
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
            1.4,0.5,0.7,0.012,2.78,1.39,1e7,true,1000,0.7,habTypes));

    std::vector<SpeciesPtr> species(1);
    species[0] = animal;

    // Habitat map and initial populaiton

    // DESIGN PARAMETERS
    DesignParametersPtr desParams(new DesignParameters(100/3.6,0,0,9000,9000,
            5,8,15,2.5,3.4,15,20,M_PI/4,M_PI/6,0,0,0,0,0,0,100,0.00003621,
            0.00000355,0.000001562,0.000024282,0.00004331,0.000001704,false));

    // COMMODITIES
    std::vector<CommodityPtr> fuels(2);
    std::vector<CommodityPtr> commodities(1);
    std::string dieselName = "diesel";
    std::string petrolName = "petrol";
    std::string commodityName = "ore";

    CommodityPtr diesel(new Commodity(roadGA,dieselName,1.2,0.01,0.01,true,0,0));
    diesel->setCurrent(1.2);
    fuels[0] = diesel;
    CommodityPtr petrol(new Commodity(roadGA,petrolName,1.05,0.01,0.01,true,0,0));
    petrol->setCurrent(1.05);
    fuels[1] = petrol;
    CommodityPtr ore(new Commodity(roadGA,commodityName,100,0.1,0.01,true,0.8,0.20));
    ore->setCurrent(120);
    commodities[0] = ore;

    // ECONOMIC
    EconomicPtr economic(new Economic(commodities,fuels,7,50));

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
            1000,20,5000));

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
    hp << 0;
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

    // EXPERIMENTAL SCENARIO
    ExperimentalScenarioPtr scenario(new ExperimentalScenario(roadGA,0,0,0,0,0,
            0,0,0,0,0,0));

    scenario->setCurrentScenario(0);

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
    roadGA->setScenario(scenario);
    // Now that we have all of the contained objects within the roadGA object,
    // we can initialise the storage elements
    roadGA->initialiseStorage();
    roadGA->defaultSurrogate();

    // Initialise species habitat maps and initial populations
    animal->generateHabitatMap(roadGA);

    // Build the initial population map if it doesn't already exist
    std::ifstream initPopFile;
    initPopFile.open("Input_Data/Inputs/Regions/Region_1/initPopFile.csv");

    if (initPopFile.good()) {
        Eigen::MatrixXd IP(Z.rows(),Z.cols());
        for (int ii = 0; ii < Z.rows(); ii++) {
            std::string line;
            std::getline(initPopFile,line,'\n');
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
                initPopFile << animal->getPopulationMap()(ii,jj);

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

    // Find the expected present value of each commodity
    petrol->computeExpPV();
    diesel->computeExpPV();
    ore->computeExpPV();
    Costs::computeUnitRevenue(roadGA);

    roadGA->optimise(true);

    std::cout << "Optimisation Complete" << std::endl;

//    // ADD A TEST ROAD
//    Eigen::RowVectorXd genome(30);

//    genome << 0,0,0,2000,0,50,4000,2000,100,4000,6000,0,1000,7000,0,2000,8000,-50,5000,7000,-150,7000,8000,-50,8000,4000,0,8500,3000,0;
//      genome << 0,0,0,-25.3584483756,1182.5281256729,0,3278.9686508845,3459.0175020783,0,6167.4882811343,3673.8464325079,0,4094.8149288515,6956.6580029934,0,2832.9369896245,1981.134285769,0,8607.9200963527,1043.5284625942,0,8030.9031247305,4049.3167395933,0,7806.6903136702,1242.7526668845,0,9000,9000,0;

//    RoadPtr trialRoad(new Road(roadGA,genome));
//    std::cout << "Read in test success" << std::endl;

//    trialRoad->designRoad();
//    std::cout << "Design success" << std::endl;
//    trialRoad->evaluateRoad(true);
//    std::cout << "Evaluate success" << std::endl;
}
