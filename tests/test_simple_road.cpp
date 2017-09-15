#include "../include/transportbase.h"

int main(int argc, char **argv) {

    std::stringstream ss;
    std::string inputFile = argv[1];
    boost::filesystem::path p(inputFile);
    std::string root;
    ss << p.parent_path().c_str();
    ss >> root;

    ss.str(std::string());
    ss.clear();
    std::string experimentName;
    ss << p.parent_path().filename();
    ss >> experimentName;

    RoadGAPtr roadGA(new RoadGA());
    roadGA->setRootFolder(root.c_str());

    ExperimentalScenarioPtr scenario(new ExperimentalScenario(roadGA,0,0,0,0,0,
            0,0,0,0,0,0,0,0));
    scenario->setCurrentScenario(0);

    roadGA->initialiseFromTextInput(inputFile);
    roadGA->setScenario(scenario);
    roadGA->initialiseStorage();
    roadGA->getExistingSurrogateData();
    roadGA->defaultSurrogate();
    Costs::computeUnitRevenue(roadGA);

    // Run design and evaluation for single road
    Eigen::RowVectorXd genome(51);

    genome << 4500,500,401.5176203498,3935.8631961997,525.2558211692,419.8884203769,3592.5792436455,734.0960982778,473.8072628079,3334.1186643945,986.7706651152,491.6322688052,3013.0509286772,1125.2630159614,518.5222083999,2798.9614192409,1240.669665274,517.3042012464,2644.8802002243,1328.6443623846,521.9427709907,2392.5885207296,1472.71237905,536.9086072057,2141.4824945145,1614.7889349374,546.0544796,1807.3003294038,1799.7327726405,579.778640484,1552.2248132383,2230.5135107138,606.8168116121,1473.1507886368,2781.6326496697,577.716556301,1365.3469200481,3046.3025534736,566.1286401587,1164.2730617201,3636.1398083932,568.2038720006,1048.1959656015,4197.0278961376,497.6596113771,650.1690148169,4465.3593067837,498.00058713,500,4500,462.9940424869;

    RoadPtr trialRoad(new Road(roadGA,genome));

    // Using Surrogate
    time_t begin = clock();
    trialRoad->designRoad();
    time_t end = clock();
    double elapsed_secs = double(end - begin) / CLOCKS_PER_SEC;
    std::cout << "Design time " << elapsed_secs << " s" << std::endl;

    // Full model
    begin = clock();
    trialRoad->evaluateRoad(true);
    end = clock();
    elapsed_secs = double(end - begin) / CLOCKS_PER_SEC;
    std::cout << "Total evaluation time " << elapsed_secs << " s" << std::endl;

    //    std::cout << "Press any key to end..." << std::endl;
    //    std::cin.get();
}
