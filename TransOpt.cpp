#include "include/transportbase.h"

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

    roadGA->initialiseFromTextInput(inputFile);

    // Run all of the experiments
    int programs = roadGA->getPrograms().size();
    int popLevels = roadGA->getVariableParams()->getPopulationLevels().size();
    int habPrefs = roadGA->getVariableParams()->getHabPref().size();
    int lambda = roadGA->getVariableParams()->getLambda().size();
    int beta = roadGA->getVariableParams()->getBeta().size();
    int ab = roadGA->getVariableParams()->getAnimalBridge().size();
    int grm = roadGA->getVariableParams()->getGrowthRatesMultipliers().size();
    int grsdm = roadGA->getVariableParams()->getGrowthRateSDMultipliers().size();
    int cm = roadGA->getVariableParams()->getCommodityMultipliers().size();
    int csdm = roadGA->getVariableParams()->getCommoditySDMultipliers().size();
    int ore = roadGA->getVariableParams()->getCommodityPropSD().size();
    int runs = roadGA->getNoRuns();

    for (int ii = 0; ii < programs; ii++) {
        for (int jj = 0; jj < popLevels; jj++) {
            for (int kk = 0; kk < habPrefs; kk++) {
                for (int ll = 0; ll < lambda; ll++) {
                    for (int mm = 0; mm < beta; mm++) {
                        for (int nn = 0; nn < ab; nn++) {
                            for (int oo = 0; oo < grm; oo++) {
                                for (int pp = 0; pp < grsdm; pp++) {
                                    for (int qq = 0; qq < cm; qq++) {
                                        for (int rr = 0; rr < csdm; rr++) {
                                            for (int ss = 0; ss < ore; ss++) {
                                                for (int tt = 0; tt < runs; tt++) {
                                                    ExperimentalScenarioPtr scenario(new ExperimentalScenario(roadGA,ii,jj,
                                                            kk,ll,mm,nn,oo,pp,qq,rr,ss,tt));

                                                    scenario->setCurrentScenario(ii*popLevels*habPrefs*lambda*beta*ab*grm*
                                                            grsdm*cm*csdm*ore*runs + jj*habPrefs*lambda*beta*ab*grm*grsdm*
                                                            cm*csdm*ore*runs + kk*lambda*beta*ab*grm*grsdm*cm*csdm*ore*runs
                                                            + ll*beta*ab*grm*grsdm*cm*csdm*ore*runs + mm*ab*grm*grsdm*cm*
                                                            csdm*ore*runs + nn*grm*grsdm*cm*csdm*ore*runs + oo*grsdm*cm*
                                                            csdm*ore*runs + pp*cm*csdm*ore*runs + qq*csdm*ore*runs + rr*ore
                                                            *runs + ss*runs + tt);

                                                    roadGA->setScenario(scenario);
                                                    roadGA->initialiseStorage();
                                                    roadGA->getExistingSurrogateData();
                                                    // Optimise
                                                    roadGA->optimise(true);
                                                }
                                            }
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
    }

    //    std::cout << "Press any key to end..." << std::endl;
    //    std::cin.get();
}
