#include "../include/transportbase.h"

int main(int argc, char **argv) {

    std::string solScheme = "GA";

    RoadGAPtr roadGA(new RoadGA(0.6,0.375,500,200,1e-6,0.95,0.95,10,100,
            solScheme,5, Optimiser::CONTROLLED,1.0,15,0.05,10,3,10,
            RoadGA::TOURNAMENT,RoadGA::RANK,0.4,0.65,5,0.1,300,true,
            Optimiser::ALGO1,Optimiser::MULTI_LOC_LIN_REG));

    // EXPERIMENTAL SCENARIO
    ExperimentalScenarioPtr scenario(new ExperimentalScenario(roadGA,0,0,0,0,0,
            0,0,0,0,0,0,0,0));

    scenario->setCurrentScenario(0);
    roadGA->setScenario(scenario);

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

    UncertaintyPtr sp1gr(new Uncertainty(roadGA,nm,0.014,0.014,0.005,0.1,0.1,0.1,
        true));

    SpeciesPtr animal(new Species(nm,false,2.52e-3,0.0928e-3,-2.52e-3,
            0.1014e-3,sp1gr,0.5,0.7,0.012,2.78,1.39,1e7,true,1000,1,
            habTypes));

    std::vector<SpeciesPtr> species(1);
    species[0] = animal;

    roadGA->setSpecies(species);
    int samples = 5000;

    // Build some random data to test the interpolation routine
    // We create the 2D function: Z = X^2 + Y^2
    Eigen::MatrixXd X(samples,1);
    Eigen::VectorXd Y(samples);
    Eigen::VectorXd Z(samples);

    // We create 500 data points

    /* initialize random seed: */
    srand (time(NULL));

    for (int ii = 0; ii < samples; ii++) {
        X(ii) = (((double) rand() / (RAND_MAX)))*4;
        Y(ii) = (((double) rand() / (RAND_MAX)))*4;
        Z(ii) = pow(X(ii),4) + pow(Y(ii),4);
    }

    // VARIABLE PARAMETERS
    Eigen::VectorXd popLevels(6);
    popLevels << 0.50,0.60,0.70,0.80,0.90,0.100;
    Eigen::VectorXi bridge(1);
    bridge << 0;
    Eigen::VectorXd hp(1);
    hp << 0;
    Eigen::VectorXd l(3);
    l << -1,0,1;
    Eigen::VectorXd beta(1);
    beta << 0;
    Eigen::VectorXd pgr(3);
    pgr << 0.5,1,1.5;
    Eigen::VectorXd pgrsd(3);
    pgrsd << 1,2,3;
    Eigen::VectorXd c(1);
    c << 100;
    Eigen::VectorXd csd(3);
    csd << 1,2,3;
    Eigen::VectorXd cpsd(3);
    cpsd << 1,2,3;
    Eigen::VectorXd cr(3);
    cr << 0.5,1,1.5;
    VariableParametersPtr varParams(new VariableParameters(popLevels,bridge,
            hp,l,beta,pgr,pgrsd,c,csd,cpsd,cr));

    // DESIGN PARAMETERS
    DesignParametersPtr desParams(new DesignParameters(100/3.6,0,0,9000,9000,
            5,8,15,2.5,3.4,15,20,M_PI/4,M_PI/6,0,0,0,0,0,0,100,0.00003621,
            0.00000355,0.000001562,0.000024282,0.00004331,0.000001704,false));

    roadGA->setDesignParams(desParams);
    roadGA->setVariableParams(varParams);
    roadGA->initialiseStorage();

    // Create surrogate data from this
    roadGA->setValues(Z);
    roadGA->setValuesSD(Z);
    roadGA->setUse(Y);
    roadGA->setIARS(X);
    roadGA->setNoSamples(samples);

    SimulateGPU::buildSurrogateROVCUDA(roadGA);

    // Plot the surrogate
    GnuplotPtr plotPtr(new Gnuplot);

    roadGA->setPlotHandle(plotPtr);

    // Prepare raw data
    std::vector<std::vector<double>> raw;
    raw.resize(samples);

    for (int ii = 0; ii < samples; ii++) {
        raw[ii].resize(3);
        raw[ii][0] = X(ii);
        raw[ii][1] = Y(ii);
        raw[ii][2] = Z(ii);
    }

    // Prepare regressed data
    std::vector<std::vector<std::vector<double>>> reg;
    reg.resize(roadGA->getSurrDimRes());

    for (int ii = 0; ii < roadGA->getSurrDimRes(); ii++) {
        reg[ii].resize(roadGA->getSurrDimRes());
        for (int jj = 0; jj < roadGA->getSurrDimRes(); jj++) {
            reg[ii][jj].resize(3);
        }
    }

    Eigen::VectorXd surrogate = roadGA->getSurrogateML()[0][0][0];

    for (int ii = 0; ii < roadGA->getSurrDimRes(); ii++) {
        for (int jj = 0; jj < roadGA->getSurrDimRes(); jj++) {
            reg[ii][jj][0] = surrogate[ii];
            reg[ii][jj][1] = surrogate[roadGA->getSurrDimRes()+jj];
            reg[ii][jj][2] = surrogate(2*roadGA->getSurrDimRes() + ii + jj*
                    roadGA->getSurrDimRes());
        }
    }

//    // Plot 1 (input data)
////    (*roadGA->getPlotHandle()) << "set multiplot layout 1,2\n";
//    (*roadGA->getPlotHandle()) << "set title 'Multiple regression'\n";
//    (*roadGA->getPlotHandle()) << "set grid\n";
//    (*roadGA->getPlotHandle()) << "set hidden3d\n";
//    (*roadGA->getPlotHandle()) << "unset key\n";
//    (*roadGA->getPlotHandle()) << "unset view\n";
//    (*roadGA->getPlotHandle()) << "unset pm3d\n";
//    (*roadGA->getPlotHandle()) << "unset xlabel\n";
//    (*roadGA->getPlotHandle()) << "unset ylabel\n";
//    (*roadGA->getPlotHandle()) << "set xrange [*:*]\n";
//    (*roadGA->getPlotHandle()) << "set yrange [*:*]\n";
//    (*roadGA->getPlotHandle()) << "set view 45,45\n";
//    (*roadGA->getPlotHandle()) << "splot '-' with points pointtype 7 \n";
//    (*roadGA->getPlotHandle()).send1d(raw);
//    (*roadGA->getPlotHandle()).flush();

    // Plot 2 (Multiple linear regression model)
    (*roadGA->getPlotHandle()) << "set title 'Multiple regression'\n";
    (*roadGA->getPlotHandle()) << "set grid\n";
//    (*roadGA->getPlotHandle()) << "set hidden3d\n";
    (*roadGA->getPlotHandle()) << "unset key\n";
    (*roadGA->getPlotHandle()) << "unset view\n";
    (*roadGA->getPlotHandle()) << "unset pm3d\n";
    (*roadGA->getPlotHandle()) << "unset xlabel\n";
    (*roadGA->getPlotHandle()) << "unset ylabel\n";
    (*roadGA->getPlotHandle()) << "set xrange [*:*]\n";
    (*roadGA->getPlotHandle()) << "set yrange [*:*]\n";
    (*roadGA->getPlotHandle()) << "set view 45,45\n";
//    (*roadGA->getPlotHandle()) << "splot '-' with lines,\n";
    (*roadGA->getPlotHandle()) << "splot '-' with lines, '-' with points pointtype 7\n";
    (*roadGA->getPlotHandle()).send2d(reg);
    (*roadGA->getPlotHandle()).send1d(raw);
    (*roadGA->getPlotHandle()).flush();

    std::cout << "Printed" << std::endl;
}
