#include "include/transportbase.h"

int main(int argc, char **argv) {

    OptimiserPtr opt(new Optimiser());

    Eigen::RowVectorXd genome(30);

    genome << 0,0,0,2000,0,50,4000,2000,100,4000,6000,0,1000,7000,0,2000,8000,-50,5000,7000,-150,7000,8000,-50,8000,4000,0,8500,3000,0;

    RoadPtr trialRoad(new Road(opt,genome));
}
