#ifndef NETWORK_H
#define NETWORK_H

class Optimiser;
typedef std::shared_ptr<Optimiser> OptimiserPtr;

class Road;
typedef std::shared_ptr<Road> RoadPtr;

/**
 * Class for managing Road Networks
 */
class Network : public Road {

public:
    // CONSTRUCTORS AND DESTRUCTORS ///////////////////////////////////////////

    Network();

    Network(OptimiserPtr op);

    Network(OptimiserPtr op, std::vector<RoadPtr>& roads);

    // ACCESSORS //////////////////////////////////////////////////////////////

    // STATIC ROUTINES ////////////////////////////////////////////////////////

    // CALCULATION ROUTINES ///////////////////////////////////////////////////

private:
    // PRIVATE ROUTINES ///////////////////////////////////////////////////////
    std::vector<RoadPtr> roads;
    std::vector<Eigen::VectorXd> flowConfigurations;
};

#endif
