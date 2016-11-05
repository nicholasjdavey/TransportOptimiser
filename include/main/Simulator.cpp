#include "../transportbase.h"

Simulator::Simulator() : MonteCarloROV() {
}

Simulator::Simulator(RoadPtr road) {
    this->road = road;
}

Simulator::~Simulator() {
}

void Simulator::simulateMTE() {

}
