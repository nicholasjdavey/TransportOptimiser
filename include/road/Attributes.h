#ifndef ATTRIBUTES_H
#define ATTRIBUTES_H

class Road;
typedef std::shared_ptr<Road> RoadPtr;

class Attributes;
typedef std::shared_ptr<Attributes> AttributesPtr;

/**
 * Class for managing road attributes
 */
class Attributes : public std::enable_shared_from_this<Attributes> {

public:
    // CONSTRUCTORS AND DESTRUCTORS ///////////////////////////////////////////

    /**
     * Constructor I
     *
     * Constructs an empty %Attributes object
     */
    Attributes(RoadPtr road);

    /**
     * Constructor II
     *
     * Constructs an %Attributes object with assigned values
     */
    Attributes(double uvc, double uvr,
            double length, double vpic, double tvm, double tvsd, double
            turov, double turovsd, RoadPtr road);

    /**
     * Destructor
     */
    ~Attributes();

    // ACCESSORS //////////////////////////////////////////////////////////////

    /**
     * Returns the road
     *
     * @return Road as RoadPtr
     */
    RoadPtr getRoad() {
    return this->road.lock();
    }
    /**
     * Sets the road
     *
     * @param road as RoadPtr
     */
    void setRoad(RoadPtr road) {
    this->road.reset();
            this->road = road;
    }

    /**
     * Returns the initial animals at risk
     *
     * @return IAR as Eigen::VectorXd&
     */
    Eigen::VectorXd& getIAR() {
            return this->initAAR;
    }
    /**
     * Sets the initial animals at risk
     *
     * @param iar as Eigen::VectorXd&
     */
    void setIAR(Eigen::VectorXd& iar) {
            this->initAAR = iar;
    }

    /**
     * Returns the end population at full traffic flow
     *
     * @return End population as Eigen::VectorXd&
     */
    Eigen::VectorXd& getEndPopMTE() {
            return this->endPopMTE;
    }
    /**
     * Sets the end population at full traffic flow
     *
     * @param endpop as Eigen::VectorXd&
     */
    void setEndPopMTE(Eigen::VectorXd& endpop) {
            this->endPopMTE = endpop;
    }

    /**
     * Returns the end population under controlled flow
     *
     * @return End population as Eigen::VectorXd&
     */
    Eigen::VectorXd& getEndPopROV() {
        return this->endPopROV;
    }
    /**
     * Sets the end population under controlled flow
     *
     * @param endpop as Eigen::VectorXd&
     */
    void setEndPopROV(Eigen::VectorXd& endpop) {
        this->endPopROV = endpop;
    }

    /**
     * Returns the road fixed costs
     *
     * @return Fixed costs as double
     */
    double getFixedCosts() {
        return this->fixedCosts;
    }
    /**
     * Sets the road fixed costs
     *
     * @param fc as double
     */
    void setFixedCosts(double fc) {
        this->fixedCosts = fc;
    }

    /**
     * Returns the unit variable costs
     *
     * @return Unit variable costs as double
     */
    double getUnitVarCosts() {
            return this->unitVarCosts;
    }
    /**
     * Sets the unit variable costs
     *
     * @param uvc as double
     * @note This does not include fuel, which is stochastic
     */
    void setUnitVarCosts(double uvc) {
            this->unitVarCosts = uvc;
    }

    /**
     * Returns the unit variable revenue
     *
     * @return Unit variable revenue as double
     * @note Currently unused as revenue can be stochastic
     */
    double getUnitVarRevenue() {
            return this->unitVarRevenue;
    }
    /**
     * Sets the unit variable revenue
     *
     * @param uvr as double
     */
    void setUnitVarRevenue(double uvr) {
            this->unitVarRevenue = uvr;
    }

    /**
     * Returns the road length
     *
     * @return Length as double
     */
    double getLength() {
            return this->length;
    }
    /**
     * Sets the road length
     *
     * @param len as double
     */
    void setLength(double len) {
            this->length = len;
    }

    /**
     * Returns the variable profit using ROV
     *
     * @return ROV profit as double
     */
    double getVarProfitIC() {
            return this->varProfitIC;
    }
    /**
     * Sets the variable profit using ROV
     *
     * @param vpic as double
     */
    void setVarProfitIC(double vpic) {
            this->varProfitIC = vpic;
    }

    /**
     * Returns the total value
     *
     * @return Total value as double
     */
    double getTotalValueMean() {
            return this->totalValueMean;
    }
    /**
     * Sets the total value
     *
     * @param tvm as double
     */
    void setTotalValueMean(double tvm) {
            this->totalValueMean = tvm;
    }

    /**
     * Returns the total value standard deviation
     *
     * @return Total value standard deviation as double
     */
    double getTotalValueSD() {
        return this->totalValueSD;
    }
    /**
     * Sets the total value standard deviation
     *
     * @param tvsd;
     */
    void setTotalValueSD(double tvsd) {
        this->totalValueSD = tvsd;
    }

    /**
     * Returns the total utilisation with ROV
     *
     * @return Total utilisation as double
     */
    double getTotalUtilisationROV() {
            return this->totalUtilisationROV;
    }
    /**
     * Sets the total utilisation with ROV
     *
     * @param turov as double
     */
    void setTotalUtilisationROV(double turov) {
            this->totalUtilisationROV = turov;
    }

    /**
     * Returns the total utilisation standard deviation with ROV
     *
     * @return Total utilisation standard deviation as double
     */
    double getTotalUtilisationROVSD() {
        return this->totalUtilisationROVSD;
    }
    /**
     * Sets the total utilisation standard deviation with ROV
     *
     * @param turovsd;
     */
    void setTotalUtilisationROVSD(double turovsd) {
        this->totalUtilisationROVSD = turovsd;
    }

    // STATIC ROUTINES ////////////////////////////////////////////////////////

    // CALCULATION ROUTINES ///////////////////////////////////////////////////
	
private:
    std::weak_ptr<Road> road;       /**< Road with these attributes */
    Eigen::VectorXd initAAR;        /**< Initial animals at risk */
    Eigen::VectorXd endPopMTE;      /**< End population if constant full flow */
    Eigen::VectorXd endPopROV;      /**< Mean end population */
    double fixedCosts;              /**< Fixed road costs */
    double unitVarCosts;            /**< Variable costs per unit traffic per hour per year (excl. stochastic factors) */
    double unitVarRevenue;          /**< Variable revenue per unit traffic per hour per year (unused) */
    double length;                  /**< Total road length (m) */
    double varProfitIC;             /**< Operating value */
    double totalValueMean;          /**< Overall value mean */
    double totalValueSD;            /**< Overall value standard deviation */
    double totalUtilisationROV;     /**< Overall utilisation mean, ROV */
    double totalUtilisationROVSD;   /**< Overall utilisation standard deviation, ROV */
};

#endif
