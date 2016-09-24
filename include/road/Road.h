#ifndef ROAD_H
#define ROAD_H

class Optimiser;
typedef std::shared_ptr<Optimiser> OptimiserPtr;

class Simulator;
typedef std::shared_ptr<Simulator> SimulatorPtr;

class PolicyMap;
typedef std::shared_ptr<PolicyMap> PolicyMapPtr;

class RoadSegments;
typedef std::shared_ptr<RoadSegments> RoadSegmentsPtr;

class RoadCells;
typedef std::shared_ptr<RoadCells> RoadCellsPtr;

class HorizontalAlignment;
typedef std::shared_ptr<HorizontalAlignment> HorizontalAlignmentPtr;

class VerticalAlignment;
typedef std::shared_ptr<VerticalAlignment> VerticalAlignmentPtr;

class Attributes;
typedef std::shared_ptr<Attributes> AttributesPtr;

class Costs;
typedef std::shared_ptr<Costs> CostsPtr;

/**
 * Class for managing %Road objects
 */
class Road : public std::enable_shared_from_this<Road> {

public:
	// CONSTRUCTORS AND DESTRUCTORS ///////////////////////////////////////////

	/**
	 * Constructor I
	 *
	 * Constructs an empty %Road object
	 */
	Road();

	/**
	 * Constructor II
	 *
	 * Constructs a %Road object with assigned values
	 */
	Road(OptimiserPtr op, SimulatorPtr sim, std::string testName,
			Eigen::VectorXd* xCoords, Eigen::VectorXd* yCoords,
			Eigen::VectorXd* zCoords);

	/**
	 * Destructor
	 */
    ~Road();

	// ACCESSORS //////////////////////////////////////////////////////////////
	
	/**
	 * Returns the Optimiser object
	 *
	 * @return Optimiser routine as OptimiserPtr
	 */
	OptimiserPtr getOptimiser() {
		return this->optimiser;
	}
	/**
	 * Sets the Optimiser object
	 *
	 * @param op as OptimiserPtr
	 */
	void setOptimiser(OptimiserPtr op) {
		this->optimiser = op;
	}

	/**
	 * Returns the Simulator object
	 *
	 * @return Simulator as SimulatorPtr
	 */
	SimulatorPtr getSimulator() {
		return this->simulator;
	}
	/**
	 * Sets the Simulator object
	 *
	 * @param sim as SimulatorPtr
	 */
	void setSimulator(SimulatorPtr sim) {
		this->simulator = sim;
	}

	/**
	 * Returns the PolicyMap for ROV
	 *
	 * @return PolicyMap as PolicyMapPtr
	 */
	PolicyMapPtr getPolicyMap() {
		return this->policyMap;
	}
	/**
	 * Sets the PolicyMap for ROV
	 *
	 * @param pm as PolicyMapPtr
	 */
	void setPolicyMap(PolicyMapPtr pm) {
		this->policyMap = pm;
	}

	/**
	 * Returns the road segments from start to end
	 *
	 * @return Road segments as RoadSegmentsPtr
	 */
	RoadSegmentsPtr getRoadSegments() {
		return this->segments;
	}
	/**
	 * Sets the road segments from start to end
	 *
	 * @param segments as RoadSegmentsPtr
	 */
	void setRoadSegments(RoadSegmentsPtr segments) {
		this->segments = segments;
	}

	/**
	 * Returns the road cells from start to end
	 *
	 * @return Road cells as RoadCellsPtr
	 */
	RoadCellsPtr getRoadCells() {
		return this->roadCells;
	}
	/**
	 * Sets the road cells from start to end
	 *
	 * @param cells as RoadCellsPtr
	 */
	void setRoadCells(RoadCellsPtr cells) {
		this->roadCells = cells;
	}

	/**
	 * Returns the HorizontalAlignment
	 *
	 * @return Horizontal alignment as HorizontalAlignmentPtr
	 */
	HorizontalAlignmentPtr getHorizontalAlignment() {
		return this->horizontalAlignment;
	}
	/**
	 * Sets the HorizontalAlignment
	 *
	 * @param ha as HorizontalAlignmentPtr
	 */
	void setHorizontalAlignment(HorizontalAlignmentPtr ha) {
		this->horizontalAlignment = ha;
	}

	/**
	 * Returns the VerticalAlignment
	 *
	 * @return VerticalAlignment as VerticalAlignmentPtr
	 */
	VerticalAlignmentPtr getVerticalAlignment() {
		return this->verticalAlignment;
	}
	/**
	 * Sets the VerticalAlignment
	 *
	 * @param va as VerticalAlignmentPtr
	 */
	void setVerticalAlignment(VerticalAlignmentPtr va) {
		this->verticalAlignment = va;
	}

	/**
	 * Returns the computed road Attributes
	 *
	 * @return Attributes as AttributesPtr
	 */
	AttributesPtr getAttributes() {
		return this->attributes;
	}
	/**
	 * Sets the computed road Attributes
	 *
	 * @param att as AttributesPtr
	 */
	void setAttributes(AttributesPtr att) {
		this->attributes = att;
	}

	/**
	 * Returns the the computed costs
	 *
	 * @return Computes costs as CostsPtr
	 */
	CostsPtr getCosts() {
		return this->costs;
	}
	/**
	 * Sets the computed costs
	 *
	 * @param costs as CostsPtr
	 */
	void setCosts(CostsPtr costs) {
		this->costs = costs;
	}

	/**
	 * Returns the test name
	 *
	 * @return Test name as std::string
	 */
	std::string getTestName() {
		return this->testName;
	}
	/**
	 * Sets the test name
	 *
	 * @param tn as std:;string
	 */
	void setTestName(std::string tn) {
		this->testName = tn;
	}

	/**
	 * Returns X coordinates of the intersection points
	 *
	 * @return X coordinates as Eigen::VectorXd*
	 */
	Eigen::VectorXd* getXCoords() {
		return &this->xCoords;
	}
	/**
	 * Sets the X coordinates of the intersection points
	 *
	 * @param xc as Eigen::VectorXd*
	 */
	void setXCoords(Eigen::VectorXd* xc) {
		this->xCoords = *xc;
	}

	/**
	 * Returns the Y coordinates of the intersection points
	 *
	 * @return Y coordinates as Eigen::VectorXd*
	 */
	Eigen::VectorXd* getYCoords() {
		return &this->yCoords;
	}
	/**
	 * Sets the Y coordinates of the intersection points
	 *
	 * @param yc as Eigen::VectorXd*
	 */
	void setYCoordinates(Eigen::VectorXd* yc) {
		this->yCoords = *yc;
	}

	/**
	 * Returns the Z coordinates of the intersection points
	 *
	 * @return Z coordinates of the the intersection points
	 */
	Eigen::VectorXd* getZCoords() {
		return &this->zCoords;
	}
	/**
	 * Sets the Z coordinates of the intersection points
	 *
	 * @param zc as Eigen::VectorXd*
	 */
	void setZCoords(Eigen::VectorXd* zc) {
		this->zCoords = *zc;
	}

	// STATIC ROUTINES ////////////////////////////////////////////////////////

	// CALCULATION ROUTINES ///////////////////////////////////////////////////

    /**
     * Design and initialise a road.
     *
     * This routine computes all aspects of a road except operating value
     */
    void designRoad();

    /**
     * Evaluate a road.
     *
     * Evaluates a road given a particular lifetime operating Program
     */
    void evaluateRoad();

    /**
     * Computes the operating costs and animal movement and mortality model
     *
     * Depending on the optimiser options, this call could compute any
     * of the following:
     * 1. Nothing. A simple area-based penalty is applied at an earlier
     *    stage.
     * 2. Full traffic flow for entire horizon.
     * 3. Traffic control.
     */
    void computeOperating();

private:
	OptimiserPtr optimiser;						/**< Calling Optimisation object */
	SimulatorPtr simulator;						/**< Simulator used to produce results */
	PolicyMapPtr policyMap;						/**< PolicyMap generated from ROV simulation */
	RoadSegmentsPtr segments;					/**< Road segments */
	RoadCellsPtr roadCells;						/**< Cells occupied by road */
	HorizontalAlignmentPtr horizontalAlignment;	/**< HorizontalAlignment */
	VerticalAlignmentPtr verticalAlignment;		/**< VerticalAlignment */
	AttributesPtr attributes;					/**< Attributes */
	CostsPtr costs;								/**< Costs */
	std::string testName;						/**< Name of test */
	Eigen::VectorXd xCoords;					/**< X coordinates of intersection points */
	Eigen::VectorXd yCoords;					/**< Y coordinates of intersection points */
	Eigen::VectorXd zCoords;					/**< Z coordinates of intersection points */
	RoadPtr me();								/**< Enables sharing from within Road class */

	// PRIVATE ROUTINES ///////////////////////////////////////////////////////

    /**
     * Builds the road alignment using the points of intersection. In order,
     * this routine calls:
     * 1. horizontalAlignment
     * 2. verticalAlignment
     * 3. plotRoadPath
     * 4. placeNetwork
     */
    void computeAlignment();

    /**
     * Computes the road cost elements required for valuation: In order, this
     * routine calls:
     * 1. earthworkCost
     * 2. locationCosts
     * 3. lengthCosts
     * 4. accidentCosts
     */
    void computeCostElements();

    /**
     * Computes the road value with the assigned optimisation routine
     */
    //void computeValue();

    /**
     * Computes the grid cells that are occupied by the road
     */
    void computeRoadCells();

	/**
	 * Computes the horizontal alignment
	 */
	void computeHorizontalAlignment();

	/**
	 * Computes the vertical alignment (requires the horizontal alignment to
	 * have already been computed).
	 */
	void computeVerticalAlignment();

	/**
	 * Computes the road path from the horizontal and vertical alignments
	 */
	void plotRoadPath();
};

#endif
