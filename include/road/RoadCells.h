#ifndef ROADCELLS_H
#define ROADCELLS_H

class RoadCells;
typedef std::shared_ptr<RoadCells> RoadCellsPtr;

/**
 * Class for managing the grid cells occupied by a road
 */
class RoadCells : public std::enable_shared_from_this<RoadCells> {

public:

	// CONSTRUCTORS AND DESTRUCTORS ///////////////////////////////////////////
	
	/**
	 * Constructs and object with the details of the cells occupied by a road.
	 */
	RoadCells(RoadPtr road);

	/**
	 * Destructor
	 */
	~RoadCells();
	
	// ACCESSORS //////////////////////////////////////////////////////////////

	/**
	 * Returns the x values
	 *
	 * @return X values as *Eigen::VectorXd
	 */
	Eigen::VectorXd* getX() {
		return &this->x;
	}
	/**
	 * Sets the x values of the segments
	 *
	 * @param x as *Eigen::VectorXd
	 */
	void setX(Eigen::VectorXd *x) {
		this->x = *x;
	}

	/**
	 * Returns the y values
	 *
	 * @return Y values as *Eigen::VectorXd
	 */
	Eigen::VectorXd* getY() {
		return &this->y;
	}
	/**
	 * Sets the y values of the segments
	 *
	 * @param y as *Eigen::VectorXd
	 */
	void setY(Eigen::VectorXd* y) {
		this->y = *y;
	}

	/**
	 * Returns the z values
	 *
	 * @return Z values as *Eigen::VectorXd
	 */
	Eigen::VectorXd* getZ() {
		return &this->z;
	}
	/**
	 * Sets the z values of the segments
	 *
	 * @param z as *Eigen::VectorXd
	 */
	void setZ(Eigen::VectorXd* z) {
		this->z = *z;
	}

	/**
     * Returns the vegetation at each point
	 *
     * @return Vegetation as *Eigen::VectorXi
	 */
    Eigen::VectorXi* getVegetation() {
        return &this->veg;
	}
	/**
     * Sets the vegetation at each point
	 *
     * @param veg as *Eigen::VectorXi
	 */
    void setVegetation(Eigen::VectorXi* veg) {
        this->veg = *veg;
	}

	/**
	 * Returns the segement lengths
	 *
	 * @return Segment lengths as *Eigen::VectorXd
	 */
	Eigen::VectorXd* getLengths() {
		return &this->len;
	}
	/**
	 * Sets the segment lengths
	 *
	 * @param len as *Eigen::VectorXd
	 */
	void setLengths(Eigen::VectorXd* len) {
		this->len = *len;
	}

	/**
	 * Returns the road segment areas
	 *
	 * @return Areas as *Eigen::VectorXd
	 */
	Eigen::VectorXd* getAreas() {
		return &this->areas;
	}
	/**
	 * Sets the road semgnet areas
	 *
	 * @param areas as *Eigen::VectorXd
	 */
	void setAreas(Eigen::VectorXd* areas) {
		this->areas = *areas;
	}

	/**
	 * Returns the road semgnet types
	 *
	 * @return Tyeps as *Eigen::VectorXd
	 */
	Eigen::VectorXi* getTypes() {
		return &this->type;
	}
	/**
	 * Sets the types of the segments
	 *
	 * @param type as *Eigen::VectorXd
	 */
	void setTypes(Eigen::VectorXi* type) {
		this->type = *type;
	}

	/**
	 * Returns the cell references
	 *
	 * Cells are referenced in (xcoord, ycoord), starting at coordinate (0,0)
	 *
     * @return Cell coordinates as *Eigen::VectorXi
	 */
    Eigen::VectorXi* getCellRefs() {
		return &this->cellRefs;
	}
	/**
	 * Sets the cell references
	 *
     * Cells are referenced as indices in column-major format
	 *
     * @param cellrefs as *Eigen::VectorXi
	 */
    void setCellRefs(Eigen::VectorXi* cellrefs) {
		this->cellRefs = *cellrefs;
	}

	/**
	 * Returns the unique cells occupied by the road
	 *
     * @return Cells as Eigen::VectorXi*
	 */
    Eigen::VectorXi* getUniqueCells() {
		return &this->uniqueCells;
	}
	/**
	 * Sets the unique cells occupied by the road
	 *
     * @param cells as Eigen::VectorXi*
	 */
    void setUniqueCells(Eigen::VectorXi* cells) {
		this->uniqueCells = *cells;
    }

	// STATIC ROUTINES ////////////////////////////////////////////////////////

	// CALCULATION ROUTINES ///////////////////////////////////////////////////

	/**
	 * Computes the grid cells to which the road belongs
	 *
	 * The RoadCells object contains
	 *
	 * X positions
	 * Y positions
	 * Z positions
	 * Length of segments
	 * Area of segments
	 * Type (road, bridge, tunnel) of segments
	 * Cell references (x,y positions)
	 * List of unique cells
	 */
	void computeRoadCells();

private:
	RoadPtr road;						/**< Road */
	Eigen::VectorXd x;					/**< X values */
	Eigen::VectorXd y;					/**< Y values */
	Eigen::VectorXd z;					/**< Z values */
    Eigen::VectorXd w;                  /**< Segment widths */
    Eigen::VectorXi veg;				/**< Habitat type */
	Eigen::VectorXd len;				/**< Length of each section */
	Eigen::VectorXd areas;				/**< Area of each section */
	Eigen::VectorXi type;				/**< Type of road section */
    Eigen::VectorXi cellRefs;			/**< Corresponding cell references (column-major indices) */
    Eigen::VectorXi uniqueCells;		/**< Same as above, duplicates removed */
};

#endif

