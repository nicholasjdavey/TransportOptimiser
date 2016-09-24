#ifndef VERTICALALIGNMENT_H
#define VERTICALALIGNMENT_H

class Road;
typedef std::shared_ptr<Road> RoadPtr;

class VerticalAlignment;
typedef std::shared_ptr<VerticalAlignment> VerticalAlignmentPtr;

/**
 * Class for managing the vertical alignment of a road design
 */
class VerticalAlignment : public std::enable_shared_from_this<VerticalAlignment> {

public:
	// CONSTRUCTORS AND DESTRUCTORS ///////////////////////////////////////////

	/**
	 * Constructor I
	 *
	 * Constructs a vertical alignment object with default values
	 */
	VerticalAlignment();

	/**
	 * Constructor II
	 *
	 * Constructs a vertical alignment object for a given road and assigns
	 * appropriate space.
	 */
	VerticalAlignment(RoadPtr road);

	/**
	 * Destructor
	 */
	~VerticalAlignment();

	// ACCESSORS //////////////////////////////////////////////////////////////

	/**
	 * Return distances of intersection points along HA
	 *
	 * @return Distances as Eigen::VectorXd*
	 */
	Eigen::VectorXd* getSDistances() {
		return &this->s;
	}
	/**
	 * Sets distances of intersection points along HA
	 *
	 * @param s as Eigen::VectorXd*
	 */
	void setSDistances(Eigen::VectorXd* s) {
		this->s = *s;
	}

	
	/**
	 * Return points of vertical curvature
	 *
	 * @return PVCs as Eigen::VectorXd*
	 */
	Eigen::VectorXd* getPVCs() {
		return &this->pvc;
	}
	/**
	 * Sets points of vertical curvature
	 *
	 * @param pvcs as Eigen::VectorXd*
	 */
	void setPVCs(Eigen::VectorXd* pvcs) {
		this->pvc = *pvcs;
	}

	/**
	 * Return points of vertical tangency
	 *
	 * @return PVTs as Eigen::VectorXd*
	 */
	Eigen::VectorXd* getPVTs() {
		return &this->pvt;
	}
	/**
	 * Sets points of vertical tangency
	 *
	 * @param pvts as Eigen::VectorXd*
	 */
	void setPVTs(Eigen::VectorXd* pvts) {
		this->pvt = *pvts;
	}

	/**
	 * Return elevations of points of vertical curvature
	 *
	 * @return EPVCs as Eigen::VectorXd*
	 */
	Eigen::VectorXd* getEPVCs() {
		return &this->epvc;
	}
	/**
	 * Sets elevations of points of vertical curvature
	 *
	 * @param epvcs as Eigen::VectorXd*
	 */
	void setEPVCs(Eigen::VectorXd* epvcs) {
		this->epvc = *epvcs;
	}

	/**
	 * Return elevations of points of vertical tangency
	 *
	 * @return EPVTs as Eigen::VectorXd*
	 */
	Eigen::VectorXd* getEPVTs() {
		return &this->epvt;
	}
	/**
	 * Sets elevations of points of vertical tangency
	 *
	 * @param epvts as Eigen::VectorXd*
	 */
	void setEPVTs(Eigen::VectorXd* epvts) {
		this->epvt = *epvts;
	}

	/**
	 * Returns polynomial coefficiens (const, x, x^2)
	 *
	 * @return Polynomial coefficients as Eigen::MatrixXd*
	 */
	Eigen::MatrixXd* getPolyCoeffs() {
		return &this->a;
	}
	/**
	 * Sets polynomial coefficients (const, x, x^2)
	 *
	 * @param coeffs as Eigen::MatrixXd*
	 */
	void setPolyCoeffs(Eigen::MatrixXd* coeffs) {
		this->a = *coeffs;
	}

	/**
	 * Return velocities at each IP
	 *
	 * @return Velocities as Eigen::VectorXd*
	 */
	Eigen::VectorXd* getVelocities() {
		return &this->v;
	}
	/**
	 * Sets velocities at each IP
	 *
	 * @param v as Eigen::VectorXd*
	 */
	void setVelocities(Eigen::VectorXd* v) {
		this->v = *v;
	}

	/**
	 * Return lengths of curvature
	 *
	 * @return Lengths of curvature as Eigen::VectorXd*
	 */
	Eigen::VectorXd* getLenghts() {
		return &this->Ls;
	}
	/**
	 * Sets the lengths of curvature
	 *
	 * @param ls as Eigen::VectorXd*
	 */
	void setLengths(Eigen::VectorXd* ls) {
		this->Ls = *ls;
	}

	/**
	 * Return segment grades
	 *
	 * @return Grades as Eigen::VectorXd*
	 */
	Eigen::VectorXd* getGrades() {
		return &this->gr;
	}
	/**
	 * Sets segment grades
	 *
	 * @param gr as Eigen::VectorXd*
	 */
	void setGrades(Eigen::VectorXd* gr) {
		this->gr = *gr;
	}

	/**
	 * Return stopping sight distances
	 *
	 * @return Stopping sight distances as Eigen::VectorXd*
	 */
	Eigen::VectorXd* getSSDs() {
		return &this->ssd;
	}
	/**
	 * Sets stopping sight distances
	 *
	 * @param ssds as Eigen::VectorXd*
	 */
	void setSSDs(Eigen::VectorXd* ssds) {
		this->ssd = *ssds;
	}

	// STATIC ROUTINES ////////////////////////////////////////////////////////

	// CALCULATION ROUTINES ///////////////////////////////////////////////////

	/**
	 * Computes the vertical alignment of a Road
	 */
	void computeAlignment();

private:
	RoadPtr road;							/**< Road */
	Eigen::VectorXd s;						/**< Distance of intersection points along HA */
	Eigen::VectorXd pvc;					/**< S points of vertical curvature */
	Eigen::VectorXd pvt;					/**< S points of vertical tangency */
	Eigen::VectorXd epvc;					/**< Elevations of PVCs */
	Eigen::VectorXd epvt;					/**< Elevations of PVTs */
	Eigen::MatrixXd a;						/**< Polynomial coefficients */
	Eigen::VectorXd v;						/**< Velocities */
	Eigen::VectorXd Ls;						/**< Curvature lengths */
	Eigen::VectorXd gr;						/**< Segment grades */
	Eigen::VectorXd ssd;					/**< Stopping sight distances */

	// PRIVATE ROUTINES ///////////////////////////////////////////////////////
};

#endif
