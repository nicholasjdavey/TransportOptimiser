#ifndef SPECIESROADPATCHES_H
#define SPECIESROADPATCHES_H

class Uncertainty;
typedef std::shared_ptr<Uncertainty> UncertaintyPtr;

class Species;
typedef std::shared_ptr<Species> SpeciesPtr;

class Road;
typedef std::shared_ptr<Road> RoadPtr;

class HabitatPatch;
typedef std::shared_ptr<HabitatPatch> HabitatPatchPtr;

/**
 * Class for managing %SpeciesRoadPatches objects
 */
class SpeciesRoadPatches : public Uncertainty,
        public std::enable_shared_from_this<SpeciesRoadPatches> {

public:
    // CONSTRUCTORS AND DESTRUCTORS ///////////////////////////////////////////

    /**
     * Constructor I
     *
     * Constructs a %SpeciesRoadPatches object with default values
     */
    SpeciesRoadPatches(SpeciesPtr species, RoadPtr road);

    /**
     * Constructor II
     *
     * Constructs a %SpeciesRoadPatches object with assigned values
     */
    SpeciesRoadPatches(SpeciesPtr species, RoadPtr road, bool active,
            double mean, double stdDev, double rev, std::string nm);
    /**
     * Destructor
     */
    ~SpeciesRoadPatches();

    // ACCESSORS //////////////////////////////////////////////////////////////

    /**
     * Returns the Species
     *
     * @return Species as SpeciesPtr
     */
    SpeciesPtr getSpecies() {
        return this->species;
    }
    /**
     * Sets the Species
     *
     * @param species as SpeciesPtr
     */
    void setSpecies(SpeciesPtr species) {
        this->species = species;
    }

    /**
     * Returns the Road
     *
     * @return Road as RoadPtr
     */
    RoadPtr getRoad() {
        return this->road;
    }
    /**
     * Sets the Road
     *
     * @param road as RoadPtr
     */
    void setRoad(RoadPtr road){
        this->road = road;
    }

    /**
     * Returns the habitat patches for simulations
     *
     * @return Habitat patches as std::vector<HabitatPatchPtr>*
     */
    std::vector<HabitatPatchPtr>* getHabPatches() {
        return &this->habPatch;
    }
    /**
     * Sets the habitat patches during simulations
     *
     * @param habp as std::vector<HabitatPatchPtr>*
     */
    void setHabPatches(std::vector<HabitatPatchPtr>* habp) {
        this->habPatch = *habp;
    }

    // STATIC ROUTINES ////////////////////////////////////////////////////////

    // CALCULATION ROUTINES ///////////////////////////////////////////////////

    /**
     * Computes the distance between each patch with every other patch.
     *
     * @return Distances as Eigen::MatrixXd
     */
    Eigen::MatrixXd habitatPatchDistances();

    /**
     * Computes the number of road crossings between each patch.
     *
     * @return Number of crossings as Eigen::MatrixXi
     */
    Eigen::MatrixXi roadCrossings();

///////////////////////////////////////////////////////////////////////////////
private:
    SpeciesPtr species;                     /**< Speices used */
    RoadPtr road;                           /**< Corresponding road */
    std::vector<HabitatPatchPtr> habPatch;  /**< Corresponding habitat patches */
};

#endif
