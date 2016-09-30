#include "../transportbase.h"

Species::Species(std::string* nm, bool sex, double lm, double lsd, double rcm,
        double rcsd, double grm, double grsd, double lenm, double lensd,
        double spm, double spsd, double cpa, bool active,
        std::vector<HabitatTypePtr>* habitat) {

	// Initialise object values
	this->setName(*nm);
    this->sex = sex;
	this->lambdaMean = lm;
	this->lambdaSD = lsd;
	this->rangingCoeffMean = rcm;
	this->rangingCoeffSD = rcsd;
	this->growthRateMean = grm;
	this->growthRateSD = grsd;
	this->lengthMean = lenm;
	this->lengthSD = lensd;
	this->speedMean = spm;
	this->speedSD = spsd;
    this->costPerAnimal = cpa;
    this->setActive(active);
	this->habitat = *habitat;
}

Species::Species(std::string* nm, bool sex, double lm, double lsd, double rcm,
        double rcsd, double grm, double grsd, double lenm, double lensd,
        double spm, double spsd, double cpa, bool active,
        std::vector<HabitatTypePtr>* habitat, double current) {

	// Initialise object values
	this->setName(*nm);
    this->sex = sex;
	this->lambdaMean = lm;
	this->lambdaSD = lsd;
	this->rangingCoeffMean = rcm;
	this->rangingCoeffSD = rcsd;
	this->growthRateMean = grm;
	this->growthRateSD = grsd;
	this->lengthMean = lenm;
	this->lengthSD = lensd;
	this->speedMean = spm;
	this->speedSD = spsd;
    this->costPerAnimal = cpa;
    this->setActive(active);
	this->habitat = *habitat;
}

SpeciesPtr Species::me() {
    return shared_from_this();
}

void Species::generateHabitatMap(OptimiserPtr optimiser) {
    Eigen::MatrixXi* veg = optimiser->getRegion()->getVegetation();

    this->habitatMap = Eigen::MatrixXi::Zero(veg->rows(),veg->cols());
    std::vector<HabitatTypePtr>* habTypes = this->getHabitatTypes();

    for (int ii = 0; ii < habTypes->size(); ii++) {
        Eigen::VectorXi* vegNos = (*habTypes)[ii]->getVegetations();
        for (int jj; jj < vegNos->size(); jj++) {
            this->habitatMap += ((veg->array() == (*vegNos)(jj))*
                    (int)((*habTypes)[ii]->getType())).cast<int>().matrix();
        }
    }
}

void Species::generateHabitatPatchesGrid(RoadPtr road) {
    // First initialise the number of habitat patches. We expect there to be no
    // more than n x y where n is the number of habitat patches and y is the
    // number of grid cells.
    RegionPtr region = road->getOptimiser()->getRegion();
    Eigen::MatrixXd* X = region->getX();
    Eigen::MatrixXd* Y = region->getY();
    std::vector<HabitatTypePtr>* habTyps = this->getHabitatTypes();
    int res = road->getOptimiser()->getGridRes();

    Eigen::VectorXd xspacing = (X->block(1,0,X->rows()-1,1)
            - X->block(0,0,X->rows()-1,1)).transpose();
    Eigen::VectorXd yspacing = Y->block(0,1,1,Y->cols()-1)
            - Y->block(0,0,1,Y->cols()-1);

    // Grid will be evenly spaced upon call
    if ((xspacing.segment(1,xspacing.size()-1)
            - xspacing.segment(0,xspacing.size()-1)).sum() > 1e-4 ||
            (yspacing.segment(1,yspacing.size()-1)
            - yspacing.segment(0,yspacing.size()-1)).sum() > 1e-4) {
        throw std::invalid_argument("Grid must be evenly spaced in both X and Y");
    }

    Eigen::MatrixXi modHab = (*this->getHabitatMap());
    Eigen::MatrixXi tempHabVec = Eigen::MatrixXi::Constant(1,
            road->getRoadCells()->getUniqueCells()->size(),
            (int)(HabitatType::ROAD));
    igl::slice_into(tempHabVec,*road->getRoadCells()->getUniqueCells(),modHab);

    // We create bins for each habitat type into which we place the patches. We
    // ignore CLEAR and ROAD habitats, hence -2
    unsigned short W = (X->rows());
    unsigned short H = (Y->cols());
    unsigned short xRes = W % res == 0 ? res : res + 1;
    unsigned short yRes = H % res == 0 ? res : res + 1;

    // Number of cells in each coarse grid cell used for creating habitat
    // patches (a sub patch)
    int skpx = std::floor((W-1)/(double)res);
    int skpy = std::floor((H-1)/(double)res);

    std::vector<HabitatPatchPtr> habPatch(pow(res,2)*habTyps->size());
    // Sub patch area
    double subPatchArea = xspacing(0)*yspacing(0);

    int iterator = 0;
    int patches = 0;
    iterator++;

    // Get number of animals that need to be relocated (animals in road cells)
    double relocateAnimals = ((modHab.array() == (int)(HabitatType::ROAD))
            .cast<double>()*this->populationMap.array()).sum();
    double totalPop = (this->populationMap).sum();
    // Factor by which to increase each population
    double factor = totalPop/(totalPop - relocateAnimals);

    for (int ii = 0; ii < habTyps->size(); ii++) {
        if ((*habTyps)[ii]->getType() == HabitatType::ROAD ||
                (*habTyps)[iterator]->getType() == HabitatType::CLEAR) {
            continue;
        }
        Eigen::MatrixXi input = (modHab.array() ==
                (int)((*habTyps)[iterator]->getType())).cast<int>();
        // Map the input array to a plain integer C-array
        int* cinput;
        Eigen::Map<Eigen::MatrixXi>(cinput,W,H) = input;

        // Map output C-array to an Eigen int matrix
        int* coutput = (int*) malloc(W*H*sizeof(int));
        memset(coutput,0,W*H*sizeof(int));
        Eigen::MatrixXi output = Eigen::Map<Eigen::MatrixXi>(coutput,W,H);

        // Separate contiguous regions of this habitat type
        int regions = LabelImage(W,H,cinput,coutput);

        // Iterate through every large cell present in the overall region
        for (int jj = 0; jj < xRes; jj++) {
            for (int kk = 0; kk < yRes; kk++) {
                Eigen::MatrixXi tempGrid = Eigen::MatrixXi::Zero(W,H);

                int blockSizeX;
                int blockSizeY;
                if (jj*skpx <= W) {
                    blockSizeX = skpx;
                } else {
                    blockSizeX = W-jj*skpx;
                }

                if (kk*skpy <= H) {
                    blockSizeY = skpy;
                } else {
                    blockSizeY = H-kk*skpy;
                }

                tempGrid.block(jj*skpx,kk*skpy,blockSizeX,blockSizeY) =
                        Eigen::MatrixXi::Constant(skpx,skpy,1);

                // For every valid habitat type, we must create a new animal
                // patch. If the patch contains valid habitat, we add it to
                // our list of patches for use later.
                for (int ll = 1; ll <= regions; ll++) {
                    Eigen::MatrixXi tempGrid2;
                    tempGrid2 = ((output.array() == ll).cast<int>()
                            *tempGrid.array()).matrix();

                    // If the patch contains this habitat, we continue
                    int noCells = tempGrid2.sum();
                    if (noCells > 0) {
                        HabitatPatchPtr hab(new HabitatPatch());
                        hab->setArea((double)(tempGrid2.sum()*subPatchArea));
                        Eigen::VectorXi xidx =
                                Eigen::VectorXi::LinSpaced(W,1,W);
                        Eigen::VectorXi yidx =
                                Eigen::VectorXi::LinSpaced(H,1,H);

                        hab->setCX((double)(xspacing(0)*((xidx.transpose()*
                                tempGrid2).sum())/noCells + (*X)(0,0)
                                - xspacing(0)));
                        hab->setCY((double)(yspacing(0)*(tempGrid2*yidx).sum()/
                                noCells + (*Y)(0,0) - yspacing(0)));

                        hab->setType((*habTyps)[ii]);
                        double thisPop = (tempGrid2.array().cast<double>()*
                                (this->populationMap).array()).sum()*factor;
                        hab->setPopulation(thisPop);
                        // For now do not store the indices of the points
                        habPatch[patches] = hab;
                        patches++;
                        // Find distance to road here?
                    }
                }
            }
        }
        iterator++;
    }

    // Remove excess patches in container
    habPatch.resize(patches);
    // Save container to road
    SpeciesRoadPatchesPtr srp(new SpeciesRoadPatches(this->me(),road));
    srp->setHabPatches(&habPatch);
    road->addSpeciesPatches(srp);
}
