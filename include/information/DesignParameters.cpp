#include "../transportbase.h"

DesignParameters::DesignParameters() {
}

DesignParameters::DesignParameters(double desVel, double sx, double sy,
        double ex, double ey, double mg, double mse, double rw, double rt,
        double dr, unsigned long ip, double sl, double cr, double fr,
        double bf, double bw, double bh, double tf, double tw, double td,
        double cpsm, double air, double noise, double water, double oil,
        double land, double chem, bool sp) {

    // Initialise values
    this->designVel = desVel;
    this->startX = sx;
    this->startY = sy;
    this->endX = ex;
    this->endY = ey;
    this->maxGrade = mg;
    this->maxSE = mse;
    this->roadWidth = rw;
    this->reactionTime = rt;
    this->deccelRate = dr;
    this->intersectPoints = ip;
    this->segmentLength = sl;
    this->cutRep = cr;
    this->fillRep = fr;
    this->bridgeFixed = bf;
    this->bridgeWidth = bw;
    this->bridgeHeight = bh;
    this->tunnelFixed = tf;
    this->tunnelWidth = tw;
    this->tunnelDepth = td;
    this->spiral = sp;
    this->costPerSM = cpsm;
    this->airPollution = air;
    this->noisePollution = noise;
    this->waterPollution = water;
    this->oilExtraction = oil;
    this->landUse = land;
    this->solidChemWaste = chem;
}
