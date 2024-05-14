//
// Created by user on 5. 11. 2017.
//

#ifndef NEURONET_MNS_H
#define NEURONET_MNS_H

#include "../som/MSOM.h"
#include "Dataset.h"

using namespace NeuroNet;

namespace MNS {

class ModelMNS {
public:
    ModelMNS();
    ~ModelMNS();

    void setMotorParams(double p_gamma1, double p_gamma2, double p_alpha, double p_beta);
    void setVisualParams(double p_gamma1, double p_gamma2, double p_alpha, double p_beta);
    void run(int p_epochs);
    void save();
    void load(string p_timestamp);

    void testAllWinners();
    void testFinalWinners();
    void testDistance();
    void testBALData();

private:
    const int _sizePMC = 8; //12
    const int _sizeSTSp = 12; //16
    const int GRASPS = 3;
    const int PERSPS = 4;

    double motor_gamma1;
    double motor_gamma2;
    double motor_alpha;
    double motor_beta;

    double visual_gamma1;
    double visual_gamma2;
    double visual_alpha;
    double visual_beta;

    Dataset _data;
    MSOM    *_msomMotor;
    MSOM    *_msomVisual;
};

}

#endif //NEURONET_MNS_H
