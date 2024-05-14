#include <iostream>
#include "som/MSOM.h"
#include "mns/ModelMNS.h"

using namespace NeuroNet;
using namespace MNS;

const int DIM_X = 16;
const int DIM_Y = 16;
const double ALPHA = 0.3;
const double BETA = 0.5;
const double GAMMA1 = 0.001;
const double GAMMA2 = 0.001;


ModelMNS save();

std::vector<double> alphas = {
        0.2, 0.2, 0.2, 0.2, 0.2,
        0.3, 0.3, 0.3, 0.3, 0.3,
        0.4, 0.4, 0.4, 0.4, 0.4,
        0.5, 0.5, 0.5, 0.5, 0.5,
        0.6, 0.6, 0.6, 0.6, 0.6,
        0.7, 0.7, 0.7, 0.7, 0.7,
        0.8, 0.8, 0.8, 0.8, 0.8
};

std::vector<double> betas = {
        0.4, 0.5, 0.6, 0.7, 0.8,
        0.4, 0.5, 0.6, 0.7, 0.8,
        0.4, 0.5, 0.6, 0.7, 0.8,
        0.4, 0.5, 0.6, 0.7, 0.8,
        0.4, 0.5, 0.6, 0.7, 0.8,
        0.4, 0.5, 0.6, 0.7, 0.8,
        0.4, 0.5, 0.6, 0.7, 0.8
};


void find_best_params() {

}
int main() {

    cout << "Begin!" << endl;

    ModelMNS model;
    model.setMotorParams(0.05, 0.05, 0.3, 0.5);
    model.setVisualParams(0.1, 0.1, 0.3 ,0.7);
    model.run(150);
    model.save();
    model.testAllWinners();
    model.testFinalWinners();
    model.testBALData();
    model.testDistance();

    cout << "End!" << endl;
    return 0;
}


