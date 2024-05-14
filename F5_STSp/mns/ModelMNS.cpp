#include <fstream>
#include <string>
#include <chrono>
#include "../NetworkUtils.h"
#include "ModelMNS.h"


using namespace MNS;
using namespace std;

ModelMNS::ModelMNS() {
    _msomMotor = nullptr;
    _msomVisual = nullptr;

}

ModelMNS::~ModelMNS() {
    delete _msomMotor;
    delete _msomVisual;
}


void ModelMNS::run(int p_epochs) {
    _data.loadData("C:\\Users\\misov\\CLionProjects\\untitled\\mns\\data\\simulated\\Trajectories.3.vd", "C:\\Users\\misov\\CLionProjects\\untitled\\mns\\data\\simulated\\Trajectories.3.md");
    _msomMotor = new MSOM(10, _sizePMC, _sizePMC, NeuralGroup::EXPONENTIAL);
    _msomVisual = new MSOM(12, _sizeSTSp, _sizeSTSp, NeuralGroup::EXPONENTIAL);
//    _msomMotor->initTraining(0.1, 0.1, 0.3, 0.5, p_epochs);
//    _msomVisual->initTraining(0.1, 0.1, 0.3 ,0.7, p_epochs);

    _msomMotor->initTraining(motor_gamma1, motor_gamma2, motor_alpha, motor_beta, p_epochs);
    _msomVisual->initTraining(visual_gamma1, visual_gamma2, visual_alpha, visual_beta, p_epochs);

    vector<Sequence*>* trainData = nullptr;
    std::vector<double> pmc_q_errors;
    std::vector<double> pmc_win_diffs;
    std::vector<double> stsp_q_errors;
    std::vector<double> stsp_win_diffs;

    for(int t = 0; t < p_epochs; t++) {
        cout << "Epoch " << t << endl;
        trainData = _data.permute();

        auto start = chrono::system_clock::now();
        for(int i = 0; i < trainData->size(); i++) {
            for(int j = 0; j < trainData->at(i)->getMotorData()->size(); j++) {
                _msomMotor->train(trainData->at(i)->getMotorData()->at(j));
            }
            _msomMotor->resetContext();
            for(int p = 0; p < PERSPS; p++) {
                for(int j = 0; j < trainData->at(i)->getVisualData(p)->size(); j++) {
                    _msomVisual->train(trainData->at(i)->getVisualData(p)->at(j));
                }
                _msomVisual->resetContext();
            }
        }
        auto end = chrono::system_clock::now();
        chrono::duration<double> elapsed_seconds = end-start;
        cout << "elapsed time: " << elapsed_seconds.count() << "s\n";
        pmc_q_errors.push_back(_msomMotor->getError());
        pmc_win_diffs.push_back(_msomMotor->getWinnerDifferentiation());
        stsp_q_errors.push_back(_msomVisual->getError());
        stsp_win_diffs.push_back(_msomVisual->getWinnerDifferentiation());
        cout << " PMC qError: " << _msomMotor->getError() << " WD: " << _msomMotor->getWinnerDifferentiation() << endl;
        cout << "STSp qError: " << _msomVisual->getError() << " WD: " << _msomVisual->getWinnerDifferentiation() << endl;
        _msomMotor->paramDecay();
        _msomVisual->paramDecay();
    }
    std::cout << "PMC qErrors: ";
    for (int i = 0; i < pmc_q_errors.size(); i++) {
        std::cout << pmc_q_errors[i] << ", ";
        if ((i + 1) % 6 == 0 && i != 0) {
            std::cout << std::endl;
        }
    }
    std::cout << std::endl;

    std::cout << "PMC WDs: ";
    for (int i = 0; i < pmc_win_diffs.size(); i++) {
        std::cout << pmc_win_diffs[i] << ", ";
        if ((i + 1) % 6 == 0 && i != 0) {
            std::cout << std::endl;
        }
    }
    std::cout << std::endl;

    std::cout << "STSp qErrors: ";
    for (int i = 0; i < stsp_q_errors.size(); i++) {
        std::cout << stsp_q_errors[i] << ", ";
        if ((i + 1) % 6 == 0 && i != 0) {
            std::cout << std::endl;
        }
    }
    std::cout << std::endl;

    std::cout << "STSp WDs: ";
    for (int i = 0; i < stsp_win_diffs.size(); i++) {
        std::cout << stsp_win_diffs[i] << ", ";
        if ((i + 1) % 6 == 0 && i != 0) {
            std::cout << std::endl;
        }
    }
    std::cout << std::endl;

}


void ModelMNS::testAllWinners() {
    const string timestamp = to_string(NetworkUtils::timestamp());

    vector<Sequence*>* trainData = _data.permute();
    int winRatePMC_Motor[_sizePMC * _sizePMC][GRASPS];
    int winRateSTSp_Visual[_sizeSTSp * _sizeSTSp][PERSPS];
    int winRateSTSp_Motor[_sizeSTSp * _sizeSTSp][GRASPS];

    for(int i = 0; i < _sizePMC * _sizePMC; i++) {
        for(int j = 0; j < 3; j++) {
            winRatePMC_Motor[i][j] = 0;
        }
    }

    for(int i = 0; i < _sizeSTSp * _sizeSTSp; i++) {
        for(int j = 0; j < PERSPS; j++) {
            winRateSTSp_Visual[i][j] = 0;
        }
        for(int j = 0; j < GRASPS; j++) {
            winRateSTSp_Motor[i][j] = 0;
        }
    }

    for(int i = 0; i < trainData->size(); i++) {
        for(int j = 0; j < trainData->at(i)->getMotorData()->size(); j++) {
            _msomMotor->activate(trainData->at(i)->getMotorData()->at(j));
            winRatePMC_Motor[_msomMotor->getWinner()][trainData->at(i)->getGrasp() - 1]++;
        }
        _msomMotor->resetContext();
        for(int p = 0; p < PERSPS; p++) {
            for(int j = 0; j < trainData->at(i)->getVisualData(p)->size(); j++) {
                _msomVisual->activate(trainData->at(i)->getVisualData(p)->at(j));
                winRateSTSp_Visual[_msomVisual->getWinner()][p]++;
                winRateSTSp_Motor[_msomVisual->getWinner()][trainData->at(i)->getGrasp() - 1]++;
            }
            _msomVisual->resetContext();
        }
    }

    ofstream motFile(timestamp + "_Apmc.mot");

    if (motFile.is_open()) {
        motFile << _sizePMC << "," << _sizePMC << endl;
        for (int i = 0; i < _sizePMC * _sizePMC; i++) {
            for (int j = 0; j < GRASPS; j++) {
                if (j == GRASPS - 1) {
                    motFile << winRatePMC_Motor[i][j];
                }
                else {
                    motFile << winRatePMC_Motor[i][j] << ",";
                }
            }
            if (i < _sizePMC * _sizePMC - 1) motFile << endl;
        }
    }

    motFile.close();

    ofstream STSvisFile(timestamp + "_Astsp.vis");

    if (STSvisFile.is_open()) {
        STSvisFile << _sizeSTSp << "," << _sizeSTSp << endl;
        for (int i = 0; i < _sizeSTSp * _sizeSTSp; i++) {
            for (int j = 0; j < PERSPS; j++) {
                if (j == PERSPS - 1) {
                    STSvisFile << winRateSTSp_Visual[i][j];
                }
                else {
                    STSvisFile << winRateSTSp_Visual[i][j] << ",";
                }
            }
            if (i < _sizeSTSp * _sizeSTSp - 1) STSvisFile << endl;
        }
    }

    STSvisFile.close();

    ofstream STSmotFile(timestamp + "_Astsp.mot");

    if (STSmotFile.is_open()) {
        STSmotFile << _sizeSTSp << "," << _sizeSTSp << endl;
        for (int i = 0; i < _sizeSTSp * _sizeSTSp; i++) {
            for (int j = 0; j < GRASPS; j++) {
                if (j == GRASPS - 1) {
                    STSmotFile << winRateSTSp_Motor[i][j];
                }
                else {
                    STSmotFile << winRateSTSp_Motor[i][j] << ",";
                }
            }
            if (i < _sizeSTSp * _sizeSTSp - 1) STSmotFile << endl;
        }
    }

    STSmotFile.close();
}

void ModelMNS::testFinalWinners() {
    const string timestamp = to_string(NetworkUtils::timestamp());

    vector<Sequence*>* trainData = _data.permute();
    int winRatePMC_Motor[_sizePMC * _sizePMC][GRASPS];
    int winRateSTSp_Visual[_sizeSTSp * _sizeSTSp][PERSPS];
    int winRateSTSp_Motor[_sizeSTSp * _sizeSTSp][GRASPS];

    for(int i = 0; i < _sizePMC * _sizePMC; i++) {
        for(int j = 0; j < 3; j++) {
            winRatePMC_Motor[i][j] = 0;
        }
    }

    for(int i = 0; i < _sizeSTSp * _sizeSTSp; i++) {
        for(int j = 0; j < PERSPS; j++) {
            winRateSTSp_Visual[i][j] = 0;
        }
        for(int j = 0; j < GRASPS; j++) {
            winRateSTSp_Motor[i][j] = 0;
        }
    }

    for(int i = 0; i < trainData->size(); i++) {
        for(int j = 0; j < trainData->at(i)->getMotorData()->size(); j++) {
            _msomMotor->activate(trainData->at(i)->getMotorData()->at(j));
        }
        winRatePMC_Motor[_msomMotor->getWinner()][trainData->at(i)->getGrasp() - 1]++;
        _msomMotor->resetContext();
        for(int p = 0; p < PERSPS; p++) {
            for(int j = 0; j < trainData->at(i)->getVisualData(p)->size(); j++) {
                _msomVisual->activate(trainData->at(i)->getVisualData(p)->at(j));
            }
            winRateSTSp_Visual[_msomVisual->getWinner()][p]++;
            winRateSTSp_Motor[_msomVisual->getWinner()][trainData->at(i)->getGrasp() - 1]++;
            _msomVisual->resetContext();
        }

    }

    ofstream motFile(timestamp + "_Bpmc.mot");

    if (motFile.is_open()) {
        motFile << _sizePMC << "," << _sizePMC << endl;
        for (int i = 0; i < _sizePMC * _sizePMC; i++) {
            for (int j = 0; j < GRASPS; j++) {
                if (j == GRASPS - 1) {
                    motFile << winRatePMC_Motor[i][j];
                }
                else {
                    motFile << winRatePMC_Motor[i][j] << ",";
                }
            }
            if (i < _sizePMC * _sizePMC - 1) motFile << endl;
        }
    }

    motFile.close();

    ofstream STSvisFile(timestamp + "_Bstsp.vis");

    if (STSvisFile.is_open()) {
        STSvisFile << _sizeSTSp << "," << _sizeSTSp << endl;
        for (int i = 0; i < _sizeSTSp * _sizeSTSp; i++) {
            for (int j = 0; j < PERSPS; j++) {
                if (j == PERSPS - 1) {
                    STSvisFile << winRateSTSp_Visual[i][j];
                }
                else {
                    STSvisFile << winRateSTSp_Visual[i][j] << ",";
                }
            }
            if (i < _sizeSTSp * _sizeSTSp - 1) STSvisFile << endl;
        }
    }

    STSvisFile.close();

    ofstream STSmotFile(timestamp + "_Bstsp.mot");

    if (STSmotFile.is_open()) {
        STSmotFile << _sizeSTSp << "," << _sizeSTSp << endl;
        for (int i = 0; i < _sizeSTSp * _sizeSTSp; i++) {
            for (int j = 0; j < GRASPS; j++) {
                if (j == GRASPS - 1) {
                    STSmotFile << winRateSTSp_Motor[i][j];
                }
                else {
                    STSmotFile << winRateSTSp_Motor[i][j] << ",";
                }
            }
            if (i < _sizeSTSp * _sizeSTSp - 1) STSmotFile << endl;
        }
    }

    STSmotFile.close();
}

void ModelMNS::testDistance() {
    const string timestamp = to_string(NetworkUtils::timestamp());

    vector<Sequence*>* trainData = _data.permute();
    double winRatePMC_Motor[_sizePMC * _sizePMC][GRASPS];
    double winRateSTSp_Visual[_sizeSTSp * _sizeSTSp][PERSPS];
    double winRateSTSp_Motor[_sizeSTSp * _sizeSTSp][GRASPS];

    for(int i = 0; i < _sizePMC * _sizePMC; i++) {
        for(int j = 0; j < 3; j++) {
            winRatePMC_Motor[i][j] = 0;
        }
    }

    for(int i = 0; i < _sizeSTSp * _sizeSTSp; i++) {
        for(int j = 0; j < PERSPS; j++) {
            winRateSTSp_Visual[i][j] = 0;
        }
        for(int j = 0; j < GRASPS; j++) {
            winRateSTSp_Motor[i][j] = 0;
        }
    }

    for(int i = 0; i < trainData->size(); i++) {
        for(int j = 0; j < trainData->at(i)->getMotorData()->size(); j++) {
            _msomMotor->activate(trainData->at(i)->getMotorData()->at(j));
        }
        for(int n = 0; n < _msomMotor->getGroup("lattice")->getDim(); n++) {
            winRatePMC_Motor[n][trainData->at(i)->getGrasp() - 1] += (*_msomMotor->getOutput())[n];
        }

        _msomMotor->resetContext();
        for(int p = 0; p < PERSPS; p++) {
            for(int j = 0; j < trainData->at(i)->getVisualData(p)->size(); j++) {
                _msomVisual->activate(trainData->at(i)->getVisualData(p)->at(j));
            }
            for(int n = 0; n < _msomVisual->getGroup("lattice")->getDim(); n++) {
                winRateSTSp_Visual[n][p] += (*_msomVisual->getOutput())[n];
                winRateSTSp_Motor[n][trainData->at(i)->getGrasp() - 1] += (*_msomVisual->getOutput())[n];
            }
            _msomVisual->resetContext();
        }
    }

    ofstream motFile(timestamp + "_Cpmc.mot");

    if (motFile.is_open()) {
        motFile << _sizePMC << "," << _sizePMC << endl;
        for (int i = 0; i < _sizePMC * _sizePMC; i++) {
            for (int j = 0; j < GRASPS; j++) {
                if (j == GRASPS - 1) {
                    motFile << winRatePMC_Motor[i][j];
                }
                else {
                    motFile << winRatePMC_Motor[i][j] << ",";
                }
            }
            if (i < _sizePMC * _sizePMC - 1) motFile << endl;
        }
    }

    motFile.close();

    ofstream STSvisFile(timestamp + "_Cstsp.vis");

    if (STSvisFile.is_open()) {
        STSvisFile << _sizeSTSp << "," << _sizeSTSp << endl;
        for (int i = 0; i < _sizeSTSp * _sizeSTSp; i++) {
            for (int j = 0; j < PERSPS; j++) {
                if (j == PERSPS - 1) {
                    STSvisFile << winRateSTSp_Visual[i][j];
                }
                else {
                    STSvisFile << winRateSTSp_Visual[i][j] << ",";
                }
            }
            if (i < _sizeSTSp * _sizeSTSp - 1) STSvisFile << endl;
        }
    }

    STSvisFile.close();

    ofstream STSmotFile(timestamp + "_Cstsp.mot");

    if (STSmotFile.is_open()) {
        STSmotFile << _sizeSTSp << "," << _sizeSTSp << endl;
        for (int i = 0; i < _sizeSTSp * _sizeSTSp; i++) {
            for (int j = 0; j < GRASPS; j++) {
                if (j == GRASPS - 1) {
                    STSmotFile << winRateSTSp_Motor[i][j];
                }
                else {
                    STSmotFile << winRateSTSp_Motor[i][j] << ",";
                }
            }
            if (i < _sizeSTSp * _sizeSTSp - 1) STSmotFile << endl;
        }
    }

    STSmotFile.close();
}

void ModelMNS::testBALData() {
    const string timestamp = to_string(NetworkUtils::timestamp());

    vector<Sequence*>* trainData = _data.permute();
    double winRatePMC[trainData->size()][_sizePMC * _sizePMC];
    double winRateSTSp[trainData->size()][_sizeSTSp * _sizeSTSp][PERSPS];

    for(int i = 0; i < trainData->size(); i++) {
        for(int j = 0; j < trainData->at(i)->getMotorData()->size(); j++) {
            _msomMotor->activate(trainData->at(i)->getMotorData()->at(j));
        }
        for(int n = 0; n < _msomMotor->getGroup("lattice")->getDim(); n++) {
            winRatePMC[i][n] = (*_msomMotor->getOutput())[n];
        }

        _msomMotor->resetContext();
        for(int p = 0; p < PERSPS; p++) {
            for(int j = 0; j < trainData->at(i)->getVisualData(p)->size(); j++) {
                _msomVisual->activate(trainData->at(i)->getVisualData(p)->at(j));
            }
            for(int n = 0; n < _msomVisual->getGroup("lattice")->getDim(); n++) {
                winRateSTSp[i][n][p] = (*_msomVisual->getOutput())[n];
            }
            _msomVisual->resetContext();
        }
    }

    ofstream motFile(timestamp + "_pmc.act");

    if (motFile.is_open()) {
        motFile << _sizePMC << "," << _sizePMC << endl;
        for(int i = 0; i < trainData->size(); i++) {
            motFile << i << "," << trainData->at(i)->getGrasp() << ",";
            for (int j = 0; j < _sizePMC * _sizePMC; j++) {
                if (j == _sizePMC * _sizePMC - 1) {
                    motFile << winRatePMC[i][j] << endl;
                }
                else {
                    motFile << winRatePMC[i][j] << ",";
                }
            }
        }
    }

    motFile.close();

    ofstream STSvisFile(timestamp + "_stsp.act");

    if (STSvisFile.is_open()) {
        STSvisFile << _sizeSTSp << "," << _sizeSTSp << endl;

        for(int i = 0; i < trainData->size(); i++) {
            for(int p = 0; p < PERSPS; p++) {
                STSvisFile << i << "," << trainData->at(i)->getGrasp() << "," << p << ",";
                for (int j = 0; j < _sizeSTSp * _sizeSTSp; j++) {
                    if (j == _sizeSTSp * _sizeSTSp - 1) {
                        STSvisFile << winRateSTSp[i][j][p] << endl;
                    }
                    else {
                        STSvisFile << winRateSTSp[i][j][p] << ",";
                    }
                }
            }
        }
    }

    STSvisFile.close();
}

void ModelMNS::save() {
    const string timestamp = to_string(NetworkUtils::timestamp());

    NetworkUtils::saveNetwork(timestamp + "_pmc.json", _msomMotor);
    NetworkUtils::saveNetwork(timestamp + "_stsp.json", _msomVisual);
}


void ModelMNS::load(string p_timestamp) {
    _msomMotor = (MSOM*)NetworkUtils::loadNetwork(p_timestamp + "_pmc.json");
    _msomVisual = (MSOM*)NetworkUtils::loadNetwork(p_timestamp + "_stsp.json");
}

void ModelMNS::setMotorParams(double p_gamma1, double p_gamma2, double p_alpha, double p_beta) {
    this->motor_gamma1 = p_gamma1;
    this->motor_gamma2 = p_gamma2;
    this->motor_alpha = p_alpha;
    this->motor_beta = p_beta;


}
void ModelMNS::setVisualParams(double p_gamma1, double p_gamma2, double p_alpha, double p_beta) {
    this->visual_gamma1 = p_gamma1;
    this->visual_gamma2 = p_gamma2;
    this->visual_alpha = p_alpha;
    this->visual_beta = p_beta;
}

