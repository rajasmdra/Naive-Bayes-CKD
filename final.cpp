#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <iomanip>
#include <algorithm>
#include <cmath>
using namespace std;

bool missingValue(vector<string> row) {
    for (auto value : row) {
        if (value == "") return true;
    }
    return false;
}

double toNumeric(string s) {
    if (s == "normal" || s == "present" || s == "yes" || s == "good" || s == "ckd") return 1;
    if (s == "abnormal" || s == "notpresent" || s == "no" || s == "poor" || s == "notckd") return 0;

    return atof(s.c_str());
}

int main() {
    ifstream file("kidney_disease.csv");
    vector<vector<string>> cleanedData;
    vector<string> fitur;
    string line;
    bool isHeader = true;
    int data = 0;
    int menu;
    
    // DATA CLEANING
    while (getline(file, line)) {
        stringstream ss(line);
        string value;
        vector<string> row;

        if (isHeader) {
            while (getline(ss, value, ',')) fitur.push_back(value);
            isHeader = false;
            continue;           
        }

        while (getline(ss, value, ',')) row.push_back(value);
        if (!missingValue(row)) cleanedData.push_back(row);
        data++;
    }
    
    file.close();

    // DATA TRANSFORMATION
    vector<vector<double>> transformedData;
    vector<int> label;

    for (int i = 0; i < cleanedData.size(); i++) {
        vector<double> row;

        for (int j = 0; j < fitur.size() - 1; j++) {
            row.push_back(toNumeric(cleanedData[i][j]));
        }

        transformedData.push_back(row);
        label.push_back(toNumeric(cleanedData[i][fitur.size() - 1]));
    }

    // SPLIT DATA
    int n = transformedData.size();
    int trainSize = n * 0.8;

    vector<vector<double>> dataTrain(transformedData.begin(), transformedData.begin() + trainSize);
    vector<vector<double>> dataTest(transformedData.begin() + trainSize, transformedData.end());
    vector<double> labelTrain(label.begin(), label.begin() + trainSize);
    vector<double> labelTest(label.begin() + trainSize, label.end());
    
    double dataCKD = count(labelTrain.begin(), labelTrain.end(), 1);
    double dataNOTCKD = count(labelTrain.begin(), labelTrain.end(), 0);
    
    double priorCKD = dataCKD / dataTrain.size();
    double priorNOTCKD = dataNOTCKD / dataTrain.size();

    vector<double> arrYesCKD(fitur.size()), arrNoCKD(fitur.size()), arrYesNOTCKD(fitur.size()), arrNoNOTCKD(fitur.size());
    vector<double> arrMeanCKD(fitur.size()), arrVarCKD(fitur.size()), arrMeanNOTCKD(fitur.size()), arrVarNOTCKD(fitur.size());
    
    // HITUNG PROBABILITAS TIAP FITUR
    for (int i = 0; i < fitur.size(); i++) {
        string h = fitur[i];
    
        if (h == "rbc" || h == "pc" || h == "pcc" || h == "ba" || h == "htn" ||
            h == "dm" || h == "cad" || h == "appet" || h == "pe" || h == "ane") {
            double yesCKD = 0, yesNOTCKD = 0, noCKD = 0, noNOTCKD = 0;
            
            for (int j = 0; j < dataTrain.size(); j++) {
                if (dataTrain[j][i] == 1 && labelTrain[j] == 1) yesCKD++;
                else if (dataTrain[j][i] == 0 && labelTrain[j] == 1) noCKD++;
                else if (dataTrain[j][i] == 1 && labelTrain[j] == 0) yesNOTCKD++;
                else if (dataTrain[j][i] == 0 && labelTrain[j] == 0) noNOTCKD++;
            }

            double PyesCKD = (yesCKD + 1) / (dataCKD + 2);
            double PnoCKD = (noCKD + 1) / (dataCKD + 2);
            double PyesNOTCKD = (yesNOTCKD + 1) / (dataNOTCKD + 2);
            double PnoNOTCKD = (noNOTCKD + 1) / (dataNOTCKD + 2);

            arrYesCKD[i] = PyesCKD;
            arrNoCKD[i] = PnoCKD;
            arrYesNOTCKD[i] = PyesNOTCKD;
            arrNoNOTCKD[i] = PnoNOTCKD;
        }
        else if (h != "id" && h != "classification") {
            double meanCKD, meanNOTCKD;
            double sumCKD = 0, sumNOTCKD = 0;
            double varCKD = 1e-9, varNOTCKD = 1e-9;

            for (int j = 0; j < dataTrain.size(); j++) {
                if (labelTrain[j] == 1) sumCKD += dataTrain[j][i];
                else sumNOTCKD += dataTrain[j][i];
            }
            
            meanCKD = sumCKD / dataCKD;
            meanNOTCKD = sumNOTCKD / dataNOTCKD;
            
            for (int j = 0; j < dataTrain.size(); j++) {
                if (labelTrain[j] == 1) varCKD += pow(dataTrain[j][i] - meanCKD, 2);
                else varNOTCKD += pow(dataTrain[j][i] - meanNOTCKD, 2);
            }

            varCKD /= dataCKD;
            varNOTCKD /= dataNOTCKD;

            arrMeanCKD[i] = meanCKD;
            arrVarCKD[i] = varCKD;
            arrMeanNOTCKD[i] = meanNOTCKD;
            arrVarNOTCKD[i] = varNOTCKD;
        }
    }
    
    vector<double> labelPredict;
    // DATA TESTING
    for (int i = 0; i < dataTest.size(); i++) {
        double likelihoodCKD = 1.0, scoreCKD, posteriorCKD;
        double likelihoodNOTCKD = 1.0, scoreNOTCKD, posteriorNOTCKD;
        double predict;

        for (int j = 0; j < fitur.size(); j++) {
            string h = fitur[j];
    
            if (h == "rbc" || h == "pc" || h == "pcc" || h == "ba" || h == "htn" ||
                h == "dm" || h == "cad" || h == "appet" || h == "pe" || h == "ane") {
                
                if (dataTest[i][j] == 1) {
                    likelihoodCKD *= arrYesCKD[j];
                    likelihoodNOTCKD *= arrYesNOTCKD[j];
                } else if (dataTest[i][j] == 0) {
                    likelihoodCKD *= arrNoCKD[j];
                    likelihoodNOTCKD *= arrNoNOTCKD[j];
                }
            }
            else if (h != "id" && h != "classification") {
                likelihoodCKD *= 1.0 / sqrt(2* 22.0/7 * arrVarCKD[j]) * exp(- pow(dataTest[i][j] - arrMeanCKD[j], 2) / (2*arrVarCKD[j]));
                likelihoodNOTCKD *= 1.0 /sqrt(2* 22.0/7 * arrVarNOTCKD[j]) * exp(- pow(dataTest[i][j] - arrMeanNOTCKD[j], 2) / (2*arrVarNOTCKD[j]));
            }
        }
        
        scoreCKD = priorCKD * likelihoodCKD;
        scoreNOTCKD = priorNOTCKD * likelihoodNOTCKD;
        posteriorCKD = scoreCKD / (scoreCKD + scoreNOTCKD);
        posteriorNOTCKD = scoreNOTCKD / (scoreCKD + scoreNOTCKD);
        
        if (posteriorCKD > posteriorNOTCKD) predict = 1;
        else predict = 0;
        
        labelPredict.push_back(predict);
    }
    
    while (true) {        
        cout << endl << "PROGRAM NAIVE BAYES" << endl;
        cout << "1. Hitung Probabilitas Tiap Fitur" << endl;
        cout << "2. Hitung Prediksi" << endl;
        cout << "3. Hitung Akurasi, Presisi, Recall" << endl;
        cout << "4. Keluar" << endl;
    
        while (true) {
            cout << "Pilih menu: ";
            cin >> menu;
            if (menu >= 1 && menu <= 4) break;
            else cout << "Input tidak valid, silahkan coba lagi." << endl;
        }

        if (menu == 1) {
            cout << endl << "DATA TRAINING\t: " << dataTrain.size() << endl;
            cout << "Data CKD\t: " << dataCKD << endl;
            cout << "Data NOTCKD\t: " << dataNOTCKD << endl;
            cout << endl << "PRIOR KELAS" << endl;
            cout << "Prior CKD\t: " << priorCKD << endl;
            cout << "Prior NOTCKD\t: " << priorNOTCKD << endl;

            cout << endl << "PROBABILITAS TIAP FITUR" << endl;
            for (int i = 0; i < fitur.size(); i++) {
                string h = fitur[i];
                if (h == "rbc" || h == "pc" || h == "pcc" || h == "ba" || h == "htn" ||
                    h == "dm" || h == "cad" || h == "appet" || h == "pe" || h == "ane") {
                    cout << endl << h << endl;
                    cout << defaultfloat;
                    cout << "P(Yes|CKD)\t: " << arrYesCKD[i] << endl;
                    cout << "P(No|CKD)\t: " << arrNoCKD[i] << endl;
                    cout << "P(Yes|NOTCKD)\t: " << arrYesNOTCKD[i] << endl;
                    cout << "P(No|NOTCKD)\t: " << arrNoNOTCKD[i] << endl;
                }
                else if (h != "id" && h != "classification") {
                    cout << endl << h << endl;
                    cout << fixed;
                    cout << "Mean(CKD)\t: " << arrMeanCKD[i] << endl;
                    cout << "Var(CKD)\t: " << arrVarCKD[i] << endl;
                    cout << "Mean(NOTCKD)\t: " << arrMeanNOTCKD[i] << endl;
                    cout << "Var(NOTCKD)\t: " << arrVarNOTCKD[i] << endl;
                }
            }
        }
        else if (menu == 2) {
            vector<double> dataUser(fitur.size());
            double likelihoodCKD = 1.0, scoreCKD, posteriorCKD;
            double likelihoodNOTCKD = 1.0, scoreNOTCKD, posteriorNOTCKD;

            for (int i = 0; i < fitur.size(); i++) {
                string h = fitur[i];

                if (h != "id" && h != "classification") {
                    while (true) {
                        cout << "Masukkan nilai " << h << ": ";
                        cin >> dataUser[i];
                        if (h == "rbc" || h == "pc" || h == "pcc" || h == "ba" || h == "htn" ||
                            h == "dm" || h == "cad" || h == "appet" || h == "pe" || h == "ane") {
                            if (dataUser[i] == 1 || dataUser[i] == 0) {
                                if (dataUser[i] == 1) {
                                    likelihoodCKD *= arrYesCKD[i];
                                    likelihoodNOTCKD *= arrYesNOTCKD[i];
                                } 
                                else if (dataUser[i] == 0) {
                                    likelihoodCKD *= arrNoCKD[i];
                                    likelihoodNOTCKD *= arrNoNOTCKD[i];
                                }
                                break;
                            }
                            else cout << "Input tidak valid, silahkan coba lagi.(Ya = 1, Tidak = 0)" << endl;
                        }
                        else {
                            likelihoodCKD *= 1.0 / sqrt(2* 22.0/7 * arrVarCKD[i]) * exp(- pow(dataUser[i] - arrMeanCKD[i], 2) / (2*arrVarCKD[i]));
                            likelihoodNOTCKD *= 1.0 /sqrt(2* 22.0/7 * arrVarNOTCKD[i]) * exp(- pow(dataUser[i] - arrMeanNOTCKD[i], 2) / (2*arrVarNOTCKD[i]));
                            break;
                        }
                    }
                }
            }

            scoreCKD = priorCKD * likelihoodCKD;
            scoreNOTCKD = priorNOTCKD * likelihoodNOTCKD;
            posteriorCKD = scoreCKD / (scoreCKD + scoreNOTCKD);
            posteriorNOTCKD = scoreNOTCKD / (scoreCKD + scoreNOTCKD);
            
            cout << fixed;
            cout << endl << "Likelihood CKD: " << likelihoodCKD << endl;
            cout << "Likelihood NOTCKD: " << likelihoodNOTCKD << endl;
            cout << "Score CKD: " << scoreCKD << endl;
            cout << "Score NOTCKD: " << scoreNOTCKD << endl;
            cout << "Posterior CKD: " << posteriorCKD << endl;
            cout << "Posterior NOTCKD: " << posteriorNOTCKD << endl;
            if (posteriorCKD > posteriorNOTCKD) cout << "Predict: CKD" << endl;
            else  cout << "Predict: NOTCKD" << endl;
        }
        else if (menu == 3) {
            double truePositive = 0, trueNegative = 0, falsePositive = 0, falseNegative = 0;
            
            for (int i = 0; i < labelTest.size(); i++) {
                if (labelTest[i] == 1 && labelPredict[i] == 1) truePositive++;
                else if (labelTest[i] == 1 && labelPredict[i] == 0) falseNegative++;
                else if (labelTest[i] == 0 && labelPredict[i] == 1) falsePositive++;
                else if (labelTest[i] == 0 && labelPredict[i] == 0) trueNegative++;
            }
            cout << endl << "DATA TESTING\t: " << dataTest.size() << endl;
        
            cout << "CONFUSION MATRIX" << endl;
            cout << left << defaultfloat;
            cout << setw(15) << "" << setw(12) << "Pred Ya" << "Pred Tidak" << endl;
            cout << setw(15) << "Aktual Ya" << setw(12) << truePositive << falseNegative <<endl;
            cout << setw(15) << "Aktual Tidak" << setw(12) << falsePositive << trueNegative << endl;
            
            double accuracy  = double(truePositive + trueNegative) / labelTest.size() * 100.0;
            double precision = (truePositive + falsePositive == 0) ? 0 : double(truePositive) / (truePositive + falsePositive) * 100.0;
            double recall    = (truePositive + falseNegative == 0) ? 0 : double(truePositive) / (truePositive + falseNegative) * 100.0;
            double f1        = (precision + recall == 0) ? 0 : 2 * precision * recall / (precision + recall);
            
            cout << endl << "Accuracy\t: " << accuracy << "%" << endl;
            cout << "Precision\t: " << precision << "%" << endl;
            cout << "Recall\t\t: " << recall << "%" << endl;
            cout << "F1-Score\t: " << f1 << "%" << endl;
        }
        else if (menu == 4) break;
    }

    return 0;
}