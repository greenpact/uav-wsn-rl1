#ifndef RADIO_MEDIUM_H
#define RADIO_MEDIUM_H

#include <omnetpp.h>
#include <vector>

using namespace omnetpp;

class RadioMedium : public cSimpleModule
{
  private:
    int numSensors;
    double areaX;
    double areaY;
    std::vector<std::pair<double,double>> positions;

  protected:
    virtual void initialize() override;
    virtual void handleMessage(cMessage *msg) override;
    virtual void forwardMessage(cMessage *msg, int srcIdx, double sx, double sy, double srange, int txRadio);
    // query stored position for node index (sensors 0..numSensors-1, uav=numSensors, bs=numSensors+1)
    void getPosition(int idx, double &x, double &y) const;
};

#endif // RADIO_MEDIUM_H
