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

  protected:
    virtual void initialize() override;
    virtual void handleMessage(cMessage *msg) override;
    virtual void forwardMessage(cMessage *msg, int srcIdx, double sx, double sy, double srange);
};

#endif // RADIO_MEDIUM_H
