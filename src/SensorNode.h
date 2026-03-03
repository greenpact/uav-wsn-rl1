#ifndef SENSOR_NODE_H
#define SENSOR_NODE_H

#include <omnetpp.h>

using namespace omnetpp;

class SensorNode : public cSimpleModule
{
  private:
    double initialX;
    double initialY;
    double range;
    cMessage *sendTimer = nullptr;

  protected:
    virtual void initialize() override;
    virtual void handleMessage(cMessage *msg) override;
    virtual void finish() override;
};

#endif // SENSOR_NODE_H
