#ifndef SENSOR_NODE_H
#define SENSOR_NODE_H

#include <omnetpp.h>

using namespace omnetpp;

class SensorNode : public cSimpleModule
{
  private:
    double initialX;
    double initialY;
    bool has80211 = false;
    bool has802154 = true;
    double range154 = 100;
    double range11 = 150;
    cMessage *sendTimer = nullptr;
    cMessage *dataTimer = nullptr;
    uint64_t seqCounter = 0;
    bool deathRecorded = false;
    double dataInterval = 1.0;
    double remainingEnergy = 0.0;
    double eElec = 50e-9;
    double eAmp = 10e-12; // free-space amplifier default

  protected:
    virtual void initialize() override;
    virtual void handleMessage(cMessage *msg) override;
    virtual void finish() override;
  public:
    double getRemainingEnergy() const { return remainingEnergy; }
    void consumeEnergyTx(int bits, double d);
    void consumeEnergyRx(int bits);
    void recordNodeDeath();
};

#endif // SENSOR_NODE_H
