#ifndef UAV_MODULE_H
#define UAV_MODULE_H

#include <omnetpp.h>

#include <deque>
#include <string>
#include <vector>

using namespace omnetpp;

namespace uav {

class UavModule : public cSimpleModule
{
  private:
    int numNodes = 0;
    int bsIndex = -1;

    double initialX = -200.0;
    double initialY = 500.0;
    double currentX = -200.0;
    double currentY = 500.0;
    double bsX = -200.0;
    double bsY = 500.0;

    double areaX = 1000.0;
    double areaY = 1000.0;
    double commRadius = 192.0;
    double uploadRadius = 280.0;
    double beaconInterval = 5.0;
    double positionUpdateInterval = 0.5;
    double uploadInterval = 0.5;
    double minSpeed = 10.0;
    double maxSpeed = 10.0;
    double missionPeriod = 280.0;
    double missionDuration = -1.0;
    double contactWindow = 8.0;

    // Guided Gauss-Markov mobility kernel.
    double gmAlpha = 0.8;
    double gmSigmaV = 0.45;
    double gmSigmaD = 0.18;
    double gmHeading = 0.0;
    double gmSpeed = 10.0;
    double sensorRange154 = 100.0;
    double maxAreaDistance = 1.0;

    struct SensorSnapshot {
      bool known = false;
      double x = 0.0;
      double y = 0.0;
      double energyRatio = 1.0;
      int degree = 0;
      simtime_t lastSeen = SIMTIME_ZERO;
    };
    std::vector<SensorSnapshot> sensorState;

    long long activeRound = -1;
    double roundEntryY = 500.0;
    double roundExitY = 500.0;
    double roundCurvePhase = 0.0;
    double roundCurveMix = 0.0;
    double roundHoldAtBs = 8.0;
    double roundTravelIn = 40.0;
    double roundSurvey = 180.0;
    double roundTravelOut = 40.0;

    cMessage *moveEvent = nullptr;
    cMessage *beaconEvent = nullptr;
    cMessage *uploadEvent = nullptr;

    std::deque<cMessage *> collected;

    long collectedPackets = 0;
    long uploadedPackets = 0;

  protected:
    virtual void initialize() override;
    virtual void handleMessage(cMessage *msg) override;
    virtual void finish() override;

  private:
    void updatePosition();
    void broadcastBeacon();
    void flushCollectedToBs();
    void sendAckToSource(const std::string& packetId, int dstIdx);
    void ensureRoundPlan(long long roundIndex);
    static double unitNoise(long long roundIndex, int salt);
    void updateGuidedGaussMarkov(double dt);
    int selectGuidanceTarget() const;
    void reflectAtBoundary(double& x, double& y, double& heading) const;

    double currentSpeed() const;
    bool inBsRegion() const;
    double distanceTo(double x, double y) const;
};

} // namespace uav

#endif // UAV_MODULE_H
