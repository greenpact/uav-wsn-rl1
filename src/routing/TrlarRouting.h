#ifndef TRLAR_ROUTING_H
#define TRLAR_ROUTING_H

#include <omnetpp.h>

#include <deque>
#include <fstream>
#include <string>
#include <unordered_map>
#include <vector>

#include "../SensorNode.h"
#include "RLAgentInterface.h"

using namespace omnetpp;

namespace trlar {

class TrlarRouting : public cSimpleModule
{
  private:
    struct NeighborInfo {
        int id = -1;
        simtime_t lastSeen = SIMTIME_ZERO;
        double linkQuality = 0.0;
        double energyRatio = 1.0;
        bool isUav = false;
        bool hasPosition = false;
        double x = 0.0;
        double y = 0.0;
        int successCount = 0;
        int failureCount = 0;
        double lastDelay = 0.0;
    };

    struct PendingTx {
        std::string packetId;
        int nextHop = -1;
        double txDistance = 0.0;
        simtime_t sentAt = SIMTIME_ZERO;
        int retries = 0;
        cMessage *backup = nullptr;
    };

    int nodeId = -1;
    int numNodes = 0;
    int uavIndex = -1;
    int bsIndex = -1;

    int bufferSize = 32;
    int maxCandidates = 5;
    int maxRetries = 1;
    int maxRouteAttempts = 6;
    int maxHops = 30;
    int dataPacketBits = 2000;
    int controlPacketBits = 200;

    double commRange = 100.0;
    double selfX = 0.0;
    double selfY = 0.0;
    double initialEnergy = 0.5;

    double minLinkQuality = 0.25;
    double minNeighborEnergyRatio = 0.05;
    double adaptationAlpha = 0.2;
    double queueHighWatermark = 0.8;
    double congestionPenalty = 0.15;
    double delayThreshold = 0.25;

    double rewardW1 = 1.0;
    double rewardW2 = 1.0;
    double rewardW3 = 0.2;
    double rewardW4 = 1.0;

    simtime_t ackTimeout = 0.6;
    simtime_t neighborTimeout = 5.0;
    simtime_t decisionInterval = 0.02;
    simtime_t uavEtaEstimate = 30.0;
    simtime_t lastUavBeacon = -1;

    long queuedDrops = 0;
    long forwardedPackets = 0;
    long successfulForwards = 0;
    long failedForwards = 0;

    std::deque<cMessage *> queue;
    std::unordered_map<int, NeighborInfo> neighbors;
    std::unordered_map<int, double> biasTable;

    PendingTx pending;
    bool inFlight = false;

    std::string transitionsPath;
    std::ofstream transitionsStream;

    SensorNode *sensorNode = nullptr;
    RLAgentInterface agent;

    cMessage *decisionEvent = nullptr;
    cMessage *ackTimeoutEvent = nullptr;

  protected:
    virtual void initialize() override;
    virtual void handleMessage(cMessage *msg) override;
    virtual void finish() override;

  private:
    void handleSelfMessage(cMessage *msg);
    void handleFromSensor(cMessage *msg);
    void handleFromMedium(cMessage *msg);

    void enqueueData(cMessage *msg, bool fromRelay);
    void tryForwardHeadPacket();
    std::vector<int> buildCandidateSet() const;
    std::vector<double> buildActionFeatures(const NeighborInfo& neighbor) const;

    void updateNeighborFromBeacon(int srcIdx, cMessage *msg);
    void updateNeighborFromData(int srcIdx, cMessage *msg);
    void sendAck(const std::string& packetId, int dstIdx);

    void completePendingTx(bool success, bool timely);
    void resetPendingState();

    double computeDistance(double x1, double y1, double x2, double y2) const;
    double estimateLinkQuality(double txX, double txY, double txRange, double prevLq) const;
    double energyRatio() const;
    double queueRatio() const;
    double neighborDensityRatio() const;
    double normalizedTau() const;
    double neighborFailureRatio(const NeighborInfo& neighbor) const;
    std::string packetIdFor(cMessage *msg) const;
    void applyCongestionPenalty();

    void openTransitionsFile();
    void logTransition(const std::string& event, const std::string& payload);
};

} // namespace trlar

#endif // TRLAR_ROUTING_H
