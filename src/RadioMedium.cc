#include "RadioMedium.h"
#include "SensorNode.h"

using namespace omnetpp;

Define_Module(RadioMedium);

void RadioMedium::initialize()
{
    numSensors = par("numSensors").intValue();
    areaX = par("areaX").doubleValue();
    areaY = par("areaY").doubleValue();
    roundDuration = par("roundDuration").doubleValue();

    // compute and record simple connectivity statistics at init
    cModule *net = getParentModule();
    int totalNodes = numSensors + 2; // sensors + uav + bs
    std::vector<std::pair<double,double>> pos(totalNodes);

    // Read positions from SensorNode submodule parameters (initialX/initialY)
    positions.resize(numSensors + 2);
    for (int i = 0; i < numSensors; ++i) {
        cModule *m = net->getSubmodule("sensors", i);
        double x = m->hasPar("initialX") ? m->par("initialX").doubleValue() : uniform(0, areaX);
        double y = m->hasPar("initialY") ? m->par("initialY").doubleValue() : uniform(0, areaY);
        positions[i] = std::make_pair(x, y);
        pos[i].first = x;
        pos[i].second = y;
    }
    cModule *uav = net->getSubmodule("uav");
    if (uav) {
        double ux = uav->hasPar("initialX") ? uav->par("initialX").doubleValue() : -200;
        double uy = uav->hasPar("initialY") ? uav->par("initialY").doubleValue() : 500;
        positions[numSensors] = std::make_pair(ux, uy);
        pos[numSensors].first = ux;
        pos[numSensors].second = uy;
    }
    cModule *bs = net->getSubmodule("bs");
    if (bs) {
        double bx = bs->hasPar("initialX") ? bs->par("initialX").doubleValue() : -200;
        double by = bs->hasPar("initialY") ? bs->par("initialY").doubleValue() : 500;
        positions[numSensors+1] = std::make_pair(bx, by);
        pos[numSensors+1].first = bx;
        pos[numSensors+1].second = by;
    }

    // simple neighbor counting using sensor range (assume all sensors same range)
    double r = net->getSubmodule("sensors",0)->par("range154").doubleValue();
    long totalNeighbors = 0;
    for (int i = 0; i < totalNodes; ++i) {
        int cnt = 0;
        for (int j = 0; j < totalNodes; ++j) {
            if (i==j) continue;
            double dx = pos[i].first - pos[j].first;
            double dy = pos[i].second - pos[j].second;
            double d = sqrt(dx*dx + dy*dy);
            if (d <= r) cnt++;
        }
        totalNeighbors += cnt;
        recordScalar((std::string("neighbors-")+std::to_string(i)).c_str(), cnt);
    }
    recordScalar("avgNeighbors", (double)totalNeighbors/totalNodes);

    // setup round timer
    roundTimer = new cMessage("roundTimer");
    scheduleAt(simTime() + roundDuration, roundTimer);
}

void RadioMedium::handleMessage(cMessage *msg)
{
    if (msg->isSelfMessage()) {
        if (msg == roundTimer) {
            handleRoundTimer();
        }
        return;
    }

    // arrival gate index corresponds to sender index (sensors: 0..numSensors-1, uav:numSensors, bs:numSensors+1)
    int gateIdx = msg->getArrivalGate()->getIndex();
    cModule *net = getParentModule();

    double sx=0, sy=0, srange=100;
    int txRadio = 154; // default
    int totalNodes = numSensors + 2;

    // use parameters carried in message if present
    if (msg->hasPar("txX")) sx = msg->par("txX").doubleValue();
    if (msg->hasPar("txY")) sy = msg->par("txY").doubleValue();
    if (msg->hasPar("txRange")) srange = msg->par("txRange").doubleValue();
    if (msg->hasPar("txRadio")) txRadio = (int)msg->par("txRadio").longValue();

    forwardMessage(msg, gateIdx, sx, sy, srange, txRadio);
    delete msg;
}

void RadioMedium::forwardMessage(cMessage *msg, int srcIdx, double sx, double sy, double srange, int txRadio)
{
    cModule *net = getParentModule();
    int totalNodes = numSensors + 2;

    for (int j = 0; j < totalNodes; ++j) {
        if (j == srcIdx) continue;
        double tx=0, ty=0;
        // dynamic UAV position lookup for uav index
        if (j == numSensors) {
            cModule *uav = net->getSubmodule("uav");
            if (uav && uav->hasPar("currentX") && uav->hasPar("currentY")) {
                tx = uav->par("currentX").doubleValue();
                ty = uav->par("currentY").doubleValue();
            } else {
                tx = positions[j].first;
                ty = positions[j].second;
            }
        } else if (j == numSensors+1) {
            tx = positions[j].first;
            ty = positions[j].second;
        } else {
            tx = positions[j].first;
            ty = positions[j].second;
        }

        double dx = sx - tx;
        double dy = sy - ty;
        double d = sqrt(dx*dx + dy*dy);
        if (d <= srange) {
            // check if recipient listens to this radio
            cModule *recipient = (j < numSensors) ? net->getSubmodule("sensors", j) : (j==numSensors ? net->getSubmodule("uav") : net->getSubmodule("bs"));
            bool listens = false;
            if (txRadio == 11) {
                if (recipient->hasPar("has80211")) listens = recipient->par("has80211").boolValue();
            }
            else if (txRadio == 154) {
                if (recipient->hasPar("has802154")) listens = recipient->par("has802154").boolValue();
            }
                if (listens) {
                    // increment control packet counter if message tagged as control
                            if (msg->hasPar("isControl") && msg->par("isControl").boolValue()) {
                                controlPacketCount++;
                            }
                            // increment hop count for forwarded messages
                            if (msg->hasPar("hopCount")) {
                                long h = msg->par("hopCount").longValue();
                                // update original so copies reflect increment
                                msg->par("hopCount") = h+1;
                            } else {
                                msg->addPar("hopCount") = 1;
                            }
                            cMessage *copy = msg->dup();
                    // annotate copy with source index so recipients can track neighbors
                    copy->addPar("srcIdx") = srcIdx;
                    send(copy, "out", j);
                }
        }
    }
}

void RadioMedium::handleRoundTimer()
{
    cModule *net = getParentModule();
    double totalEnergy = 0.0;
    for (int i = 0; i < numSensors; ++i) {
        cModule *m = net->getSubmodule("sensors", i);
        if (m) {
            SensorNode *sn = check_and_cast<SensorNode*>(m);
            totalEnergy += sn->getRemainingEnergy();
        }
    }

    recordScalar("roundIndex", roundIndex);
    recordScalar("totalNetworkEnergy", totalEnergy);
    recordScalar("controlPacketCount", controlPacketCount);

    controlPacketCount = 0;
    roundIndex++;
    scheduleAt(simTime() + roundDuration, roundTimer);
}

void RadioMedium::finish()
{
    if (roundTimer) cancelAndDelete(roundTimer);
    roundTimer = nullptr;
}
