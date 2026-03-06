#include "RadioMedium.h"
#include "SensorNode.h"

#include <cmath>

Define_Module(RadioMedium);

void RadioMedium::initialize()
{
    numSensors = par("numSensors").intValue();
    areaX = par("areaX").doubleValue();
    areaY = par("areaY").doubleValue();
    roundDuration = par("roundDuration").doubleValue();

    // Compute and record simple connectivity statistics at init.
    cModule *net = getParentModule();
    const int totalNodes = numSensors + 2; // sensors + uav + bs
    std::vector<std::pair<double,double>> pos(totalNodes);

    // Read static positions from module parameters.
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

    const double r = (numSensors > 0)
        ? net->getSubmodule("sensors",0)->par("range154").doubleValue()
        : 100.0;

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

    // Arrival gate index corresponds to sender index:
    // sensors: 0..numSensors-1, uav:numSensors, bs:numSensors+1.
    const int gateIdx = msg->getArrivalGate()->getIndex();

    double defaultX = 0.0;
    double defaultY = 0.0;
    getPosition(gateIdx, defaultX, defaultY);

    double sx = defaultX;
    double sy = defaultY;
    double srange = 100.0;
    int txRadio = 154; // default

    // Prefer sender-provided TX context, fallback to known source position.
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
    const int totalNodes = numSensors + 2;
    const bool hasUnicastDst = msg->hasPar("dstIdx");
    const int dstIdx = hasUnicastDst ? static_cast<int>(msg->par("dstIdx").longValue()) : -1;

    for (int j = 0; j < totalNodes; ++j) {
        if (j == srcIdx)
            continue;
        if (hasUnicastDst && dstIdx >= 0 && j != dstIdx)
            continue;

        double tx = 0.0;
        double ty = 0.0;
        getPosition(j, tx, ty);

        const double dx = sx - tx;
        const double dy = sy - ty;
        const double d = std::sqrt(dx * dx + dy * dy);
        if (d <= srange) {
            cModule *recipient = (j < numSensors)
                ? net->getSubmodule("sensors", j)
                : ((j == numSensors) ? net->getSubmodule("uav") : net->getSubmodule("bs"));
            if (!recipient)
                continue;
            if (!recipientListens(recipient, txRadio))
                continue;

            if (msg->hasPar("isControl") && msg->par("isControl").boolValue())
                controlPacketCount++;

            cMessage *copy = msg->dup();
            const long hop = copy->hasPar("hopCount") ? copy->par("hopCount").longValue() : 0;
            if (copy->hasPar("hopCount"))
                copy->par("hopCount") = hop + 1;
            else
                copy->addPar("hopCount") = hop + 1;

            if (copy->hasPar("srcIdx"))
                copy->par("srcIdx") = srcIdx;
            else
                copy->addPar("srcIdx") = srcIdx;

            send(copy, "out", j);
        }
    }
}

void RadioMedium::getPosition(int idx, double &x, double &y) const
{
    x = 0.0;
    y = 0.0;

    if (idx < 0 || idx >= static_cast<int>(positions.size()))
        return;

    cModule *net = getParentModule();
    if (idx == numSensors) {
        cModule *uav = net->getSubmodule("uav");
        if (uav && uav->hasPar("currentX") && uav->hasPar("currentY")) {
            x = uav->par("currentX").doubleValue();
            y = uav->par("currentY").doubleValue();
            return;
        }
    }

    x = positions[idx].first;
    y = positions[idx].second;
}

bool RadioMedium::recipientListens(cModule *recipient, int txRadio) const
{
    if (!recipient)
        return false;
    if (txRadio == 11)
        return recipient->hasPar("has80211") && recipient->par("has80211").boolValue();
    if (txRadio == 154)
        return recipient->hasPar("has802154") && recipient->par("has802154").boolValue();
    return true;
}

void RadioMedium::handleRoundTimer()
{
    cModule *net = getParentModule();
    double totalEnergy = 0.0;
    for (int i = 0; i < numSensors; ++i) {
        cModule *m = net->getSubmodule("sensors", i);
        if (m) {
            SensorNode *sn = dynamic_cast<SensorNode*>(m);
            if (sn)
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
