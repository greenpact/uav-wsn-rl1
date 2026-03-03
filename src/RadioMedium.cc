#include "RadioMedium.h"

using namespace omnetpp;

Define_Module(RadioMedium);

void RadioMedium::initialize()
{
    numSensors = par("numSensors").intValue();
    areaX = par("areaX").doubleValue();
    areaY = par("areaY").doubleValue();

    // compute and record simple connectivity statistics at init
    cModule *net = getParentModule();
    int totalNodes = numSensors + 2; // sensors + uav + bs
    std::vector<std::pair<double,double>> pos(totalNodes);

    for (int i = 0; i < numSensors; ++i) {
        cModule *n = net->getSubmodule("sensors", i);
        pos[i].first = n->par("initialX").doubleValue();
        pos[i].second = n->par("initialY").doubleValue();
    }
    cModule *uav = net->getSubmodule("uav");
    pos[numSensors].first = uav->par("initialX").doubleValue();
    pos[numSensors].second = uav->par("initialY").doubleValue();
    cModule *bs = net->getSubmodule("bs");
    pos[numSensors+1].first = bs->par("initialX").doubleValue();
    pos[numSensors+1].second = bs->par("initialY").doubleValue();

    // simple neighbor counting using sensor range (assume all sensors same range)
    double r = net->getSubmodule("sensors",0)->par("range").doubleValue();
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
}

void RadioMedium::handleMessage(cMessage *msg)
{
    // arrival gate index corresponds to sender index (sensors: 0..numSensors-1, uav:numSensors, bs:numSensors+1)
    int gateIdx = msg->getArrivalGate()->getIndex();
    cModule *net = getParentModule();

    double sx=0, sy=0, srange=100;
    int totalNodes = numSensors + 2;

    if (gateIdx < numSensors) {
        cModule *sender = net->getSubmodule("sensors", gateIdx);
        sx = sender->par("initialX").doubleValue();
        sy = sender->par("initialY").doubleValue();
        srange = sender->par("range").doubleValue();
    }
    else if (gateIdx == numSensors) {
        cModule *sender = net->getSubmodule("uav");
        sx = sender->par("initialX").doubleValue();
        sy = sender->par("initialY").doubleValue();
        srange = sender->par("range").doubleValue();
    }
    else {
        cModule *sender = net->getSubmodule("bs");
        sx = sender->par("initialX").doubleValue();
        sy = sender->par("initialY").doubleValue();
        srange = sender->par("range").doubleValue();
    }

    forwardMessage(msg, gateIdx, sx, sy, srange);
    delete msg;
}

void RadioMedium::forwardMessage(cMessage *msg, int srcIdx, double sx, double sy, double srange)
{
    cModule *net = getParentModule();
    int totalNodes = numSensors + 2;

    for (int j = 0; j < totalNodes; ++j) {
        if (j == srcIdx) continue;
        double tx=0, ty=0;
        if (j < numSensors) {
            cModule *n = net->getSubmodule("sensors", j);
            tx = n->par("initialX").doubleValue();
            ty = n->par("initialY").doubleValue();
        }
        else if (j == numSensors) {
            cModule *u = net->getSubmodule("uav");
            tx = u->par("initialX").doubleValue();
            ty = u->par("initialY").doubleValue();
        }
        else {
            cModule *b = net->getSubmodule("bs");
            tx = b->par("initialX").doubleValue();
            ty = b->par("initialY").doubleValue();
        }

        double dx = sx - tx;
        double dy = sy - ty;
        double d = sqrt(dx*dx + dy*dy);
        if (d <= srange) {
            cMessage *copy = msg->dup();
            send(copy, "out", j);
        }
    }
}
