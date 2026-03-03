#include "SensorNode.h"

Define_Module(SensorNode);

void SensorNode::initialize()
{
    initialX = par("initialX").doubleValue();
    initialY = par("initialY").doubleValue();
    range = par("range").doubleValue();

    // schedule first beacon shortly after t=0
    sendTimer = new cMessage("beaconTimer");
    scheduleAt(uniform(0,1), sendTimer);
}

void SensorNode::handleMessage(cMessage *msg)
{
    if (msg == sendTimer) {
        cMessage *b = new cMessage("BEACON");
        b->addPar("txX") = initialX;
        b->addPar("txY") = initialY;
        b->addPar("txRange") = range;
        send(b, "out");
        // periodic beacons every 1s
        scheduleAt(simTime() + 1.0, sendTimer);
    }
    else {
        // message arrived from medium
        recordScalar("rx_count", 1);
        delete msg;
    }
}

void SensorNode::finish()
{
    cancelAndDelete(sendTimer);
}
