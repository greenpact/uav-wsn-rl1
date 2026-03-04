#include "SensorNode.h"

Define_Module(SensorNode);

void SensorNode::initialize()
{
    initialX = par("initialX").doubleValue();
    initialY = par("initialY").doubleValue();
    has80211 = par("has80211").boolValue();
    has802154 = par("has802154").boolValue();
    range11 = par("range11").doubleValue();
    range154 = par("range154").doubleValue();

    sendTimer = new cMessage("beaconTimer");
    // randomize start to avoid synchronized beacons
    scheduleAt(simTime() + uniform(0,0.5), sendTimer);
}

void SensorNode::handleMessage(cMessage *msg)
{
    if (msg == sendTimer) {
        // send beacon on available radios; if both available, alternate
        static int toggle = 0;
        if (has802154) {
            cMessage *b = new cMessage("BEACON");
            b->addPar("txX") = initialX;
            b->addPar("txY") = initialY;
            b->addPar("txRange") = range154;
            b->addPar("txRadio") = 154;
            send(b, "out");
        }
        if (has80211) {
            cMessage *b2 = new cMessage("BEACON");
            b2->addPar("txX") = initialX;
            b2->addPar("txY") = initialY;
            b2->addPar("txRange") = range11;
            b2->addPar("txRadio") = 11;
            send(b2, "out");
        }
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
