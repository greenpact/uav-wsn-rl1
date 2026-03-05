#include "SensorNode.h"
#include <fstream>
#include <filesystem>

Define_Module(SensorNode);

void SensorNode::initialize()
{
    initialX = par("initialX").doubleValue();
    initialY = par("initialY").doubleValue();
    has80211 = par("has80211").boolValue();
    has802154 = par("has802154").boolValue();
    range11 = par("range11").doubleValue();
    range154 = par("range154").doubleValue();

    // energy params
    remainingEnergy = par("initialEnergy").doubleValue();
    eElec = par("eElec").doubleValue();
    eAmp = par("eFreeSpace").doubleValue();

    sendTimer = new cMessage("beaconTimer");
    // beacon interval parameter (default 1.0)
    double interval = par("beaconInterval").doubleValue();
    if (interval <= 0) interval = 1.0;
    // randomize start to avoid synchronized beacons
    scheduleAt(simTime() + uniform(0, std::min(0.5, interval)), sendTimer);

    // data generation timer
    dataInterval = par("dataInterval").doubleValue();
    if (dataInterval <= 0) dataInterval = 1.0;
    dataTimer = new cMessage("dataTimer");
    scheduleAt(simTime() + uniform(0, std::min(0.5, dataInterval)), dataTimer);
}

void SensorNode::handleMessage(cMessage *msg)
{
    if (msg == sendTimer) {
        // send beacon on available radios; if both available, alternate
        static int toggle = 0;
        if (has802154) {
            int bits = (int)par("controlPacketSize").doubleValue();
            cMessage *b = new cMessage("BEACON");
            b->addPar("txX") = initialX;
            b->addPar("txY") = initialY;
            b->addPar("txRange") = range154;
            b->addPar("txRadio") = 154;
            b->addPar("isControl") = true;
            // consume energy for transmit
            consumeEnergyTx(bits, range154);
            send(b, "out");
        }
        if (has80211) {
            int bits = (int)par("controlPacketSize").doubleValue();
            cMessage *b2 = new cMessage("BEACON");
            b2->addPar("txX") = initialX;
            b2->addPar("txY") = initialY;
            b2->addPar("txRange") = range11;
            b2->addPar("txRadio") = 11;
            b2->addPar("isControl") = true;
            consumeEnergyTx(bits, range11);
            send(b2, "out");
        }
        double interval = par("beaconInterval").doubleValue();
        if (interval <= 0) interval = 1.0;
        scheduleAt(simTime() + interval, sendTimer);
    }
    else if (msg == dataTimer) {
        // generate a data packet and send to routing
        cMessage *d = new cMessage("DATA");
        // annotate with generation time, sequence and source id
        d->addPar("t_gen") = simTime().dbl();
        d->addPar("seq") = (long)seqCounter++;
        int idx = getIndex();
        d->addPar("srcIdx") = idx;
        // radio params
        d->addPar("txX") = initialX;
        d->addPar("txY") = initialY;
        d->addPar("txRange") = range154;
        d->addPar("txRadio") = 154;
        d->addPar("isControl") = false;
        d->addPar("hopCount") = 0;
        // consume tx energy for generation (approx)
        int bits = (int)par("dataPacketSize").doubleValue();
        consumeEnergyTx(bits, range154);
        recordScalar("dataGenerated", 1);
        // write to CSV generated log
        try {
            std::filesystem::path p("/home/wte/uavwsn-rl/rl-1/uav-wsn-rl1/results/packet_generated.csv");
            if (!std::filesystem::exists(p)) {
                std::ofstream h(p.string(), std::ios::app);
                h << "t,src,seq,t_gen" << std::endl;
                h.close();
            }
            std::ofstream ofs(p.string(), std::ios::app);
            if (ofs.is_open()) {
                ofs << simTime().dbl() << "," << idx << "," << (seqCounter-1) << "," << simTime().dbl() << "\n";
                ofs.close();
            }
        } catch(...) {}
        double interval = dataInterval;
        scheduleAt(simTime() + interval, dataTimer);
        // send into routing via upward gate (same as beacon send)
        send(d, "out");
    }
    else {
        // message arrived from medium
        // receiving consumes energy
        int bits = (int)par("dataPacketSize").doubleValue();
        consumeEnergyRx(bits);
        recordScalar("rx_count", 1);
        // if this is a data packet and this module is the base station (bs), log delivery
        if (strcmp(msg->getName(), "DATA") == 0) {
            // compute e2e delay if t_gen present
            if (msg->hasPar("t_gen")) {
                double tgen = msg->par("t_gen").doubleValue();
                double delay = simTime().dbl() - tgen;
                recordScalar("e2eDelay", delay);
                // write per-delivery CSV
                try {
                    std::filesystem::path p("/home/wte/uavwsn-rl/rl-1/uav-wsn-rl1/results/packet_delivered.csv");
                    if (!std::filesystem::exists(p)) {
                        std::ofstream h(p.string(), std::ios::app);
                        h << "t,src,seq,t_gen,t_recv,hopCount,delay" << std::endl;
                        h.close();
                    }
                    std::ofstream ofs(p.string(), std::ios::app);
                    if (ofs.is_open()) {
                        int src = msg->hasPar("srcIdx") ? (int)msg->par("srcIdx").longValue() : -1;
                        int seq = msg->hasPar("seq") ? (int)msg->par("seq").longValue() : -1;
                        int hops = msg->hasPar("hopCount") ? (int)msg->par("hopCount").longValue() : 0;
                        ofs << simTime().dbl() << "," << src << "," << seq << "," << tgen << "," << simTime().dbl() << "," << hops << "," << delay << "\n";
                        ofs.close();
                    }
                } catch(...) {}
            }
        }
        // check for node death after rx
        if (remainingEnergy <= 0 && !deathRecorded) recordNodeDeath();
        delete msg;
    }
}

void SensorNode::finish()
{
    cancelAndDelete(sendTimer);
    if (dataTimer) cancelAndDelete(dataTimer);
}

void SensorNode::consumeEnergyTx(int bits, double d)
{
    double E = eElec * bits + eAmp * bits * d * d;
    remainingEnergy -= E;
    if (remainingEnergy < 0) remainingEnergy = 0;
    if (remainingEnergy <= 0 && !deathRecorded) recordNodeDeath();
}

void SensorNode::consumeEnergyRx(int bits)
{
    double E = eElec * bits;
    remainingEnergy -= E;
    if (remainingEnergy < 0) remainingEnergy = 0;
    if (remainingEnergy <= 0 && !deathRecorded) recordNodeDeath();
}

void SensorNode::recordNodeDeath()
{
    deathRecorded = true;
    int idx = getIndex();
    try {
        std::filesystem::path p("/home/wte/uavwsn-rl/rl-1/uav-wsn-rl1/results/node_deaths.csv");
        if (!std::filesystem::exists(p)) {
            std::ofstream h(p.string(), std::ios::app);
            h << "node,t_death" << std::endl;
            h.close();
        }
        std::ofstream ofs(p.string(), std::ios::app);
        if (ofs.is_open()) {
            ofs << idx << "," << simTime().dbl() << "\n";
            ofs.close();
        }
    } catch(...) {}
}
