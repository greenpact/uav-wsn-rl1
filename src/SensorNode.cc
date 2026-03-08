#include "SensorNode.h"

#include <algorithm>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <string>

Define_Module(SensorNode);

namespace {

bool gRunCsvInitialized = false;

void initializeRunCsvOutputs()
{
    if (gRunCsvInitialized)
        return;

    std::filesystem::path dir("results");
    std::filesystem::create_directories(dir);

    {
        std::ofstream out(dir / "packet_generated.csv", std::ios::out | std::ios::trunc);
        if (out.is_open())
            out << "t,src,seq,t_gen" << '\n';
    }
    {
        std::ofstream out(dir / "packet_delivered.csv", std::ios::out | std::ios::trunc);
        if (out.is_open())
            out << "t,src,seq,t_gen,t_recv,hopCount,delay" << '\n';
    }
    {
        std::ofstream out(dir / "node_deaths.csv", std::ios::out | std::ios::trunc);
        if (out.is_open())
            out << "node,t_death" << '\n';
    }
    {
        std::ofstream out(dir / "node_positions.csv", std::ios::out | std::ios::trunc);
        if (out.is_open())
            out << "module,index,role,x,y" << '\n';
    }

    gRunCsvInitialized = true;
}

std::filesystem::path ensureResultFile(const std::string& filename, const std::string& header)
{
    std::filesystem::path dir("results");
    std::filesystem::create_directories(dir);
    std::filesystem::path filePath = dir / filename;

    if (!std::filesystem::exists(filePath)) {
        std::ofstream out(filePath, std::ios::out);
        if (out.is_open())
            out << header << '\n';
    }
    return filePath;
}

void appendCsvRow(const std::filesystem::path& filePath, const std::string& row)
{
    std::ofstream out(filePath, std::ios::app);
    if (out.is_open())
        out << row << '\n';
}

} // namespace

void SensorNode::initialize()
{
    initializeRunCsvOutputs();

    initialX = par("initialX").doubleValue();
    initialY = par("initialY").doubleValue();
    has80211 = par("has80211").boolValue();
    has802154 = par("has802154").boolValue();
    range11 = par("range11").doubleValue();
    range154 = par("range154").doubleValue();
    enableBeaconing = par("enableBeaconing").boolValue();
    enableDataGeneration = par("enableDataGeneration").boolValue();
    isBaseStation = par("isBaseStation").boolValue();

    try {
        const std::filesystem::path p = ensureResultFile(
            "node_positions.csv",
            "module,index,role,x,y"
        );
        const int idx = getIndex();
        const std::string role = isBaseStation ? "base_station" : "sensor";
        appendCsvRow(
            p,
            getFullPath() + "," +
            std::to_string(idx) + "," +
            role + "," +
            std::to_string(initialX) + "," +
            std::to_string(initialY)
        );
    } catch(...) {}

    // energy params
    remainingEnergy = par("initialEnergy").doubleValue();
    eElec = par("eElec").doubleValue();
    eAmp = par("eFreeSpace").doubleValue();

    if (enableBeaconing && !isBaseStation) {
        sendTimer = new cMessage("beaconTimer");
        double interval = par("beaconInterval").doubleValue();
        if (interval <= 0)
            interval = 1.0;
        scheduleAt(simTime() + uniform(0, std::min(0.5, interval)), sendTimer);
    }

    if (enableDataGeneration && !isBaseStation) {
        dataInterval = par("dataInterval").doubleValue();
        if (dataInterval <= 0)
            dataInterval = 1.0;
        dataTimer = new cMessage("dataTimer");
        scheduleAt(simTime() + uniform(0, std::min(0.5, dataInterval)), dataTimer);
    }
}

void SensorNode::handleMessage(cMessage *msg)
{
    if (msg == sendTimer) {
        if (remainingEnergy <= 0) {
            delete msg;
            sendTimer = nullptr;
            return;
        }

        // send beacon on available radios; if both available, alternate
        if (has802154) {
            cMessage *b = new cMessage("BEACON");
            b->addPar("txX") = initialX;
            b->addPar("txY") = initialY;
            b->addPar("txRange") = range154;
            b->addPar("txRadio") = 154;
            b->addPar("isControl") = true;
            send(b, "out");
        }
        if (has80211) {
            cMessage *b2 = new cMessage("BEACON");
            b2->addPar("txX") = initialX;
            b2->addPar("txY") = initialY;
            b2->addPar("txRange") = range11;
            b2->addPar("txRadio") = 11;
            b2->addPar("isControl") = true;
            send(b2, "out");
        }
        double interval = par("beaconInterval").doubleValue();
        if (interval <= 0)
            interval = 1.0;
        scheduleAt(simTime() + interval, sendTimer);
    }
    else if (msg == dataTimer) {
        if (remainingEnergy <= 0) {
            delete msg;
            dataTimer = nullptr;
            return;
        }

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
        recordScalar("dataGenerated", 1);

        try {
            const std::filesystem::path p = ensureResultFile("packet_generated.csv", "t,src,seq,t_gen");
            appendCsvRow(p, std::to_string(simTime().dbl()) + "," +
                            std::to_string(idx) + "," +
                            std::to_string(seqCounter - 1) + "," +
                            std::to_string(simTime().dbl()));
        } catch(...) {}

        double interval = dataInterval;
        scheduleAt(simTime() + interval, dataTimer);
        // send into routing via upward gate (same as beacon send)
        send(d, "out");
    }
    else {
        // message arrived from medium
        // receiving consumes energy
        int bits = msg->hasPar("isControl") && msg->par("isControl").boolValue()
            ? (int)par("controlPacketSize").doubleValue()
            : (int)par("dataPacketSize").doubleValue();
        consumeEnergyRx(bits);
        recordScalar("rx_count", 1);

        // if this is a data packet and this module is the base station (bs), log delivery
        if (isBaseStation && strcmp(msg->getName(), "DATA") == 0) {
            if (msg->hasPar("t_gen")) {
                double tgen = msg->par("t_gen").doubleValue();
                double delay = simTime().dbl() - tgen;
                recordScalar("e2eDelay", delay);

                try {
                    const std::filesystem::path p = ensureResultFile("packet_delivered.csv", "t,src,seq,t_gen,t_recv,hopCount,delay");
                    const int src = msg->hasPar("srcIdx") ? (int)msg->par("srcIdx").longValue() : -1;
                    const int seq = msg->hasPar("seq") ? (int)msg->par("seq").longValue() : -1;
                    const int hops = msg->hasPar("hopCount") ? (int)msg->par("hopCount").longValue() : 0;
                    appendCsvRow(p,
                        std::to_string(simTime().dbl()) + "," +
                        std::to_string(src) + "," +
                        std::to_string(seq) + "," +
                        std::to_string(tgen) + "," +
                        std::to_string(simTime().dbl()) + "," +
                        std::to_string(hops) + "," +
                        std::to_string(delay));
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
    if (sendTimer)
        cancelAndDelete(sendTimer);
    if (dataTimer)
        cancelAndDelete(dataTimer);
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
        const std::filesystem::path p = ensureResultFile("node_deaths.csv", "node,t_death");
        appendCsvRow(p, std::to_string(idx) + "," + std::to_string(simTime().dbl()));
    } catch(...) {}
}
