#include "UavModule.h"

#include <algorithm>
#include <cmath>
#include <cstring>

namespace uav {

Define_Module(UavModule);

void UavModule::initialize()
{
    numNodes = getParentModule()->par("numNodes").intValue();
    bsIndex = numNodes + 1;

    initialX = par("initialX").doubleValue();
    initialY = par("initialY").doubleValue();
    areaX = par("areaX").doubleValue();
    areaY = par("areaY").doubleValue();
    commRadius = par("commRadius").doubleValue();
    uploadRadius = par("uploadRadius").doubleValue();
    beaconInterval = par("beaconInterval").doubleValue();
    positionUpdateInterval = par("positionUpdateInterval").doubleValue();
    uploadInterval = par("uploadInterval").doubleValue();
    minSpeed = par("minSpeed").doubleValue();
    maxSpeed = par("maxSpeed").doubleValue();
    missionPeriod = par("missionPeriod").doubleValue();
    missionDuration = par("missionDuration").doubleValue();
    contactWindow = par("contactWindow").doubleValue();

    currentX = initialX;
    currentY = initialY;

    cModule *bs = getParentModule()->getSubmodule("bs");
    if (bs) {
        bsX = bs->par("initialX").doubleValue();
        bsY = bs->par("initialY").doubleValue();
    }

    par("currentX").setDoubleValue(currentX);
    par("currentY").setDoubleValue(currentY);

    moveEvent = new cMessage("uavMove");
    beaconEvent = new cMessage("uavBeacon");
    uploadEvent = new cMessage("uavUpload");

    scheduleAt(simTime() + positionUpdateInterval, moveEvent);
    scheduleAt(simTime() + beaconInterval, beaconEvent);
    scheduleAt(simTime() + uploadInterval, uploadEvent);
}

void UavModule::handleMessage(cMessage *msg)
{
    if (msg->isSelfMessage()) {
        if (msg == moveEvent) {
            updatePosition();
            scheduleAt(simTime() + positionUpdateInterval, moveEvent);
            return;
        }
        if (msg == beaconEvent) {
            broadcastBeacon();
            scheduleAt(simTime() + beaconInterval, beaconEvent);
            return;
        }
        if (msg == uploadEvent) {
            flushCollectedToBs();
            scheduleAt(simTime() + uploadInterval, uploadEvent);
            return;
        }
        delete msg;
        return;
    }

    if (strcmp(msg->getName(), "DATA") == 0) {
        const int dst = msg->hasPar("dstIdx") ? static_cast<int>(msg->par("dstIdx").longValue()) : numNodes;
        if (dst != -1 && dst != numNodes) {
            delete msg;
            return;
        }

        if (msg->hasPar("packetId") && msg->hasPar("srcIdx")) {
            const std::string packetId = msg->par("packetId").stringValue();
            const int srcIdx = static_cast<int>(msg->par("srcIdx").longValue());
            if (srcIdx >= 0 && srcIdx < numNodes)
                sendAckToSource(packetId, srcIdx);
        }

        if (!msg->hasPar("uavPickupTime"))
            msg->addPar("uavPickupTime") = simTime().dbl();

        collected.push_back(msg);
        collectedPackets++;
        return;
    }

    delete msg;
}

void UavModule::updatePosition()
{
    const double pi = 3.14159265358979323846;
    const double t = simTime().dbl();

    if (missionDuration > 0.0 && t >= missionDuration) {
        currentX = bsX;
        currentY = bsY;
    }
    else {
        const double transitDuration = std::max(10.0, 0.15 * missionPeriod);
        const double centerX = 0.5 * areaX;
        const double centerY = 0.5 * areaY;

        if (t < transitDuration) {
            const double alpha = t / transitDuration;
            currentX = initialX + alpha * (centerX - initialX);
            currentY = initialY + alpha * (centerY - initialY);
        }
        else {
            const double tt = t - transitDuration;
            const double omega = (2.0 * pi) / std::max(30.0, missionPeriod);
            const double radiusX = 0.42 * areaX;
            const double radiusY = 0.40 * areaY;
            currentX = centerX + radiusX * std::sin(omega * tt);
            currentY = centerY + radiusY * std::sin(2.0 * omega * tt + pi / 4.0);
        }
    }

    currentX = std::max(-250.0, std::min(areaX + 250.0, currentX));
    currentY = std::max(-250.0, std::min(areaY + 250.0, currentY));

    par("currentX").setDoubleValue(currentX);
    par("currentY").setDoubleValue(currentY);
    getDisplayString().setTagArg("p", 0, currentX);
    getDisplayString().setTagArg("p", 1, currentY);
}

void UavModule::broadcastBeacon()
{
    cMessage *beacon = new cMessage("BEACON");
    beacon->addPar("isControl") = true;
    beacon->addPar("isUavBeacon") = true;
    beacon->addPar("txX") = currentX;
    beacon->addPar("txY") = currentY;
    beacon->addPar("txRange") = commRadius;
    beacon->addPar("txRadio") = 154;
    beacon->addPar("uavSpeed") = currentSpeed();
    beacon->addPar("contactWindow") = contactWindow;
    beacon->addPar("predictedMissionPeriod") = missionPeriod;
    send(beacon, "out");
}

void UavModule::flushCollectedToBs()
{
    if (!inBsRegion())
        return;

    while (!collected.empty()) {
        cMessage *packet = collected.front();
        collected.pop_front();

        if (packet->hasPar("dstIdx"))
            packet->par("dstIdx") = bsIndex;
        else
            packet->addPar("dstIdx") = bsIndex;

        if (packet->hasPar("txX"))
            packet->par("txX") = currentX;
        else
            packet->addPar("txX") = currentX;

        if (packet->hasPar("txY"))
            packet->par("txY") = currentY;
        else
            packet->addPar("txY") = currentY;

        if (packet->hasPar("txRange"))
            packet->par("txRange") = uploadRadius;
        else
            packet->addPar("txRange") = uploadRadius;

        if (packet->hasPar("txRadio"))
            packet->par("txRadio") = 11;
        else
            packet->addPar("txRadio") = 11;

        if (packet->hasPar("isControl"))
            packet->par("isControl") = false;
        else
            packet->addPar("isControl") = false;

        send(packet, "out");
        uploadedPackets++;
    }
}

void UavModule::sendAckToSource(const std::string& packetId, int dstIdx)
{
    cMessage *ack = new cMessage("ACK");
    ack->addPar("packetId") = packetId.c_str();
    ack->addPar("dstIdx") = dstIdx;
    ack->addPar("txX") = currentX;
    ack->addPar("txY") = currentY;
    ack->addPar("txRange") = commRadius;
    ack->addPar("txRadio") = 154;
    ack->addPar("isControl") = true;
    send(ack, "out");
}

double UavModule::currentSpeed() const
{
    return 0.5 * (std::max(0.1, minSpeed) + std::max(0.1, maxSpeed));
}

bool UavModule::inBsRegion() const
{
    return distanceTo(bsX, bsY) <= uploadRadius;
}

double UavModule::distanceTo(double x, double y) const
{
    const double dx = currentX - x;
    const double dy = currentY - y;
    return std::sqrt(dx * dx + dy * dy);
}

void UavModule::finish()
{
    if (moveEvent)
        cancelAndDelete(moveEvent);
    if (beaconEvent)
        cancelAndDelete(beaconEvent);
    if (uploadEvent)
        cancelAndDelete(uploadEvent);

    while (!collected.empty()) {
        delete collected.front();
        collected.pop_front();
    }

    recordScalar("uavCollectedPackets", collectedPackets);
    recordScalar("uavUploadedPackets", uploadedPackets);
}

} // namespace uav
