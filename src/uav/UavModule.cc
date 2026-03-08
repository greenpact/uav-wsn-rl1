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
    gmSpeed = currentSpeed();
    gmHeading = std::atan2(0.5 * areaY - currentY, 0.5 * areaX - currentX);
    maxAreaDistance = std::sqrt(areaX * areaX + areaY * areaY);

    cModule *bs = getParentModule()->getSubmodule("bs");
    if (bs) {
        bsX = bs->par("initialX").doubleValue();
        bsY = bs->par("initialY").doubleValue();
    }

    par("currentX").setDoubleValue(currentX);
    par("currentY").setDoubleValue(currentY);

    sensorState.resize(static_cast<size_t>(numNodes));
    if (numNodes > 0) {
        cModule *s0 = getParentModule()->getSubmodule("sensors", 0);
        if (s0 && s0->hasPar("range154"))
            sensorRange154 = s0->par("range154").doubleValue();
    }

    for (int i = 0; i < numNodes; ++i) {
        cModule *sm = getParentModule()->getSubmodule("sensors", i);
        if (!sm)
            continue;
        SensorSnapshot& st = sensorState[static_cast<size_t>(i)];
        st.known = true;
        st.x = sm->par("initialX").doubleValue();
        st.y = sm->par("initialY").doubleValue();
        st.energyRatio = 1.0;
        st.lastSeen = SIMTIME_ZERO;
    }

    // Pre-compute static neighborhood degree for guidance weighting.
    for (int i = 0; i < numNodes; ++i) {
        int degree = 0;
        for (int j = 0; j < numNodes; ++j) {
            if (i == j)
                continue;
            const double dx = sensorState[static_cast<size_t>(i)].x - sensorState[static_cast<size_t>(j)].x;
            const double dy = sensorState[static_cast<size_t>(i)].y - sensorState[static_cast<size_t>(j)].y;
            if (std::sqrt(dx * dx + dy * dy) <= sensorRange154)
                degree++;
        }
        sensorState[static_cast<size_t>(i)].degree = degree;
    }

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

    if (strcmp(msg->getName(), "BEACON") == 0) {
        const int srcIdx = msg->hasPar("srcIdx") ? static_cast<int>(msg->par("srcIdx").longValue()) : -1;
        if (srcIdx >= 0 && srcIdx < numNodes) {
            SensorSnapshot& st = sensorState[static_cast<size_t>(srcIdx)];
            st.known = true;
            if (msg->hasPar("txX"))
                st.x = msg->par("txX").doubleValue();
            if (msg->hasPar("txY"))
                st.y = msg->par("txY").doubleValue();
            if (msg->hasPar("senderEnergyRatio")) {
                const double er = msg->par("senderEnergyRatio").doubleValue();
                st.energyRatio = std::max(0.0, std::min(1.0, er));
            }
            st.lastSeen = simTime();
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
    const double t = simTime().dbl();

    if (missionDuration > 0.0 && t >= missionDuration) {
        currentX = bsX;
        currentY = bsY;
    }
    else {
        const double cycle = std::max(30.0, missionPeriod);
        const long long roundIndex = static_cast<long long>(std::floor(std::max(0.0, t) / cycle));
        ensureRoundPlan(roundIndex);

        const double phase = std::max(0.0, t) - static_cast<double>(roundIndex) * cycle;

        if (phase < roundHoldAtBs) {
            currentX = bsX;
            currentY = bsY;
        }
        else if (phase < roundHoldAtBs + roundTravelIn) {
            const double alpha = (phase - roundHoldAtBs) / std::max(1e-9, roundTravelIn);
            currentX = bsX + alpha * (0.0 - bsX);
            currentY = bsY + alpha * (roundEntryY - bsY);
        }
        else if (phase < roundHoldAtBs + roundTravelIn + roundSurvey) {
            updateGuidedGaussMarkov(positionUpdateInterval);
        }
        else if (phase < roundHoldAtBs + roundTravelIn + roundSurvey + roundTravelOut) {
            const double alpha =
                (phase - roundHoldAtBs - roundTravelIn - roundSurvey) / std::max(1e-9, roundTravelOut);
            currentX = areaX + alpha * (bsX - areaX);
            currentY = roundExitY + alpha * (bsY - roundExitY);
        }
        else {
            currentX = bsX;
            currentY = bsY;
        }
    }

    currentX = std::max(-250.0, std::min(areaX + 250.0, currentX));
    currentY = std::max(-250.0, std::min(areaY + 250.0, currentY));

    par("currentX").setDoubleValue(currentX);
    par("currentY").setDoubleValue(currentY);
    getDisplayString().setTagArg("p", 0, currentX);
    getDisplayString().setTagArg("p", 1, currentY);
}

void UavModule::ensureRoundPlan(long long roundIndex)
{
    if (roundIndex == activeRound)
        return;

    activeRound = roundIndex;

    const double cycle = std::max(30.0, missionPeriod);
    roundHoldAtBs = std::max(2.0, std::min(contactWindow, 0.20 * cycle));
    roundTravelIn = std::max(4.0, 0.18 * cycle);
    roundTravelOut = std::max(4.0, 0.18 * cycle);
    roundSurvey = std::max(2.0, cycle - roundHoldAtBs - roundTravelIn - roundTravelOut);

    // Randomized entry/exit points at the BS-facing side of the field (x=0).
    roundEntryY = areaY * unitNoise(roundIndex, 11);
    roundExitY = areaY * unitNoise(roundIndex, 29);

    // Per-round curve shape controls for distinct sweep directions.
    roundCurvePhase = 2.0 * 3.14159265358979323846 * unitNoise(roundIndex, 47);
    roundCurveMix = unitNoise(roundIndex, 73);
}

double UavModule::unitNoise(long long roundIndex, int salt)
{
    const double x = std::sin((roundIndex + 1) * (12.9898 + salt) + 78.233 + 0.123 * salt) * 43758.5453;
    return x - std::floor(x);
}

int UavModule::selectGuidanceTarget() const
{
    int bestIdx = -1;
    double bestScore = -1e18;

    for (int i = 0; i < numNodes; ++i) {
        const SensorSnapshot& st = sensorState[static_cast<size_t>(i)];
        if (!st.known)
            continue;

        const double age = (simTime() - st.lastSeen).dbl();
        if (st.lastSeen > SIMTIME_ZERO && age > std::max(3.0, beaconInterval * 4.0))
            continue;

        const double dist = std::sqrt((st.x - currentX) * (st.x - currentX) + (st.y - currentY) * (st.y - currentY));
        const double distNorm = std::max(0.0, std::min(1.0, dist / std::max(1e-9, maxAreaDistance)));
        const double degreeNorm = std::max(0.0, std::min(1.0, st.degree / 20.0));

        // Guided attraction to energetic high-degree nodes while avoiding very far pivots.
        const double score =
            0.55 * st.energyRatio +
            0.35 * degreeNorm +
            0.25 * (1.0 - distNorm);

        if (score > bestScore) {
            bestScore = score;
            bestIdx = i;
        }
    }

    return bestIdx;
}

void UavModule::reflectAtBoundary(double& x, double& y, double& heading) const
{
    if (x < 0.0) {
        x = 0.0;
        heading = 3.14159265358979323846 - heading;
    }
    else if (x > areaX) {
        x = areaX;
        heading = 3.14159265358979323846 - heading;
    }

    if (y < 0.0) {
        y = 0.0;
        heading = -heading;
    }
    else if (y > areaY) {
        y = areaY;
        heading = -heading;
    }
}

void UavModule::updateGuidedGaussMarkov(double dt)
{
    const int target = selectGuidanceTarget();

    double tx = 0.5 * areaX;
    double ty = 0.5 * areaY;
    if (target >= 0) {
        tx = sensorState[static_cast<size_t>(target)].x;
        ty = sensorState[static_cast<size_t>(target)].y;
    }

    const double dx = tx - currentX;
    const double dy = ty - currentY;
    const double dBias = std::atan2(dy, dx);
    const double vBias = std::max(minSpeed, maxSpeed);

    const double noiseScale = std::sqrt(std::max(0.0, 1.0 - gmAlpha * gmAlpha));
    gmSpeed = gmAlpha * gmSpeed + (1.0 - gmAlpha) * vBias + noiseScale * normal(0.0, gmSigmaV);
    gmHeading = gmAlpha * gmHeading + (1.0 - gmAlpha) * dBias + noiseScale * normal(0.0, gmSigmaD);

    gmSpeed = std::max(std::max(0.1, minSpeed), std::min(std::max(minSpeed, maxSpeed), gmSpeed));

    double nx = currentX + gmSpeed * std::cos(gmHeading) * dt;
    double ny = currentY + gmSpeed * std::sin(gmHeading) * dt;
    reflectAtBoundary(nx, ny, gmHeading);

    currentX = nx;
    currentY = ny;
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
