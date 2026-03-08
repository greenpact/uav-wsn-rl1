#include "TrlarRouting.h"

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <cctype>
#include <cstdint>
#include <filesystem>
#include <iomanip>
#include <numeric>
#include <sstream>

namespace trlar {

Define_Module(TrlarRouting);

std::string TrlarRouting::trim(const std::string& in)
{
    size_t s = 0;
    while (s < in.size() && std::isspace(static_cast<unsigned char>(in[s])))
        s++;
    size_t e = in.size();
    while (e > s && std::isspace(static_cast<unsigned char>(in[e - 1])))
        e--;
    return in.substr(s, e - s);
}

std::vector<std::string> TrlarRouting::splitCsv(const std::string& csv)
{
    std::vector<std::string> out;
    std::stringstream ss(csv);
    std::string token;
    while (std::getline(ss, token, ',')) {
        const std::string t = trim(token);
        if (!t.empty())
            out.push_back(t);
    }
    return out;
}

std::string TrlarRouting::jsonEscape(const std::string& in)
{
    std::ostringstream oss;
    for (char c : in) {
        switch (c) {
            case '\\': oss << "\\\\"; break;
            case '"': oss << "\\\""; break;
            case '\n': oss << "\\n"; break;
            case '\r': oss << "\\r"; break;
            case '\t': oss << "\\t"; break;
            default: oss << c; break;
        }
    }
    return oss.str();
}

std::string TrlarRouting::jsonArray(const std::vector<double>& values)
{
    std::ostringstream oss;
    oss << '[';
    for (size_t i = 0; i < values.size(); ++i) {
        if (i)
            oss << ',';
        oss << std::setprecision(10) << values[i];
    }
    oss << ']';
    return oss.str();
}

std::string TrlarRouting::jsonArray2D(const std::vector<std::vector<double>>& values)
{
    std::ostringstream oss;
    oss << '[';
    for (size_t i = 0; i < values.size(); ++i) {
        if (i)
            oss << ',';
        oss << jsonArray(values[i]);
    }
    oss << ']';
    return oss.str();
}

bool TrlarRouting::parsePolicyProfile(const std::string& token, PolicyProfile& profile) const
{
    const std::string t = trim(token);
    if (t.empty())
        return false;

    if (t == "greedy") {
        profile.kind = PolicyKind::GREEDY;
        profile.param = 0.0;
        profile.tag = "greedy";
        return true;
    }

    const std::string epsPrefix = "eps_";
    if (t.rfind(epsPrefix, 0) == 0) {
        profile.kind = PolicyKind::EPS_GREEDY;
        profile.param = std::stod(t.substr(epsPrefix.size()));
        profile.tag = t;
        return true;
    }

    const std::string softPrefix = "softmax_";
    if (t.rfind(softPrefix, 0) == 0) {
        profile.kind = PolicyKind::SOFTMAX;
        profile.param = std::stod(t.substr(softPrefix.size()));
        profile.tag = t;
        return true;
    }

    return false;
}

void TrlarRouting::parsePolicyPoolConfig(const std::string& poolCsv, const std::string& weightsCsv)
{
    policyPool.clear();
    policyWeights.clear();

    const std::vector<std::string> poolTokens = splitCsv(poolCsv);
    for (const std::string& tok : poolTokens) {
        PolicyProfile p;
        if (parsePolicyProfile(tok, p))
            policyPool.push_back(p);
    }

    if (policyPool.empty()) {
        PolicyProfile p;
        p.kind = PolicyKind::EPS_GREEDY;
        p.param = 0.10;
        p.tag = "eps_0.10";
        policyPool.push_back(p);
    }

    policyWeights.assign(policyPool.size(), 1.0);
    const std::vector<std::string> weightTokens = splitCsv(weightsCsv);
    for (size_t i = 0; i < weightTokens.size() && i < policyWeights.size(); ++i) {
        const double w = std::max(0.0, std::stod(weightTokens[i]));
        policyWeights[i] = w;
    }

    const double sum = std::accumulate(policyWeights.begin(), policyWeights.end(), 0.0);
    if (sum <= 0.0)
        std::fill(policyWeights.begin(), policyWeights.end(), 1.0);
}

void TrlarRouting::updateEpisodePolicy()
{
    const double dur = std::max(1.0, episodeDuration.dbl());
    const long long episode = static_cast<long long>(std::floor(simTime().dbl() / dur));
    if (episode == currentEpisode && activePolicyIndex >= 0)
        return;

    currentEpisode = episode;

    if (policyPool.empty()) {
        activePolicyIndex = -1;
        return;
    }

    if (episodePolicyMode != "random_stratified") {
        activePolicyIndex = 0;
        return;
    }

    uint64_t h = 1469598103934665603ULL;
    auto mix = [&h](uint64_t v) {
        h ^= v;
        h *= 1099511628211ULL;
    };
    mix(static_cast<uint64_t>(episode + 1));
    mix(static_cast<uint64_t>(getEnvir()->getConfigEx()->getActiveRunNumber() + 1));

    const double totalW = std::accumulate(policyWeights.begin(), policyWeights.end(), 0.0);
    const double unit = static_cast<double>(h % 1000000ULL) / 1000000.0;
    const double draw = unit * std::max(1e-9, totalW);
    double csum = 0.0;
    activePolicyIndex = static_cast<int>(policyPool.size()) - 1;
    for (size_t i = 0; i < policyWeights.size(); ++i) {
        csum += policyWeights[i];
        if (draw <= csum) {
            activePolicyIndex = static_cast<int>(i);
            break;
        }
    }
}

void TrlarRouting::initialize()
{
    nodeId = par("nodeId").intValue();
    numNodes = getParentModule()->par("numNodes").intValue();
    uavIndex = numNodes;
    bsIndex = numNodes + 1;

    bufferSize = par("bufferSize").intValue();
    maxCandidates = par("maxCandidates").intValue();
    maxRetries = par("maxRetries").intValue();
    maxRouteAttempts = par("maxRouteAttempts").intValue();
    maxHops = par("maxHops").intValue();

    minLinkQuality = par("minLinkQuality").doubleValue();
    minNeighborEnergyRatio = par("minNeighborEnergyRatio").doubleValue();
    adaptationAlpha = par("adaptationAlpha").doubleValue();
    queueHighWatermark = par("queueHighWatermark").doubleValue();
    congestionPenalty = par("congestionPenalty").doubleValue();
    delayThreshold = par("delayThreshold").doubleValue();

    rewardW1 = par("rewardW1").doubleValue();
    rewardW2 = par("rewardW2").doubleValue();
    rewardW3 = par("rewardW3").doubleValue();
    rewardW4 = par("rewardW4").doubleValue();

    ackTimeout = par("ackTimeout");
    neighborTimeout = par("neighborTimeout");
    decisionInterval = par("decisionInterval");

    cModule *sensorModule = getParentModule()->getSubmodule("sensors", nodeId);
    sensorNode = dynamic_cast<SensorNode *>(sensorModule);
    if (!sensorNode)
        throw cRuntimeError("Routing module %d cannot bind to sensors[%d]", nodeId, nodeId);

    selfX = sensorModule->par("initialX").doubleValue();
    selfY = sensorModule->par("initialY").doubleValue();
    areaX = getParentModule()->par("areaX").doubleValue();
    areaY = getParentModule()->par("areaY").doubleValue();
    areaDiag = std::sqrt(areaX * areaX + areaY * areaY);
    commRange = sensorModule->par("range154").doubleValue();
    initialEnergy = sensorModule->par("initialEnergy").doubleValue();
    dataPacketBits = static_cast<int>(sensorModule->par("dataPacketSize").doubleValue());
    controlPacketBits = static_cast<int>(sensorModule->par("controlPacketSize").doubleValue());

    transitionsPath = par("transitionsFile").stringValue();
    openTransitionsFile();

    episodePolicyMode = par("episodePolicyMode").stringValue();
    episodeDuration = par("episodeDuration");
    utilityCoeffLq = par("utilityCoeffLq").doubleValue();
    utilityCoeffNeighEnergy = par("utilityCoeffNeighEnergy").doubleValue();
    utilityCoeffQueue = par("utilityCoeffQueue").doubleValue();
    utilityCoeffFail = par("utilityCoeffFail").doubleValue();
    utilityCoeffUavDistance = par("utilityCoeffUavDistance").doubleValue();
    mobilityTag = par("mobilityTag").stringValue();
    parsePolicyPoolConfig(par("policyPool").stringValue(), par("policyMixWeights").stringValue());

    const std::string modelFile = par("modelFile").stringValue();
    behaviorPolicyEnabled = modelFile.empty();
    if (!modelFile.empty()) {
        const bool loaded = agent.loadModel(modelFile);
        EV_INFO << "Routing node " << nodeId << " model load " << (loaded ? "succeeded" : "failed")
                << " from " << modelFile << "\n";
    }

    std::filesystem::path tpath(transitionsPath);
    if (!transitionsPath.empty()) {
        metadataPath = (tpath.has_parent_path() ? tpath.parent_path() : std::filesystem::path(".")) / "metadata.json";
    }

    if (nodeId == 0)
        writeRunMetadata();

    decisionEvent = new cMessage("routingDecision");
}

void TrlarRouting::handleMessage(cMessage *msg)
{
    if (msg->isSelfMessage()) {
        handleSelfMessage(msg);
        return;
    }

    const char *gateName = msg->getArrivalGate()->getName();
    if (strcmp(gateName, "inFromSensor") == 0)
        handleFromSensor(msg);
    else if (strcmp(gateName, "inFromMedium") == 0)
        handleFromMedium(msg);
    else
        delete msg;
}

void TrlarRouting::handleSelfMessage(cMessage *msg)
{
    if (msg == decisionEvent) {
        tryForwardHeadPacket();
        if (!queue.empty() && !inFlight && !decisionEvent->isScheduled())
            scheduleAt(simTime() + decisionInterval, decisionEvent);
        return;
    }

    if (msg == ackTimeoutEvent) {
        if (!inFlight) {
            delete ackTimeoutEvent;
            ackTimeoutEvent = nullptr;
            return;
        }

        const bool shouldRetry = pending.backup && (pending.retries < maxRetries);
        completePendingTx(false, false);

        cMessage *retryPacket = nullptr;
        if (shouldRetry) {
            retryPacket = pending.backup->dup();
            const int retryCount = pending.retries + 1;
            if (retryPacket->hasPar("routeRetries"))
                retryPacket->par("routeRetries") = retryCount;
            else
                retryPacket->addPar("routeRetries") = retryCount;
        }

        resetPendingState();

        delete ackTimeoutEvent;
        ackTimeoutEvent = nullptr;

        if (retryPacket)
            queue.push_front(retryPacket);

        if (!queue.empty() && !decisionEvent->isScheduled())
            scheduleAt(simTime() + decisionInterval, decisionEvent);
        return;
    }

    delete msg;
}

void TrlarRouting::handleFromSensor(cMessage *msg)
{
    if (sensorNode->getRemainingEnergy() <= 0.0) {
        delete msg;
        return;
    }

    if (strcmp(msg->getName(), "BEACON") == 0) {
        const double er = energyRatio();
        if (msg->hasPar("senderEnergyRatio"))
            msg->par("senderEnergyRatio") = er;
        else
            msg->addPar("senderEnergyRatio") = er;

        const double qr = queueRatio();
        if (msg->hasPar("senderQueueRatio"))
            msg->par("senderQueueRatio") = qr;
        else
            msg->addPar("senderQueueRatio") = qr;

        if (msg->hasPar("txX"))
            msg->par("txX") = selfX;
        else
            msg->addPar("txX") = selfX;
        if (msg->hasPar("txY"))
            msg->par("txY") = selfY;
        else
            msg->addPar("txY") = selfY;
        if (msg->hasPar("txRange"))
            msg->par("txRange") = commRange;
        else
            msg->addPar("txRange") = commRange;
        if (msg->hasPar("txRadio"))
            msg->par("txRadio") = 154;
        else
            msg->addPar("txRadio") = 154;
        if (msg->hasPar("isControl"))
            msg->par("isControl") = true;
        else
            msg->addPar("isControl") = true;

        sensorNode->consumeEnergyTx(controlPacketBits, commRange);
        send(msg, "outToMedium");
        return;
    }

    if (strcmp(msg->getName(), "DATA") == 0) {
        if (!msg->hasPar("packetId"))
            msg->addPar("packetId") = packetIdFor(msg).c_str();
        if (!msg->hasPar("genTime"))
            msg->addPar("genTime") = simTime().dbl();
        if (!msg->hasPar("routeAttempts"))
            msg->addPar("routeAttempts") = 0;
        if (!msg->hasPar("routeRetries"))
            msg->addPar("routeRetries") = 0;
        if (!msg->hasPar("hopCount"))
            msg->addPar("hopCount") = 0;
        enqueueData(msg, false);
        return;
    }

    delete msg;
}

void TrlarRouting::handleFromMedium(cMessage *msg)
{
    const int srcIdx = msg->hasPar("srcIdx") ? static_cast<int>(msg->par("srcIdx").longValue()) : -1;

    if (strcmp(msg->getName(), "BEACON") == 0) {
        updateNeighborFromBeacon(srcIdx, msg);
        delete msg;
        return;
    }

    if (strcmp(msg->getName(), "ACK") == 0) {
        const int dst = msg->hasPar("dstIdx") ? static_cast<int>(msg->par("dstIdx").longValue()) : nodeId;
        if (dst == nodeId && inFlight && msg->hasPar("packetId")) {
            const std::string ackPacketId = msg->par("packetId").stringValue();
            if (ackPacketId == pending.packetId) {
                const bool timely = (simTime() - pending.sentAt).dbl() <= delayThreshold;
                completePendingTx(true, timely);
                if (ackTimeoutEvent && ackTimeoutEvent->isScheduled())
                    cancelEvent(ackTimeoutEvent);
                delete ackTimeoutEvent;
                ackTimeoutEvent = nullptr;
                resetPendingState();

                if (!queue.empty() && !decisionEvent->isScheduled())
                    scheduleAt(simTime(), decisionEvent);
            }
        }
        delete msg;
        return;
    }

    if (strcmp(msg->getName(), "DATA") == 0) {
        const int dst = msg->hasPar("dstIdx") ? static_cast<int>(msg->par("dstIdx").longValue()) : -1;
        if (dst != -1 && dst != nodeId) {
            delete msg;
            return;
        }

        if (sensorNode->getRemainingEnergy() <= 0.0) {
            delete msg;
            return;
        }

        const int hops = msg->hasPar("hopCount") ? static_cast<int>(msg->par("hopCount").longValue()) : 0;
        if (hops > maxHops) {
            failedForwards++;
            logTransition("drop_hops", "\"reason\":\"max_hops\"");
            delete msg;
            return;
        }

        sensorNode->consumeEnergyRx(dataPacketBits);
        updateNeighborFromData(srcIdx, msg);

        if (msg->hasPar("packetId") && srcIdx >= 0 && srcIdx < numNodes)
            sendAck(msg->par("packetId").stringValue(), srcIdx);

        enqueueData(msg, true);
        return;
    }

    delete msg;
}

void TrlarRouting::enqueueData(cMessage *msg, bool fromRelay)
{
    if (static_cast<int>(queue.size()) >= bufferSize) {
        queuedDrops++;
        logTransition("drop_buffer", fromRelay ? "\"source\":\"relay\"" : "\"source\":\"local\"");
        delete msg;
        return;
    }

    queue.push_back(msg);
    if (!inFlight && !decisionEvent->isScheduled())
        scheduleAt(simTime(), decisionEvent);
}

void TrlarRouting::tryForwardHeadPacket()
{
    if (inFlight || queue.empty())
        return;

    if (sensorNode->getRemainingEnergy() <= 0.0) {
        while (!queue.empty()) {
            delete queue.front();
            queue.pop_front();
        }
        return;
    }

    cMessage *packet = queue.front();

    std::vector<int> candidates = buildCandidateSet();
    if (candidates.empty()) {
        int attempts = packet->hasPar("routeAttempts") ? static_cast<int>(packet->par("routeAttempts").longValue()) : 0;
        attempts++;
        if (packet->hasPar("routeAttempts"))
            packet->par("routeAttempts") = attempts;
        else
            packet->addPar("routeAttempts") = attempts;

        queue.pop_front();
        if (attempts > maxRouteAttempts) {
            queuedDrops++;
            logTransition("drop_noroute", "\"reason\":\"no_candidate\"");
            delete packet;
        }
        else {
            queue.push_back(packet);
            if (!decisionEvent->isScheduled())
                scheduleAt(simTime() + decisionInterval, decisionEvent);
        }
        return;
    }

    const std::vector<double> stateContext = buildStateContext();
    const std::vector<std::vector<double>> candidateFeatures = buildCandidateFeatures(candidates);

    std::vector<double> actionScores;
    std::string policyTag;
    double policyParam = 0.0;
    std::string explorationTier;
    int actionIdx = chooseActionIndex(candidateFeatures, actionScores, policyTag, policyParam, explorationTier);

    int bestHop = -1;
    double bestScore = -1e18;
    if (actionIdx >= 0 && actionIdx < static_cast<int>(candidates.size())
        && actionIdx < static_cast<int>(candidateFeatures.size())
        && actionIdx < static_cast<int>(actionScores.size())) {
        bestHop = candidates[static_cast<size_t>(actionIdx)];
        auto bIt = biasTable.find(bestHop);
        const double bias = (bIt != biasTable.end()) ? bIt->second : 0.0;
        bestScore = actionScores[static_cast<size_t>(actionIdx)] + bias;
    }

    if (bestHop < 0) {
        queue.pop_front();
        queue.push_back(packet);
        if (!decisionEvent->isScheduled())
            scheduleAt(simTime() + decisionInterval, decisionEvent);
        return;
    }

    queue.pop_front();

    if (!packet->hasPar("packetId"))
        packet->addPar("packetId") = packetIdFor(packet).c_str();
    const std::string packetId = packet->par("packetId").stringValue();

    if (packet->hasPar("dstIdx"))
        packet->par("dstIdx") = bestHop;
    else
        packet->addPar("dstIdx") = bestHop;

    if (packet->hasPar("txX"))
        packet->par("txX") = selfX;
    else
        packet->addPar("txX") = selfX;
    if (packet->hasPar("txY"))
        packet->par("txY") = selfY;
    else
        packet->addPar("txY") = selfY;
    if (packet->hasPar("txRange"))
        packet->par("txRange") = commRange;
    else
        packet->addPar("txRange") = commRange;
    if (packet->hasPar("txRadio"))
        packet->par("txRadio") = 154;
    else
        packet->addPar("txRadio") = 154;
    if (packet->hasPar("isControl"))
        packet->par("isControl") = false;
    else
        packet->addPar("isControl") = false;

    if (packet->hasPar("senderEnergyRatio"))
        packet->par("senderEnergyRatio") = energyRatio();
    else
        packet->addPar("senderEnergyRatio") = energyRatio();

    double txDistance = commRange;
    auto nIt = neighbors.find(bestHop);
    if (nIt != neighbors.end() && nIt->second.hasPosition)
        txDistance = computeDistance(selfX, selfY, nIt->second.x, nIt->second.y);

    sensorNode->consumeEnergyTx(dataPacketBits, txDistance);

    pending.packetId = packetId;
    pending.nextHop = bestHop;
    pending.sentAt = simTime();
    pending.decisionAt = simTime();
    pending.generatedAt = (packet->hasPar("genTime") ? packet->par("genTime").doubleValue() : packet->getCreationTime().dbl());
    pending.txDistance = txDistance;
    pending.retries = packet->hasPar("routeRetries") ? static_cast<int>(packet->par("routeRetries").longValue()) : 0;
    pending.policyTag = policyTag;
    pending.policyParam = policyParam;
    pending.explorationTier = explorationTier;
    pending.stateContext = stateContext;
    pending.candidateFeatures = candidateFeatures;
    pending.actionIndex = actionIdx;
    if (actionIdx >= 0 && actionIdx < static_cast<int>(candidateFeatures.size()))
        pending.actionFeatures = candidateFeatures[static_cast<size_t>(actionIdx)];
    else
        pending.actionFeatures.clear();
    pending.backup = packet->dup();
    inFlight = true;

    forwardedPackets++;

    std::ostringstream decisionPayload;
    decisionPayload << "\"packet\":\"" << packetId << "\",";
    decisionPayload << "\"run\":" << getEnvir()->getConfigEx()->getActiveRunNumber() << ",";
    decisionPayload << "\"config\":\"" << jsonEscape(getEnvir()->getConfigEx()->getActiveConfigName()) << "\",";
    decisionPayload << "\"policy_mode\":\"" << jsonEscape(policyTag) << "\",";
    decisionPayload << "\"policy_param\":" << policyParam << ",";
    decisionPayload << "\"exploration_tier\":\"" << jsonEscape(explorationTier) << "\",";
    decisionPayload << "\"nextHop\":" << bestHop << ",";
    decisionPayload << "\"action_index\":" << actionIdx << ",";
    decisionPayload << "\"score\":" << bestScore << ",";
    decisionPayload << "\"state\":" << jsonArray(stateContext) << ",";
    decisionPayload << "\"candidates\":" << jsonArray2D(candidateFeatures) << ",";
    decisionPayload << "\"action_features\":" << jsonArray(pending.actionFeatures) << ",";
    decisionPayload << "\"queue\":" << queue.size();
    logTransition("decision", decisionPayload.str());

    send(packet, "outToMedium");

    if (ackTimeoutEvent) {
        if (ackTimeoutEvent->isScheduled())
            cancelEvent(ackTimeoutEvent);
        delete ackTimeoutEvent;
    }
    ackTimeoutEvent = new cMessage("ackTimeout");
    scheduleAt(simTime() + ackTimeout, ackTimeoutEvent);
}

std::vector<int> TrlarRouting::buildCandidateSet() const
{
    std::vector<int> candidates;
    candidates.reserve(neighbors.size());

    for (const auto& entry : neighbors) {
        const NeighborInfo& n = entry.second;
        if (n.id == nodeId || n.id == bsIndex)
            continue;
        if ((simTime() - n.lastSeen) > neighborTimeout)
            continue;

        if (n.id != uavIndex) {
            if (n.linkQuality < minLinkQuality)
                continue;
            if (n.energyRatio < minNeighborEnergyRatio)
                continue;
        }

        candidates.push_back(n.id);
    }

    std::sort(candidates.begin(), candidates.end(), [this](int a, int b) {
        const NeighborInfo& na = neighbors.at(a);
        const NeighborInfo& nb = neighbors.at(b);
        const auto aIt = biasTable.find(a);
        const auto bIt = biasTable.find(b);
        const double aBias = (aIt != biasTable.end()) ? aIt->second : 0.0;
        const double bBias = (bIt != biasTable.end()) ? bIt->second : 0.0;
        const double scoreA = na.linkQuality + 0.35 * na.energyRatio - 0.4 * neighborFailureRatio(na) + 0.15 * aBias;
        const double scoreB = nb.linkQuality + 0.35 * nb.energyRatio - 0.4 * neighborFailureRatio(nb) + 0.15 * bBias;
        return scoreA > scoreB;
    });

    if (static_cast<int>(candidates.size()) > maxCandidates)
        candidates.resize(static_cast<size_t>(maxCandidates));

    return candidates;
}

std::vector<double> TrlarRouting::buildActionFeatures(const NeighborInfo& neighbor) const
{
    return {
        std::max(0.0, std::min(1.0, energyRatio())),
        std::max(0.0, std::min(1.0, queueRatio())),
        std::max(0.0, std::min(1.0, neighborDensityRatio())),
        std::max(0.0, std::min(1.0, normalizedTau())),
        std::max(0.0, std::min(1.0, neighbor.linkQuality)),
        std::max(0.0, std::min(1.0, neighbor.energyRatio)),
        std::max(0.0, std::min(1.0, neighborFailureRatio(neighbor))),
        std::max(0.0, std::min(1.0, normalizedUavDistance()))
    };
}

std::vector<double> TrlarRouting::buildStateContext() const
{
    return {
        std::max(0.0, std::min(1.0, energyRatio())),
        std::max(0.0, std::min(1.0, queueRatio())),
        std::max(0.0, std::min(1.0, neighborDensityRatio())),
        std::max(0.0, std::min(1.0, normalizedTau())),
        0.0,
        0.0,
        0.0,
        std::max(0.0, std::min(1.0, normalizedUavDistance()))
    };
}

std::vector<std::vector<double>> TrlarRouting::buildCandidateFeatures(const std::vector<int>& candidateIds) const
{
    std::vector<std::vector<double>> out;
    out.reserve(candidateIds.size());
    for (int id : candidateIds) {
        auto it = neighbors.find(id);
        if (it == neighbors.end())
            continue;
        out.push_back(buildActionFeatures(it->second));
    }
    return out;
}

double TrlarRouting::computeUtility(const std::vector<double>& features) const
{
    if (features.size() < 8)
        return 0.0;
    const double queue = std::max(0.0, std::min(1.0, features[1]));
    const double lq = std::max(0.0, std::min(1.0, features[4]));
    const double neighEnergy = std::max(0.0, std::min(1.0, features[5]));
    const double failRatio = std::max(0.0, std::min(1.0, features[6]));
    const double uavDistance = std::max(0.0, std::min(1.0, features[7]));
    return (utilityCoeffLq * lq)
        + (utilityCoeffNeighEnergy * neighEnergy)
        - (utilityCoeffQueue * queue)
        - (utilityCoeffFail * failRatio)
        - (utilityCoeffUavDistance * uavDistance);
}

int TrlarRouting::chooseActionIndex(const std::vector<std::vector<double>>& candidateFeatures,
                                    std::vector<double>& scoresOut,
                                    std::string& policyTagOut,
                                    double& policyParamOut,
                                    std::string& explorationTierOut)
{
    scoresOut.clear();
    if (candidateFeatures.empty())
        return -1;

    updateEpisodePolicy();

    PolicyProfile policy;
    if (activePolicyIndex >= 0 && activePolicyIndex < static_cast<int>(policyPool.size()))
        policy = policyPool[static_cast<size_t>(activePolicyIndex)];
    else
        policy = policyPool.front();

    policyTagOut = policy.tag;
    policyParamOut = policy.param;
    explorationTierOut = (policy.kind == PolicyKind::EPS_GREEDY && policy.param >= 0.25) ? "high" : "standard";

    scoresOut.reserve(candidateFeatures.size());
    for (const auto& f : candidateFeatures)
        scoresOut.push_back(computeUtility(f));

    int bestIdx = 0;
    for (size_t i = 1; i < scoresOut.size(); ++i) {
        if (scoresOut[i] > scoresOut[static_cast<size_t>(bestIdx)])
            bestIdx = static_cast<int>(i);
    }

    if (!behaviorPolicyEnabled) {
        policyTagOut = "model_inference";
        policyParamOut = 0.0;
        explorationTierOut = "none";
        scoresOut.clear();
        scoresOut.reserve(candidateFeatures.size());
        for (const auto& f : candidateFeatures)
            scoresOut.push_back(agent.scoreAction(f));
        bestIdx = 0;
        for (size_t i = 1; i < scoresOut.size(); ++i) {
            if (scoresOut[i] > scoresOut[static_cast<size_t>(bestIdx)])
                bestIdx = static_cast<int>(i);
        }
        return bestIdx;
    }

    if (policy.kind == PolicyKind::GREEDY)
        return bestIdx;

    if (policy.kind == PolicyKind::EPS_GREEDY) {
        const double eps = std::max(0.0, std::min(1.0, policy.param));
        if (uniform(0.0, 1.0) < eps)
            return intuniform(0, static_cast<int>(candidateFeatures.size()) - 1);
        return bestIdx;
    }

    const double T = std::max(1e-6, policy.param);
    double maxScore = scoresOut[0];
    for (double s : scoresOut)
        maxScore = std::max(maxScore, s);

    std::vector<double> probs(scoresOut.size(), 0.0);
    double sumP = 0.0;
    for (size_t i = 0; i < scoresOut.size(); ++i) {
        probs[i] = std::exp((scoresOut[i] - maxScore) / T);
        sumP += probs[i];
    }
    if (sumP <= 0.0)
        return bestIdx;

    const double u = uniform(0.0, 1.0);
    double csum = 0.0;
    for (size_t i = 0; i < probs.size(); ++i) {
        csum += probs[i] / sumP;
        if (u <= csum)
            return static_cast<int>(i);
    }

    return static_cast<int>(probs.size()) - 1;
}

void TrlarRouting::updateNeighborFromBeacon(int srcIdx, cMessage *msg)
{
    if (srcIdx < 0)
        return;

    NeighborInfo& info = neighbors[srcIdx];
    info.id = srcIdx;
    info.lastSeen = simTime();

    if (msg->hasPar("senderEnergyRatio"))
        info.energyRatio = std::max(0.0, std::min(1.0, msg->par("senderEnergyRatio").doubleValue()));

    if (msg->hasPar("txX") && msg->hasPar("txY") && msg->hasPar("txRange")) {
        const double txX = msg->par("txX").doubleValue();
        const double txY = msg->par("txY").doubleValue();
        const double txRange = msg->par("txRange").doubleValue();
        info.linkQuality = estimateLinkQuality(txX, txY, txRange, info.linkQuality);
        info.hasPosition = true;
        info.x = txX;
        info.y = txY;
    }

    const bool isUavBeacon = (srcIdx == uavIndex)
                         || (msg->hasPar("isUavBeacon") && msg->par("isUavBeacon").boolValue());
    info.isUav = isUavBeacon;

    if (isUavBeacon) {
        const double ux = msg->hasPar("txX") ? msg->par("txX").doubleValue() : selfX;
        const double uy = msg->hasPar("txY") ? msg->par("txY").doubleValue() : selfY;
        const double ur = msg->hasPar("txRange") ? msg->par("txRange").doubleValue() : commRange;
        const double speed = msg->hasPar("uavSpeed") ? std::max(0.1, msg->par("uavSpeed").doubleValue()) : 10.0;
        const double d = computeDistance(selfX, selfY, ux, uy);
        const double eta = d <= ur ? 0.0 : (d - ur) / speed;
        lastUavBeacon = simTime();
        uavEtaEstimate = eta;
        hasUavPosition = true;
        lastUavX = ux;
        lastUavY = uy;
        info.energyRatio = 1.0;
    }

    if (biasTable.find(srcIdx) == biasTable.end())
        biasTable[srcIdx] = 0.0;
}

void TrlarRouting::updateNeighborFromData(int srcIdx, cMessage *msg)
{
    if (srcIdx < 0)
        return;

    NeighborInfo& info = neighbors[srcIdx];
    info.id = srcIdx;
    info.lastSeen = simTime();

    if (msg->hasPar("senderEnergyRatio"))
        info.energyRatio = std::max(0.0, std::min(1.0, msg->par("senderEnergyRatio").doubleValue()));

    if (msg->hasPar("txX") && msg->hasPar("txY") && msg->hasPar("txRange")) {
        const double txX = msg->par("txX").doubleValue();
        const double txY = msg->par("txY").doubleValue();
        const double txRange = msg->par("txRange").doubleValue();
        info.linkQuality = estimateLinkQuality(txX, txY, txRange, info.linkQuality);
        info.hasPosition = true;
        info.x = txX;
        info.y = txY;
    }

    if (biasTable.find(srcIdx) == biasTable.end())
        biasTable[srcIdx] = 0.0;
}

void TrlarRouting::sendAck(const std::string& packetId, int dstIdx)
{
    cMessage *ack = new cMessage("ACK");
    ack->addPar("packetId") = packetId.c_str();
    ack->addPar("dstIdx") = dstIdx;
    ack->addPar("txX") = selfX;
    ack->addPar("txY") = selfY;
    ack->addPar("txRange") = commRange;
    ack->addPar("txRadio") = 154;
    ack->addPar("isControl") = true;

    sensorNode->consumeEnergyTx(controlPacketBits, commRange);
    send(ack, "outToMedium");
}

void TrlarRouting::completePendingTx(bool success, bool timely)
{
    if (!inFlight || pending.nextHop < 0)
        return;

    NeighborInfo& n = neighbors[pending.nextHop];
    if (success) {
        n.successCount++;
        successfulForwards++;
    }
    else {
        n.failureCount++;
        failedForwards++;
    }

    const double outcome = (success && timely) ? 1.0 : -1.0;
    double& bias = biasTable[pending.nextHop];
    bias = (1.0 - adaptationAlpha) * bias + adaptationAlpha * outcome;

    const double delay = (simTime() - pending.sentAt).dbl();
    n.lastDelay = delay;

    if (queueRatio() > queueHighWatermark && (!success || !timely)) {
        bias -= congestionPenalty;
        applyCongestionPenalty();
    }

    const double eElec = sensorNode->par("eElec").doubleValue();
    const double eAmp = sensorNode->par("eFreeSpace").doubleValue();
    const double txEnergy = eElec * dataPacketBits + eAmp * dataPacketBits * pending.txDistance * pending.txDistance;
    const double reward = rewardW1 * (success ? 1.0 : 0.0)
                        - rewardW2 * txEnergy
                        - rewardW3 * delay
                        - rewardW4 * (success ? 0.0 : 1.0);

    const std::vector<double> nextState = buildStateContext();
    const std::vector<int> nextCandidateIds = buildCandidateSet();
    const std::vector<std::vector<double>> nextCandidates = buildCandidateFeatures(nextCandidateIds);

    const double tGen = pending.generatedAt.dbl();
    const double tDecision = pending.decisionAt.dbl();
    const double tTx = pending.sentAt.dbl();
    const double tAck = simTime().dbl();
    const double tRx = pending.sentAt.dbl() + std::max(0.0, delay);
    const double tUavPickup = (success && pending.nextHop == uavIndex) ? simTime().dbl() : -1.0;
    const double tBsReceive = -1.0;

    std::ostringstream payload;
    payload << "\"packet\":\"" << pending.packetId << "\",";
    payload << "\"run_id\":\"run_" << getEnvir()->getConfigEx()->getActiveRunNumber() << "\",";
    payload << "\"seed\":" << getEnvir()->getConfigEx()->getActiveRunNumber() << ",";
    payload << "\"config_name\":\"" << jsonEscape(getEnvir()->getConfigEx()->getActiveConfigName()) << "\",";
    payload << "\"policy_tag\":\"" << jsonEscape(pending.policyTag) << "\",";
    payload << "\"policy_mode\":\"" << jsonEscape(pending.policyTag) << "\",";
    payload << "\"mobility_tag\":\"" << jsonEscape(mobilityTag) << "\",";
    payload << "\"epsilon_or_temperature\":" << pending.policyParam << ",";
    payload << "\"exploration_tier\":\"" << jsonEscape(pending.explorationTier) << "\",";
    payload << "\"node_id\":" << nodeId << ",";
    payload << "\"packet_id\":\"" << jsonEscape(pending.packetId) << "\",";
    payload << "\"state\":" << jsonArray(pending.stateContext) << ",";
    payload << "\"candidates\":" << jsonArray2D(pending.candidateFeatures) << ",";
    payload << "\"action_index\":" << pending.actionIndex << ",";
    payload << "\"action_features\":" << jsonArray(pending.actionFeatures) << ",";
    payload << "\"reward\":" << reward << ",";
    payload << "\"next_state_context\":" << jsonArray(nextState) << ",";
    payload << "\"next_candidates\":" << jsonArray2D(nextCandidates) << ",";
    payload << "\"done\":0,";
    payload << "\"t_gen\":" << tGen << ",";
    payload << "\"t_decision\":" << tDecision << ",";
    payload << "\"t_tx\":" << tTx << ",";
    payload << "\"t_rx\":" << tRx << ",";
    payload << "\"t_ack\":" << tAck << ",";
    payload << "\"t_uav_pickup\":" << tUavPickup << ",";
    payload << "\"t_bs_receive\":" << tBsReceive << ",";
    payload << "\"delay_e2e\":" << ((tBsReceive >= 0.0) ? (tBsReceive - tGen) : -1.0) << ",";
    payload << "\"schema_version\":1,";
    payload << "\"nextHop\":" << pending.nextHop << ",";
    payload << "\"success\":" << (success ? 1 : 0) << ",";
    payload << "\"timely\":" << (timely ? 1 : 0) << ",";
    payload << "\"delay\":" << delay << ",";
    payload << "\"event_quality\":{\"ack\":" << (success ? 1 : 0)
            << ",\"delay\":" << delay
            << ",\"drop\":" << (success ? 0 : 1)
            << ",\"retries\":" << pending.retries << "}";
    logTransition("transition", payload.str());
}

void TrlarRouting::resetPendingState()
{
    if (pending.backup) {
        delete pending.backup;
        pending.backup = nullptr;
    }
    pending = PendingTx();
    inFlight = false;
}

double TrlarRouting::computeDistance(double x1, double y1, double x2, double y2) const
{
    const double dx = x1 - x2;
    const double dy = y1 - y2;
    return std::sqrt(dx * dx + dy * dy);
}

double TrlarRouting::estimateLinkQuality(double txX, double txY, double txRange, double prevLq) const
{
    const double d = computeDistance(selfX, selfY, txX, txY);
    const double base = 1.0 - (d / std::max(1.0, txRange));
    const double clipped = std::max(0.0, std::min(1.0, base));
    return 0.7 * prevLq + 0.3 * clipped;
}

double TrlarRouting::energyRatio() const
{
    return std::max(0.0, std::min(1.0, sensorNode->getRemainingEnergy() / std::max(1e-9, initialEnergy)));
}

double TrlarRouting::queueRatio() const
{
    if (bufferSize <= 0)
        return 1.0;
    return std::max(0.0, std::min(1.0, static_cast<double>(queue.size()) / static_cast<double>(bufferSize)));
}

double TrlarRouting::neighborDensityRatio() const
{
    int active = 0;
    for (const auto& entry : neighbors) {
        if ((simTime() - entry.second.lastSeen) <= neighborTimeout && entry.second.id != uavIndex)
            active++;
    }
    return std::max(0.0, std::min(1.0, static_cast<double>(active) / 20.0));
}

double TrlarRouting::normalizedTau() const
{
    if (lastUavBeacon < SIMTIME_ZERO)
        return 1.0;

    const simtime_t elapsed = simTime() - lastUavBeacon;
    double remaining = (uavEtaEstimate - elapsed).dbl();
    if (remaining < 0.0)
        remaining = 0.0;
    return std::max(0.0, std::min(1.0, remaining / 60.0));
}

double TrlarRouting::normalizedUavDistance() const
{
    if (!hasUavPosition)
        return 1.0;

    const double d = computeDistance(selfX, selfY, lastUavX, lastUavY);
    return std::max(0.0, std::min(1.0, d / std::max(1e-9, areaDiag)));
}

double TrlarRouting::neighborFailureRatio(const NeighborInfo& neighbor) const
{
    const int total = neighbor.successCount + neighbor.failureCount;
    if (total <= 0)
        return 0.0;
    return static_cast<double>(neighbor.failureCount) / static_cast<double>(total);
}

std::string TrlarRouting::packetIdFor(cMessage *msg) const
{
    if (msg->hasPar("srcIdx") && msg->hasPar("seq")) {
        std::ostringstream oss;
        oss << msg->par("srcIdx").longValue() << "-" << msg->par("seq").longValue();
        return oss.str();
    }

    std::ostringstream fallback;
    fallback << nodeId << "-" << msg->getId();
    return fallback.str();
}

void TrlarRouting::applyCongestionPenalty()
{
    for (auto& entry : neighbors) {
        const NeighborInfo& n = entry.second;
        if ((simTime() - n.lastSeen) > neighborTimeout)
            continue;
        if (neighborFailureRatio(n) > 0.6 || n.lastDelay > delayThreshold)
            biasTable[entry.first] -= 0.5 * congestionPenalty;
    }
}

void TrlarRouting::openTransitionsFile()
{
    if (transitionsPath.empty())
        return;

    std::filesystem::path p(transitionsPath);
    // Avoid concurrent append corruption by sharding transitions per routing node.
    const std::string ext = p.has_extension() ? p.extension().string() : std::string(".jsonl");
    const std::string stem = p.has_stem() ? p.stem().string() : std::string("transitions");
    p = (p.has_parent_path() ? p.parent_path() : std::filesystem::path("."))
        / (stem + "-node" + std::to_string(nodeId) + ext);
    if (p.has_parent_path())
        std::filesystem::create_directories(p.parent_path());

    transitionsStream.open(p, std::ios::out | std::ios::app);
}

void TrlarRouting::logTransition(const std::string& event, const std::string& payload)
{
    if (transitionsPath.empty())
        return;

    if (!transitionsStream.is_open())
        openTransitionsFile();
    if (!transitionsStream.is_open())
        return;

    transitionsStream << "{\"t\":" << simTime().dbl()
                      << ",\"node\":" << nodeId
                      << ",\"event\":\"" << event << "\"";
    if (!payload.empty())
        transitionsStream << "," << payload;
    transitionsStream << "}\n";
}

void TrlarRouting::writeRunMetadata()
{
    if (metadataWritten || metadataPath.empty())
        return;

    std::filesystem::path p(metadataPath);
    if (p.has_parent_path())
        std::filesystem::create_directories(p.parent_path());

    std::ofstream meta(p, std::ios::out | std::ios::trunc);
    if (!meta.is_open())
        return;

    std::ostringstream pool;
    pool << '[';
    for (size_t i = 0; i < policyPool.size(); ++i) {
        if (i)
            pool << ',';
        pool << '"' << jsonEscape(policyPool[i].tag) << '"';
    }
    pool << ']';

    std::ostringstream weights;
    weights << '[';
    for (size_t i = 0; i < policyWeights.size(); ++i) {
        if (i)
            weights << ',';
        weights << policyWeights[i];
    }
    weights << ']';

    const char* gitCommitEnv = std::getenv("UAVWSN_GIT_COMMIT");
    const char* omnetVersionEnv = std::getenv("UAVWSN_OMNET_VERSION");
    const char* configHashEnv = std::getenv("UAVWSN_CONFIG_HASH");

    meta << "{\n";
    meta << "  \"run_number\": " << getEnvir()->getConfigEx()->getActiveRunNumber() << ",\n";
    meta << "  \"config_name\": \"" << jsonEscape(getEnvir()->getConfigEx()->getActiveConfigName()) << "\",\n";
    meta << "  \"network\": \"UavWsnNetwork\",\n";
    meta << "  \"git_commit\": \"" << jsonEscape(gitCommitEnv ? gitCommitEnv : "unknown") << "\",\n";
    meta << "  \"omnetpp_version\": \"" << jsonEscape(omnetVersionEnv ? omnetVersionEnv : "unknown") << "\",\n";
    meta << "  \"config_hash\": \"" << jsonEscape(configHashEnv ? configHashEnv : "unknown") << "\",\n";
    meta << "  \"mobility_tag\": \"" << jsonEscape(mobilityTag) << "\",\n";
    meta << "  \"schema_version\": 1,\n";
    meta << "  \"feature_order\": [\"energy\",\"queue\",\"density\",\"tau\",\"lq\",\"neighEnergy\",\"neighFailRatio\",\"uavDistance\"],\n";
    meta << "  \"episode_policy_mode\": \"" << jsonEscape(episodePolicyMode) << "\",\n";
    meta << "  \"episode_duration\": " << episodeDuration.dbl() << ",\n";
    meta << "  \"policy_pool\": " << pool.str() << ",\n";
    meta << "  \"policy_mix_weights\": " << weights.str() << ",\n";
    meta << "  \"utility_coeff\": {\"lq\": " << utilityCoeffLq
         << ", \"neighEnergy\": " << utilityCoeffNeighEnergy
         << ", \"queue\": " << utilityCoeffQueue
         << ", \"failRatio\": " << utilityCoeffFail
         << ", \"uavDistance\": " << utilityCoeffUavDistance << "},\n";
    meta << "  \"topology\": {\"numNodes\": " << numNodes << ", \"areaX\": " << areaX << ", \"areaY\": " << areaY << "}\n";
    meta << "}\n";
    metadataWritten = true;
}

void TrlarRouting::finish()
{
    if (decisionEvent) {
        cancelAndDelete(decisionEvent);
        decisionEvent = nullptr;
    }

    if (ackTimeoutEvent) {
        if (ackTimeoutEvent->isScheduled())
            cancelEvent(ackTimeoutEvent);
        delete ackTimeoutEvent;
        ackTimeoutEvent = nullptr;
    }

    resetPendingState();

    while (!queue.empty()) {
        delete queue.front();
        queue.pop_front();
    }

    if (transitionsStream.is_open())
        transitionsStream.close();

    recordScalar("queueDrops", queuedDrops);
    recordScalar("forwardedPackets", forwardedPackets);
    recordScalar("successfulForwards", successfulForwards);
    recordScalar("failedForwards", failedForwards);
}

} // namespace trlar
