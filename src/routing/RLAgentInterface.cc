#include "RLAgentInterface.h"

#include <algorithm>
#include <cmath>
#include <fstream>
#include <limits>
#include <sstream>

namespace trlar {

double RLAgentInterface::relu(double value)
{
    return value > 0.0 ? value : 0.0;
}

double RLAgentInterface::clip01(double value)
{
    return std::max(0.0, std::min(1.0, value));
}

bool RLAgentInterface::parseLayer(std::istream& in, Layer& layer)
{
    if (!(in >> layer.outDim >> layer.inDim))
        return false;
    if (layer.outDim <= 0 || layer.inDim <= 0)
        return false;

    layer.weights.resize(static_cast<size_t>(layer.outDim * layer.inDim));
    layer.bias.resize(static_cast<size_t>(layer.outDim));

    std::string token;
    if (!(in >> token) || token != "weights")
        return false;
    for (double& w : layer.weights) {
        if (!(in >> w))
            return false;
    }

    if (!(in >> token) || token != "bias")
        return false;
    for (double& b : layer.bias) {
        if (!(in >> b))
            return false;
    }

    return true;
}

bool RLAgentInterface::readToken(std::istream& in, const char *expected)
{
    std::string token;
    return (in >> token) && token == expected;
}

bool RLAgentInterface::parseMlp(std::istream& in)
{
    int numLayers = 0;
    if (!(in >> numLayers) || numLayers <= 0)
        return false;

    std::vector<Layer> parsedLayers;
    parsedLayers.reserve(static_cast<size_t>(numLayers));

    for (int i = 0; i < numLayers; ++i) {
        if (!readToken(in, "layer"))
            return false;
        Layer layer;
        if (!parseLayer(in, layer))
            return false;
        if (!parsedLayers.empty() && parsedLayers.back().outDim != layer.inDim)
            return false;
        parsedLayers.push_back(std::move(layer));
    }

    std::string token;
    if (in >> token) {
        if (token != "norm")
            return false;

        int n = 0;
        if (!(in >> n) || n <= 0)
            return false;
        normMean_.resize(static_cast<size_t>(n));
        normStd_.resize(static_cast<size_t>(n));

        if (!readToken(in, "mean"))
            return false;
        for (double& v : normMean_) {
            if (!(in >> v))
                return false;
        }

        if (!readToken(in, "std"))
            return false;
        for (double& v : normStd_) {
            if (!(in >> v))
                return false;
            if (std::fabs(v) < 1e-12)
                v = 1.0;
        }
    }

    layers_ = std::move(parsedLayers);
    modelType_ = ModelType::MLP;
    return !layers_.empty();
}

bool RLAgentInterface::parseQTable(std::istream& in)
{
    if (!readToken(in, "dims"))
        return false;

    int dims = 0;
    if (!(in >> dims) || dims <= 0)
        return false;

    if (!readToken(in, "bins"))
        return false;
    qBins_.resize(static_cast<size_t>(dims));
    for (int& b : qBins_) {
        if (!(in >> b) || b < 2)
            return false;
    }

    std::string token;
    if (!(in >> token) || (token != "state_min" && token != "stateMin"))
        return false;
    qMin_.resize(static_cast<size_t>(dims));
    for (double& v : qMin_) {
        if (!(in >> v))
            return false;
    }

    if (!(in >> token) || (token != "state_max" && token != "stateMax"))
        return false;
    qMax_.resize(static_cast<size_t>(dims));
    for (int i = 0; i < dims; ++i) {
        if (!(in >> qMax_[static_cast<size_t>(i)]))
            return false;
        if (qMax_[static_cast<size_t>(i)] <= qMin_[static_cast<size_t>(i)])
            qMax_[static_cast<size_t>(i)] = qMin_[static_cast<size_t>(i)] + 1.0;
    }

    if (!readToken(in, "default"))
        return false;
    if (!(in >> qDefault_) || !std::isfinite(qDefault_))
        return false;

    if (!readToken(in, "entries"))
        return false;
    long long numEntries = 0;
    if (!(in >> numEntries) || numEntries < 0)
        return false;

    qEntries_.clear();
    qEntries_.reserve(static_cast<size_t>(numEntries));

    std::vector<int> index(static_cast<size_t>(dims), 0);
    for (long long e = 0; e < numEntries; ++e) {
        for (int i = 0; i < dims; ++i) {
            if (!(in >> index[static_cast<size_t>(i)]))
                return false;
            if (index[static_cast<size_t>(i)] < 0 || index[static_cast<size_t>(i)] >= qBins_[static_cast<size_t>(i)])
                return false;
        }

        double q = 0.0;
        if (!(in >> q) || !std::isfinite(q))
            return false;
        qEntries_[qKey(index)] = q;
    }

    modelType_ = ModelType::QTABLE;
    return !qBins_.empty();
}

long long RLAgentInterface::qKey(const std::vector<int>& index) const
{
    long long key = 0;
    long long mul = 1;
    for (size_t i = 0; i < index.size(); ++i) {
        key += static_cast<long long>(index[i]) * mul;
        mul *= static_cast<long long>(qBins_[i]);
    }
    return key;
}

double RLAgentInterface::scoreQTable(const std::vector<double>& features) const
{
    if (features.size() != qBins_.size() || qBins_.empty())
        return fallbackHeuristic(features);

    std::vector<int> index(features.size(), 0);
    for (size_t i = 0; i < features.size(); ++i) {
        const double lo = qMin_[i];
        const double hi = qMax_[i];
        const int bins = qBins_[i];

        if (hi <= lo + std::numeric_limits<double>::epsilon()) {
            index[i] = 0;
            continue;
        }

        const double clamped = std::max(lo, std::min(hi, features[i]));
        double ratio = (clamped - lo) / (hi - lo);
        if (ratio < 0.0)
            ratio = 0.0;
        if (ratio > 1.0)
            ratio = 1.0;

        int bin = static_cast<int>(std::floor(ratio * bins));
        if (bin >= bins)
            bin = bins - 1;
        if (bin < 0)
            bin = 0;
        index[i] = bin;
    }

    const auto it = qEntries_.find(qKey(index));
    return (it != qEntries_.end()) ? it->second : qDefault_;
}

bool RLAgentInterface::loadModel(const std::string& filePath)
{
    loaded_ = false;
    modelType_ = ModelType::NONE;

    layers_.clear();
    normMean_.clear();
    normStd_.clear();
    qBins_.clear();
    qMin_.clear();
    qMax_.clear();
    qEntries_.clear();
    qDefault_ = 0.0;

    std::ifstream file(filePath);
    if (!file.is_open())
        return false;

    // Strip comments to keep a tiny parser dependency-free.
    std::stringstream cleaned;
    std::string line;
    while (std::getline(file, line)) {
        const std::size_t commentPos = line.find('#');
        if (commentPos != std::string::npos)
            line = line.substr(0, commentPos);
        cleaned << line << '\n';
    }

    std::string token;
    if (!(cleaned >> token))
        return false;

    bool ok = false;
    if (token == "layers") {
        ok = parseMlp(cleaned);
    }
    else if (token == "format") {
        std::string format;
        if (!(cleaned >> format))
            return false;
        if (format == "qtable_v1")
            ok = parseQTable(cleaned);
        else
            return false;
    }
    else if (token == "qtable_v1") {
        ok = parseQTable(cleaned);
    }
    else {
        return false;
    }

    loaded_ = ok;
    return loaded_;
}

std::vector<double> RLAgentInterface::applyNetwork(const std::vector<double>& input) const
{
    if (layers_.empty())
        return std::vector<double>{0.0};

    std::vector<double> activations = input;

    if (!normMean_.empty() && normMean_.size() == activations.size() && normStd_.size() == activations.size()) {
        for (size_t i = 0; i < activations.size(); ++i)
            activations[i] = (activations[i] - normMean_[i]) / normStd_[i];
    }

    for (size_t layerIdx = 0; layerIdx < layers_.size(); ++layerIdx) {
        const Layer& layer = layers_[layerIdx];
        std::vector<double> out(static_cast<size_t>(layer.outDim), 0.0);
        for (int o = 0; o < layer.outDim; ++o) {
            double sum = layer.bias[static_cast<size_t>(o)];
            for (int i = 0; i < layer.inDim; ++i)
                sum += layer.weights[static_cast<size_t>(o * layer.inDim + i)] * activations[static_cast<size_t>(i)];
            if (layerIdx + 1 < layers_.size())
                sum = relu(sum);
            out[static_cast<size_t>(o)] = sum;
        }
        activations.swap(out);
    }

    return activations;
}

double RLAgentInterface::fallbackHeuristic(const std::vector<double>& features) const
{
    // Features: [energy, queue, density, tau, lq, neighEnergy, neighFailRatio]
    if (features.size() < 7)
        return 0.0;

    const double energy = clip01(features[0]);
    const double queue = clip01(features[1]);
    const double density = clip01(features[2]);
    const double tau = clip01(features[3]);
    const double lq = clip01(features[4]);
    const double neighEnergy = clip01(features[5]);
    const double failRatio = clip01(features[6]);

    return (1.7 * lq) + (0.9 * neighEnergy) + (0.3 * energy) + (0.2 * density)
         - (1.1 * queue) - (0.7 * tau) - (1.0 * failRatio);
}

double RLAgentInterface::scoreAction(const std::vector<double>& features) const
{
    if (!loaded_)
        return fallbackHeuristic(features);

    if (modelType_ == ModelType::QTABLE)
        return scoreQTable(features);

    if (modelType_ != ModelType::MLP || layers_.empty())
        return fallbackHeuristic(features);

    if (features.size() != static_cast<size_t>(layers_.front().inDim))
        return fallbackHeuristic(features);

    const std::vector<double> out = applyNetwork(features);
    if (out.empty())
        return fallbackHeuristic(features);
    return out.front();
}

} // namespace trlar
