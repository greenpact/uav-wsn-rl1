#ifndef RL_AGENT_INTERFACE_H
#define RL_AGENT_INTERFACE_H

#include <iosfwd>
#include <unordered_map>
#include <string>
#include <vector>

namespace trlar {

class RLAgentInterface {
  public:
    bool loadModel(const std::string& filePath);
    double scoreAction(const std::vector<double>& features) const;
    bool isLoaded() const { return loaded_; }

  private:
    enum class ModelType {
        NONE,
        MLP,
        QTABLE
    };

    struct Layer {
        int outDim = 0;
        int inDim = 0;
        std::vector<double> weights;
        std::vector<double> bias;
    };

    ModelType modelType_ = ModelType::NONE;
    std::vector<Layer> layers_;
    std::vector<double> normMean_;
    std::vector<double> normStd_;

    std::vector<int> qBins_;
    std::vector<double> qMin_;
    std::vector<double> qMax_;
    double qDefault_ = 0.0;
    std::unordered_map<long long, double> qEntries_;

    bool loaded_ = false;

    static double relu(double value);
    static double clip01(double value);
    static bool parseLayer(std::istream& in, Layer& layer);
    static bool readToken(std::istream& in, const char *expected);

    bool parseMlp(std::istream& in);
    bool parseQTable(std::istream& in);
    long long qKey(const std::vector<int>& index) const;
    double scoreQTable(const std::vector<double>& features) const;

    std::vector<double> applyNetwork(const std::vector<double>& input) const;
    double fallbackHeuristic(const std::vector<double>& features) const;
};

} // namespace trlar

#endif // RL_AGENT_INTERFACE_H
