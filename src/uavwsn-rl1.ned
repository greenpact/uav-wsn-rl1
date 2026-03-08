package src;

simple SensorNode
{
    parameters:
        double initialX = default(0.0);
        double initialY = default(0.0);
        double initialEnergy = default(0.5);
        double controlPacketSize = default(200);
        double dataPacketSize = default(2000);
        double eElec = default(50e-9);
        double eFreeSpace = default(10e-12);
        double eMultiPath = default(0.0013e-12);
        double idleListeningPower = default(0.0);
        bool has80211 = default(false);
        bool has802154 = default(true);
        double range154 = default(100.0);
        double range11 = default(150.0);
        double beaconInterval = default(1.0);
        double dataInterval = default(1.0);
        bool enableBeaconing = default(true);
        bool enableDataGeneration = default(true);
        bool isBaseStation = default(false);
        @display("i=samples/inet4.5/images/misc/sensor");
    gates:
        input in;
        output out;
}

simple TrlarRouting
{
    parameters:
        @class(trlar::TrlarRouting);
        int nodeId = default(0);
        string modelFile = default("");
        string transitionsFile = default("results/transitions.jsonl");
        int bufferSize = default(32);
        int maxCandidates = default(5);
        int maxRetries = default(1);
        int maxRouteAttempts = default(6);
        int maxHops = default(30);
        double minLinkQuality = default(0.25);
        double minNeighborEnergyRatio = default(0.05);
        double adaptationAlpha = default(0.2);
        double queueHighWatermark = default(0.8);
        double congestionPenalty = default(0.15);
        double delayThreshold = default(0.25);
        double rewardW1 = default(1.0);
        double rewardW2 = default(1.0);
        double rewardW3 = default(0.2);
        double rewardW4 = default(1.0);
        double ackTimeout = default(0.6);
        double neighborTimeout = default(5.0);
        double decisionInterval = default(0.02);
        string policyPool = default("eps_0.10,softmax_0.50,eps_0.20,greedy");
        string policyMixWeights = default("1,1,1,1");
        string episodePolicyMode = default("random_stratified");
        double episodeDuration = default(280.0);
        double utilityCoeffLq = default(1.7);
        double utilityCoeffNeighEnergy = default(0.9);
        double utilityCoeffQueue = default(1.1);
        double utilityCoeffFail = default(1.0);
        double utilityCoeffUavDistance = default(0.35);
        string mobilityTag = default("ggm-base");
        @display("i=block/routing");
    gates:
        input inFromSensor;
        input inFromMedium;
        output outToMedium;
        output outToSensor;
}

simple UavModule
{
    parameters:
        @class(uav::UavModule);
        bool has80211 = default(true);
        bool has802154 = default(true);
        double initialX = default(-200.0);
        double initialY = default(500.0);
        double currentX @mutable = default(-200.0);
        double currentY @mutable = default(500.0);
        double areaX = default(1000.0);
        double areaY = default(1000.0);
        double commRadius = default(192.0);
        double uploadRadius = default(280.0);
        double beaconInterval = default(5.0);
        double positionUpdateInterval = default(0.5);
        double uploadInterval = default(0.5);
        double minSpeed = default(10.0);
        double maxSpeed = default(10.0);
        double missionPeriod = default(280.0);
        double missionDuration = default(-1.0);
        double contactWindow = default(8.0);
        @display("i=device/airplane");
    gates:
        input in;
        output out;
}

simple RadioMedium
{
    parameters:
        int numSensors = default(100);
        double areaX = default(1000.0);
        double areaY = default(1000.0);
        double roundDuration = default(200.0);
        @display("i=block/broadcast");
    gates:
        input in[numSensors+2];
        output out[numSensors+2];
}

network UavWsnNetwork
{
    parameters:
        int numNodes = default(400);
        double areaX = default(1000.0);
        double areaY = default(1000.0);
    submodules:
        radioMedium: RadioMedium {
            parameters:
                numSensors = parent.numNodes;
                areaX = parent.areaX;
                areaY = parent.areaY;
            @display("p=200,200");
        }
        bs: SensorNode {
            parameters:
                initialX = -200;
                initialY = 500;
                has80211 = true;
                has802154 = false;
                range11 = 300;
                enableBeaconing = false;
                enableDataGeneration = false;
                isBaseStation = true;
            @display("p=50,100;i=device/antennatower");
        }
        uav: UavModule {
            parameters:
                initialX = -200;
                initialY = 500;
                areaX = parent.areaX;
                areaY = parent.areaY;
                commRadius = 192;
                beaconInterval = 5.0;
            @display("p=100,100");
        }
        sensors[numNodes]: SensorNode {
            parameters:
                initialX = uniform(0, parent.areaX);
                initialY = uniform(0, parent.areaY);
                has802154 = true;
                has80211 = false;
                range154 = 100;
            @display("p=${initialX},${initialY}");
        }
        routing[numNodes]: TrlarRouting {
            parameters:
                nodeId = index;
            @display("i=block/routing");
        }
    connections allowunconnected:
        for i=0..numNodes-1 {
            sensors[i].out --> routing[i].inFromSensor;
            routing[i].outToMedium --> radioMedium.in[i];
            radioMedium.out[i] --> routing[i].inFromMedium;
            routing[i].outToSensor --> sensors[i].in;
        }

        uav.out --> radioMedium.in[numNodes];
        radioMedium.out[numNodes] --> uav.in;

        bs.out --> radioMedium.in[numNodes+1];
        radioMedium.out[numNodes+1] --> bs.in;
}
