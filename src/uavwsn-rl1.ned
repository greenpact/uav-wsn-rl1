package src;

// Sample-based simple NED (cleaned and fixed for this project)

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
    gates:
        input in;
        output out;
}
@display("i=samples/inet4.5/images/misc/sensor");

simple RadioMedium
{
    parameters:
        int numSensors = default(100);
        double areaX = default(1000.0);
        double areaY = default(1000.0);
        double roundDuration = default(200.0);
    gates:
        input in[numSensors+2];
        output out[numSensors+2];
}
@display("i=block/broadcast");

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
            @display("p=200,200;i=block/broadcast");
        }
        bs: SensorNode {
            parameters:
                initialX = -200;
                initialY = 500;
                has80211 = true;
                has802154 = false;
                range11 = 300;
            @display("p=50,100;i=images/device/antennatower");
        }
        uav: src.uav.UavModule {
            parameters:
                initialX = -200;
                initialY = 500;
                commRadius = 192;
                beaconInterval = 5.0;
            @display("p=100,100;i=device/airplane");
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
        routing[numNodes]: src.routing.TrlarRouting {
            parameters:
                nodeId = index;
                modelFile = "";
            @display("i=block/routing");
        }
    connections allowunconnected:
        for i=0..numNodes-1 {
            // sensor sends to routing, routing forwards to medium
            sensors[i].out --> routing[i].inFromSensor;
            routing[i].outToMedium --> radioMedium.in[i];
            // medium sends to routing, routing forwards to sensor
            radioMedium.out[i] --> routing[i].inFromMedium;
            routing[i].outToSensor --> sensors[i].in;
        }
        uav.out --> radioMedium.in[numNodes];
        radioMedium.out[numNodes] --> uav.in;

        bs.out --> radioMedium.in[numNodes+1];
        radioMedium.out[numNodes+1] --> bs.in;
}
