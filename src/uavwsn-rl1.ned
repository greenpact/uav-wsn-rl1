package src;

// First-order radio model: SensorNode/UAV/BaseStation are simple modules
// RadioMedium is a simple module that forwards messages to nodes within range.

simple RadioMedium
{
    parameters:
        int numSensors;
        double areaX = 1000;
        double areaY = 1000;
    gates:
        input in[numSensors+2];
        output out[numSensors+2];
}
@display("i=block/radio");

simple SensorNode
{
    parameters:
        double initialX;
        double initialY;
        double range;
    gates:
        input in;
        output out;
}
@display("i=device/sensor");

// UAVNode and BaseStation use the same simple SensorNode type (handled in network)

network UAVWSN_RL1
{
    parameters:
        int numSensors = 400;
        double areaX = 1000;
        double areaY = 1000;
    submodules:
        radioMedium: RadioMedium {
            parameters:
                numSensors = parent.numSensors;
            @display("p=200,200;i=block/radio");
        }

        sensors[numSensors]: SensorNode {
            parameters:
                initialX = uniform(0, parent.areaX);
                initialY = uniform(0, parent.areaY);
                range = 100;
            @display("i=device/sensor");
        }

        uav: SensorNode {
            parameters:
                initialX = -200;
                initialY = 500;
                range = 150;
            @display("p=100,100;i=device/airplane");
        }

        bs: SensorNode {
            parameters:
                initialX = -200;
                initialY = 500;
                range = 100;
            @display("p=50,100;i=device/antenna");
        }

    connections allowunconnected:
        for i=0..numSensors-1 {
            sensors[i].out --> radioMedium.in[i];
            radioMedium.out[i] --> sensors[i].in;
        }
        uav.out --> radioMedium.in[numSensors];
        radioMedium.out[numSensors] --> uav.in;

        bs.out --> radioMedium.in[numSensors+1];
        radioMedium.out[numSensors+1] --> bs.in;
}
