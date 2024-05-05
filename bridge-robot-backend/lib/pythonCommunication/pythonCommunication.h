#ifndef PYTHON_COMMUNICATION_H
#define PYTHON_COMMUNICATION_H

#include <Arduino.h>

class PythonCommunication {
    public:
    PythonCommunication();
    void begin();
    void readData(String *data);
    void waitForData();
    void sendData(String data);
    void sendAcknowledgement();
    void waitForAcknowledgement();
    void sendError(String data);
};

#endif 