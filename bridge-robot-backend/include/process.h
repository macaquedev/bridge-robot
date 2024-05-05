#include "config.h"
class BaseProcess
{
public:
    BaseProcess(){};
    virtual void tick() = 0;
    virtual bool done() = 0;
    virtual void complete() = 0;
    virtual ~BaseProcess() {}
};

class ProcessList
{
private:
public:
    BaseProcess *processes[10];
    int processCount = 0;
    ProcessList()
    {
        for (int i = 0; i < 10; i++)
        {
            processes[i] = nullptr;
        }
    }

    void addProcess(BaseProcess *process)
    {
        processes[processCount++] = process;
    }

    void tick()
    {

        for (int i = 0; i < processCount; i++)
        {
            if (processes[i] == nullptr)
                continue;
            processes[i]->tick();
            if (processes[i]->done())
            {
                processes[i]->complete();
                delete processes[i];
                processes[i] = nullptr;
            }
        }
    }

    void clear()
    {
        for (int i = 0; i < processCount; i++)
        {
            delete processes[i];
            processes[i] = nullptr;
        }
        processCount = 0;
    }

    bool done()
    {
        for (int i = 0; i < processCount; i++)
        {
            if (processes[i] != nullptr and !processes[i]->done())
            {
                return false;
            }
        }
        return true;
    }

    ~ProcessList()
    {
        clear(); // Delete all ServoProcess objects
        // Continue with any other necessary cleanup...
    }
};

class ServoProcess : public BaseProcess
{
private:
    ArmServo &servoObj;
    double dt;
    double angleIncrement = 2;
    double targetAngle;
    int num;
    unsigned long long lastMoved;

public:
    ServoProcess(ArmServo &servoObj, int position, int time, int num) : servoObj(servoObj)
    {
        update(position, time);
        this->num = num;
    }

    void update(int position, double time)
    {
        this->targetAngle = position;
        double changeInAngle = abs(this->targetAngle - this->servoObj.pos);
        this->angleIncrement = ANGLE_INCREMENT * (this->targetAngle > this->servoObj.pos ? 1 : -1);
        this->dt = time * (abs(angleIncrement) * 1000 / changeInAngle);
        this->lastMoved = micros();
    }

    void tick() override
    {
        unsigned long long t = micros();
        if (t - this->lastMoved < dt)
            return;
        this->lastMoved = t;
        if (abs(this->targetAngle - this->servoObj.pos) >= abs(angleIncrement))
        {
            this->servoObj.write(this->servoObj.pos + angleIncrement);
        }
        else
        {
            this->servoObj.write(this->targetAngle);
        }
    }

    bool done() override
    {
        return abs(this->targetAngle - this->servoObj.pos) < 0.01;
    }

    void complete() override
    {
    }
};

class StepperProcess : public BaseProcess
{
private:
    Stepper &stepperObj;

public:
    StepperProcess(Stepper &stepperObj)
        : stepperObj(stepperObj)
    {
    }
    StepperProcess(Stepper &stepperObj, double target)
        : stepperObj(stepperObj)
    {
        target = max(0, target);
        target = min(target, X_MAX_POS);
        update(target);
    }

    void tick() override
    {
        this->stepperObj.run();
    }

    bool done() override
    {
        return !this->stepperObj.isRunning();
    }

    void update(double new_pos) {
        this->stepperObj.moveTo(new_pos);
    }

    void complete() override
    {
    }
};