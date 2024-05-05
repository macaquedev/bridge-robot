#ifndef CONFIG_H
#define CONFIG_H

#include <Arduino.h>
#include <AccelStepper.h>
#include <Servo.h>
#include "pythonCommunication.h"
#include "io.h"

double deg_to_rad = 3.141593 / 180;
double rad_to_deg = 180 / 3.141593;

const int lengths[4] = {104, 99, 34, 30};
const double eta = atan2(lengths[2], lengths[3]) * rad_to_deg;
const double s = sqrt(lengths[2] * lengths[2] + lengths[3] * lengths[3]);

#define ANGLE_INCREMENT 1

#define X_STEPPER_STEP A0
#define X_STEPPER_DIR A1
#define X_STEPPER_ENABLE 38
#define X_MAX_POS 405
#define STEPS_PER_MM 4.935

#define ENDSTOP 3

#define PUMP_PIN 8

const int SERVO_PINS[] = {11, 6, 5, 4};
const int LOW_PWM[] = {570, 385, 620, 450};
const int HIGH_PWM[] = {2455, 2330, 2630, 2390};
const int STARTING_POSITIONS[] = {90, 90, 0, 0};
const int BENT_POSITIONS[] = {90, 45, 125, 10};

class Vector3 {  // thank you Copilot!
public:
  Vector3() {
    x = 0;
    y = 0;
    z = 0;
  }

  Vector3(int x, int y, int z) {
    this->x = x;
    this->y = y;
    this->z = z;
  }

  Vector3 operator+(const Vector3& v) {
    return Vector3(x + v.x, y + v.y, z + v.z);
  }

  Vector3 operator-(const Vector3& v) {
    return Vector3(x - v.x, y - v.y, z - v.z);
  }

  Vector3 operator*(const double& s) {
    return Vector3(x * s, y * s, z * s);
  }

  Vector3 operator/(const double& s) {
    return Vector3(x / s, y / s, z / s);
  }

  double dot(const Vector3& v) {
    return x * v.x + y * v.y + z * v.z;
  }

  Vector3 cross(const Vector3& v) {
    return Vector3(y * v.z - z * v.y, z * v.x - x * v.z, x * v.y - y * v.x);
  }

  double length() {
    return sqrt(x * x + y * y + z * z);
  }

  Vector3 normalize() {
    return *this / length();
  }
  double x;
  double y;
  double z;
};

const Vector3 GRAB_POSITION(0, 139, -50);
const Vector3 PULL_POSITION(0, 135, -50);
const Vector3 LIFT_POSITION(0, 182, 50);
const Vector3 HOME_POSITION(0, 120, -45);
const Vector3 RELEASE_POSITION(0, 180, 0);
const Vector3 RELEASE_POSITION_2(0, 120, 50);
const int RELEASE_ANGLE = -60;
const int GRAB_ANGLE = -25;

class ArmServo
{
    Servo servo;

public:
    int pos;

    ArmServo() : servo()
    {
    }
    void attach(int pin, int lowPwm, int highPwm, int startPos)
    {
        this->servo.attach(pin, lowPwm, highPwm);
        write(startPos);
    }

    void write(int value)
    {
        this->servo.write(value);
        pos = value;
    }
};

class Stepper
{
    AccelStepper stepper;

public:
    double pos;
    Stepper(int step, int dir) : stepper(AccelStepper::DRIVER, step, dir) {}

    void setSpeedAccel(int maxSpeed, int acceleration = 1000)
    {
        this->stepper.setMaxSpeed(maxSpeed);
        this->stepper.setAcceleration(acceleration);
    }

    void bringToEndstop(int endstop)
    {
        this->stepper.setSpeed(100);
        this->stepper.move(-100000);
        while (digitalRead(endstop))
        {
            this->stepper.run();
        }
        this->stepper.move(0);
        this->stepper.setSpeed(0);
        this->stepper.setCurrentPosition(0);
    }

    void config(int enable, int endstop, int maxSpeed, int acceleration)
    {
        pinMode(enable, OUTPUT);
        digitalWrite(enable, LOW);
        setSpeedAccel(maxSpeed, acceleration);
        bringToEndstop(endstop);
    }

    void moveTo(double target)
    {
        double distance = (target - this->pos) * STEPS_PER_MM;
        stepper.move(distance);
    }

    void run()
    {
        stepper.run();
        pos = stepper.currentPosition() / STEPS_PER_MM;
    }

    bool isRunning()
    {
        return stepper.isRunning();
    }

    void stop()
    {
        stepper.stop();
    }
};

PythonCommunication pc;
ArmServo servo[4];
Stepper stepper(X_STEPPER_STEP, X_STEPPER_DIR);

void armDown()
{
    for (int i = 0; i <= 90; i++)
    {
        servo[2].write(i);
        servo[3].write(i);
        servo[0].write(map(i, 0, 90, 90, 120));
        delay(12);
    }

    for (int i = 0; i <= 90; i++)
    {
        servo[1].write(map(i, 0, 90, 90, BENT_POSITIONS[1]));
        servo[2].write(map(i, 0, 90, 90, BENT_POSITIONS[2]));
        servo[3].write(map(i, 0, 90, 90, BENT_POSITIONS[3]));
        delay(12);
    }

    for (int i = 120; i >= BENT_POSITIONS[0]; i--)
    {
        servo[0].write(i);
        delay(6);
    }
}

void armUp()
{
    for (int i = BENT_POSITIONS[0]; i <= 120; i++)
    {
        servo[0].write(i);
        delay(6);
    }

    for (int i = 0; i<=90; i++)
    {
        servo[1].write(map(i, 0, 90, BENT_POSITIONS[1], 90));
        servo[2].write(map(i, 0, 90, BENT_POSITIONS[2], 90));
        servo[3].write(map(i, 0, 90, BENT_POSITIONS[3], 90));
        delay(12);
    }
    for (int i = 0; i <= 90; i++)
    {
        servo[2].write(90-i);
        servo[3].write(90-i);
        servo[0].write(map(i, 0, 90, 120, 90));
        delay(12);
    }
}

void configServos(bool motors = true)
{
    while (digitalRead(ENDSTOP))
        ;
    while (!digitalRead(ENDSTOP))
        ;

    if (motors)
    {
        for (int i = 0; i < 4; i++)
            servo[i].attach(SERVO_PINS[i], LOW_PWM[i], HIGH_PWM[i], STARTING_POSITIONS[i]);
        armDown();
    }
    else
    {
        for (int i = 0; i < 4; i++)
            servo[i].attach(SERVO_PINS[i], LOW_PWM[i], HIGH_PWM[i], BENT_POSITIONS[i]);
    }
}

void config(bool motors = true)
{
    pc.begin();
    pinMode(9, OUTPUT);
    pinMode(PUMP_PIN, OUTPUT);
    pinMode(ENDSTOP, INPUT_PULLUP);
    digitalWrite(9, HIGH);
    configServos(motors);
    if (motors)
        stepper.config(X_STEPPER_ENABLE, ENDSTOP, 1000, 10000);
}

#endif