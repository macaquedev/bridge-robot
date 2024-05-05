#include "config.h"
#include "ik.h"

void blink(int t, int n) {
  for (int i=0; i<n; i++) {
    digitalWrite(13, HIGH);
    delay(t);
    digitalWrite(13, LOW);
    delay(t);
  }
}

void setup()
{
  pinMode(13, OUTPUT);
  config();
}

void loop()
{
  String pcData;
  while (true) {
    pc.waitForData();
    pc.readData(&pcData);
    if (pcData.startsWith("STEPPER")) {
      int steps = pcData.substring(7).toInt();
      StepperProcess p(stepper, steps);
      while (!p.done())
      {
        p.tick();
      }
      pc.sendAcknowledgement();
    } else if (pcData.startsWith("MOVE")) {
      String tokens[4];
      String delimiters = "\r ";
      pcData = pcData.substring(4);
      splitStringIntoTokens(&pcData, 4, &delimiters, tokens);
      int x = tokens[0].toInt();
      int y = tokens[1].toInt();
      int z = tokens[2].toInt();
      int a = tokens[3].toInt();
      ProcessList p;
      moveArmToPos(p, Vector3(x, y, z), a, 1000);
      while (!p.done())
      {
        p.tick();
      }
      pc.sendAcknowledgement();
    } else if (pcData.startsWith("BLINK")) {
      String tokens[2];
      String delimiters = "\r ";
      pcData = pcData.substring(5);
      splitStringIntoTokens(&pcData, 2, &delimiters, tokens);
      int t = tokens[0].toInt();
      int n = tokens[1].toInt();
      blink(t, n);
      pc.sendAcknowledgement();
    } else if (pcData.startsWith("SUCK")) {
      int on = pcData.substring(4).toInt();
      if (on) {
        suctionOn();
      } else {
        suctionOff();
      }
      pc.sendAcknowledgement();
    } else if (pcData.startsWith("ARMUP")) {
      armUp();
      pc.sendAcknowledgement();
    } else if (pcData.startsWith("ARMDOWN")) {
      armDown();
      pc.sendAcknowledgement();
    } else if (pcData.startsWith("GRAB")) {
      ProcessList p;
      moveArmToPos(p, GRAB_POSITION, GRAB_ANGLE, 1000);
      while (!p.done())
      {
        p.tick();
      }
      suctionOn();
      moveArmToPos(p, PULL_POSITION, GRAB_ANGLE, 100);
      while (!p.done())
      {
        p.tick();
      }
      for (int curr_step = 0; curr_step < 5; curr_step++) {
        int x = map(curr_step, 0, 4, PULL_POSITION.x, LIFT_POSITION.x);
        int y = map(curr_step, 0, 4, PULL_POSITION.y, LIFT_POSITION.y);
        int z = map(curr_step, 0, 4, PULL_POSITION.z, LIFT_POSITION.z);
        moveArmToPos(p, Vector3(x, y, z), GRAB_ANGLE, 150);
        while (!p.done())
        {
          p.tick();
        }
      }
      pc.sendAcknowledgement();
    } else if (pcData.startsWith("RELEASE")) {
      ProcessList p;
      moveArmToPos(p, RELEASE_POSITION, RELEASE_ANGLE, 1000);
      while (!p.done())
      {
        p.tick();
      }
      suctionOff();
      moveArmToPos(p, RELEASE_POSITION_2, 0, 1000);
      while (!p.done())
      {
        p.tick();
      }
      moveArmToPos(p, HOME_POSITION, 0, 1000);
      while (!p.done())
      {
        p.tick();
      }
      pc.sendAcknowledgement();
    } else {
      pc.sendError(F("Invalid command"));
    }
  }
}

/*
else if (pcData.startsWith("GRAB")) {
      ProcessList p;
      moveArmToPos(p, GRAB_POSITION, GRAB_ANGLE, 1000);
      while (!p.done())
      {
        p.tick();
      }
      suctionOn();
      moveArmToPos(p, PULL_POSITION, GRAB_ANGLE, 100);
      while (!p.done())
      {
        p.tick();
      }
      StepperProcess p2(stepper, stepper.pos - 50);
      while (!p2.done())
      {
        p2.tick();
      }
      moveArmToPos(p, LIFT_POSITION, GRAB_ANGLE, 1000);
      while (!p.done())
      {
        p.tick();
      }
      pc.sendAcknowledgement();
    } else if (pcData.startsWith("RELEASE")) {
      ProcessList p;
      moveArmToPos(p, RELEASE_POSITION, RELEASE_ANGLE, 1000);
      while (!p.done())
      {
        p.tick();
      }
      suctionOff();
      moveArmToPos(p, LIFT_POSITION, GRAB_ANGLE, 1000);
      while (!p.done())
      {
        p.tick();
      }
      moveArmToPos(p, HOME_POSITION, 0, 1000);
      while (!p.done())
      {
        p.tick();
      }
      pc.sendAcknowledgement();
    }*/