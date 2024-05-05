#include "process.h"


void suctionOn() {
  digitalWrite(PUMP_PIN, HIGH);
  delay(1000);
}

void suctionOff() {
  digitalWrite(PUMP_PIN, LOW);
  delay(1500);
}

void moveArmToPos(ProcessList& processList, Vector3 pos, double angle, int time=1000, bool clearProcess=true)
{

  double ox = pos.x;
  double oy = pos.y;
  double oz = pos.z;
  double w = sqrt(ox * ox + oy * oy);
  double theta = atan2(oy, ox) * rad_to_deg;
  double psi = 90 - angle - eta;
  w -= s * cos(psi * deg_to_rad);
  double z = oz + s * sin(psi * deg_to_rad);
  
  double phi = atan2(z, w) * rad_to_deg;

  double alpha = acos((lengths[0] * lengths[0] + w * w + z * z - lengths[1] * lengths[1]) / (2 * sqrt(w*w + z*z) * lengths[0]));
  double beta = acos((lengths[0] * lengths[0] + lengths[1] * lengths[1] - w * w - z * z) / (2 * lengths[0] * lengths[1])) * rad_to_deg;
  
  alpha = alpha * rad_to_deg;
  if (clearProcess) processList.clear();
  processList.addProcess(new ServoProcess(servo[0], theta, time, 0));
  processList.addProcess(new ServoProcess(servo[1], alpha + phi, time, 1));
  processList.addProcess(new ServoProcess(servo[2], 180 - beta, time, 2));
  processList.addProcess(new ServoProcess(servo[3], alpha + beta + phi - 90 - angle, time, 3));
}