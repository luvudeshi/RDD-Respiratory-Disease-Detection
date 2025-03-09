#define SAMPLES 128
#define SAMPLING_FREQUENCY 8000

int audioBuffer[SAMPLES];
unsigned long previousMillis = 0;
const long interval = 500;
const int MIC_PIN = A0;
const int GAIN_PIN = 5;
const int THRESHOLD = 10;
int maxValue = 0;
int minValue = 1023;
int avgValue = 0;

void setup() {
  Serial.begin(115200);
  if (GAIN_PIN > 0) {
    pinMode(GAIN_PIN, OUTPUT);
    digitalWrite(GAIN_PIN, LOW);
  }
  while (!Serial) {
    ;
  }
  Serial.println("ARDUINO_READY");
  Serial.println("MAX9814 Microphone Test");
}

void loop() {
  unsigned long currentMillis = millis();
  if (currentMillis - previousMillis >= interval) {
    previousMillis = currentMillis;
    sampleAudioImproved();
    sendAudioData();
  }
  checkCommands();
}

void sampleAudioImproved() {
  maxValue = 0;
  minValue = 1023;
  long sum = 0;
  for (int i = 0; i < 10; i++) {
    sum += analogRead(MIC_PIN);
    delayMicroseconds(100);
  }
  int baseline = sum / 10;
  for (int i = 0; i < SAMPLES; i++) {
    sum = 0;
    for (int j = 0; j < 3; j++) {
      sum += analogRead(MIC_PIN);
      delayMicroseconds(10);
    }
    int reading = sum / 3;
    if (reading > maxValue) maxValue = reading;
    if (reading < minValue) minValue = reading;
    audioBuffer[i] = reading;
    delayMicroseconds(85);
  }
  sum = 0;
  for (int i = 0; i < SAMPLES; i++) {
    sum += audioBuffer[i];
  }
  avgValue = sum / SAMPLES;
  int range = maxValue - minValue;
  if (range < THRESHOLD) {
    for (int i = 0; i < SAMPLES; i++) {
      int centered = audioBuffer[i] - baseline;
      centered = centered * (50 / max(1, range));
      audioBuffer[i] = constrain(centered + 512, 0, 1023);
    }
    maxValue = 0;
    minValue = 1023;
    for (int i = 0; i < SAMPLES; i++) {
      if (audioBuffer[i] > maxValue) maxValue = audioBuffer[i];
      if (audioBuffer[i] < minValue) minValue = audioBuffer[i];
    }
  }
  for (int i = 0; i < SAMPLES; i++) {
    audioBuffer[i] += random(-5, 6);
    audioBuffer[i] = constrain(audioBuffer[i], 0, 1023);
  }
}

void sendAudioData() {
  Serial.println("START_DATA");
  Serial.print("TIME:");
  Serial.println(millis());
  Serial.print("STATS:");
  Serial.print(minValue);
  Serial.print(",");
  Serial.print(maxValue);
  Serial.print(",");
  Serial.print(avgValue);
  Serial.print(",");
  Serial.println(maxValue - minValue);
  for (int i = 0; i < SAMPLES; i++) {
    Serial.println(audioBuffer[i]);
    delayMicroseconds(50);
  }
  Serial.println("END_DATA");
}

void checkCommands() {
  if (Serial.available() > 0) {
    String command = Serial.readStringUntil('\n');
    if (command == "PING") {
      Serial.println("PONG");
    }
  }
}
