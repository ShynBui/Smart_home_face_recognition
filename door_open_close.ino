#include <Servo.h>

/*
90 -> đóng
0 -> mở
*/

Servo servoMotor; 
const int servoPin = 9;

void setup() {
  Serial.begin(9600);  // Kết nối serial
  servoMotor.attach(servoPin);  // Kết nối servo vào chân số 9
  servoMotor.write(90);  // Đặt cửa vào trạng thái đóng (90 độ)
  delay(500);  // Thời gian chờ để servo vào vị trí ban đầu
}

void loop() {
  if (Serial.available() > 0) {
    char command = Serial.read();  // Đọc lệnh từ cổng serial
    if (command == 'O') {  // Nếu lệnh là 'O', mở cửa

      // Gửi lệnh đóng lại để khởi động servo, giúp tránh giật nhẹ
      servoMotor.write(90);  // Đảm bảo servo đã ở vị trí 90 độ
      delay(100);  // Chờ 100ms để ổn định servo

      // Lệnh mở cửa thực sự
      servoMotor.write(0);  // Xoay servo về 0 độ để mở cửa
      delay(4000);  // Giữ cửa mở trong 3 giây
      servoMotor.write(90);  // Đóng cửa lại về góc 90 độ
    }
  }
}
