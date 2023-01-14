int ledpin = 10;
String str;
int relaypin = 11;
void setup() {
  // put your setup code here, to run once:
  Serial.begin(115200);
  pinMode(ledpin,OUTPUT);
  pinMode(relaypin,OUTPUT);

}

void loop() {
  // put your main code here, to run repeatedly:
  if (Serial.available() > 0) {
                str = Serial.readStringUntil('\n');  
             
                if (str == "turn on led"){
               
                  digitalWrite(ledpin,HIGH);
                }

                 if (str == "turn off led"){
                  
                  digitalWrite(ledpin,LOW);
                }
                if (str =="motor on"){
                  digitalWrite(relaypin,HIGH);
                   
                   }
                 if(str="motor off"){
                  digitalWrite(relaypin,LOW);
                  }
        }
}
