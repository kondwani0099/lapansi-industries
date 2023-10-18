
# Lapansi Industries

Lapansi Industries is a project that combines Arduino and Python to control LED and motor devices using voice commands. With this project, you can use voice input to send commands to an Arduino board via Python, allowing you to control devices like LEDs and motors with ease.

## Table of Contents

- [Features](#features)
- [Requirements](#requirements)
- [Getting Started](#getting-started)
- [Usage](#usage)
- [Contributing](#contributing)
- [License](#license)

## Features

- Voice-controlled Arduino devices.
- Switch LED lights on and off.
- Control a motor's rotation using voice commands.

## Requirements

- Arduino board.
- Python (3.x recommended).
- Required Python libraries (e.g., pyserial for serial communication).
- Microphone (for voice input).

## Getting Started

1. Clone the repository to your local machine:

   ```bash
   git clone https://github.com/kondwani0099/lapansi-industries.git
   ```

2. Install the required Python libraries:

   ```bash
   pip install -r requirements.txt
   ```

3. Connect your Arduino board to your computer.

4. Upload the Arduino sketch (provided in the project) to your Arduino board.

5. Run the Python script to start the voice-controlled device.

## Usage

1. Start the Python script:

   ```bash
   python main.py
   ```

2. Follow the voice commands to control the LED and motor.

3. For example:
   - Saying "Turn on the light" should turn on the LED.
   - Saying "Rotate the motor" should control the motor's rotation.

4. Enjoy controlling your devices using voice commands!

## Contributing

If you'd like to contribute to this project, please follow these steps:

1. Fork the repository.
2. Create a new branch for your feature: `git checkout -b feature-name`.
3. Make your changes and commit them: `git commit -m 'Add new feature'`.
4. Push to your branch: `git push origin feature-name`.
5. Create a pull request to the `main` branch of this repository.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

