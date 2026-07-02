# House Price Prediction Model

This repository contains a PyTorch-based neural network model for predicting house prices using the California Housing dataset.

## Project Structure
- `house_prices.py`: The main script that loads data, trains the model, and evaluates it.
- `house_prices.csv`: The dataset used for training the model.
- `Dockerfile`: Configuration file to build and run the project inside a Docker container.
- `requirements.txt`: Python dependencies needed to run the project natively.

## How to Run

### Using Docker (Recommended)
1. Build the Docker image:
   ```bash
   docker build -t house-price-project .
   ```
2. Run the container:
   ```bash
   docker run --rm -v ${PWD}:/app house-price-project
   ```
This will train the model and save a plot named `output.png` into your local directory.

### Running Locally
If you prefer not to use Docker, you can run the code natively:
1. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```
2. Run the training script:
   ```bash
   python house_prices.py
   ```

## Output
Upon successful execution, the script evaluates the model's correlation and saves `output.png`, which plots the Training Loss Decay and the Actual vs. Predicted Prices.
