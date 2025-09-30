# Go Neural Network

This project implements a feed-forward neural network in Go, designed to be
flexible and adaptable to various datasets. It currently supports training new
models and loading pre-trained models for prediction.

## Features

* **Modular Design:** Code is organized into separate packages (`cli`, `data`,
`neuralnetwork`, `utils`) for better maintainability and reusability.
* **Dynamic Network Architecture:** A feed-forward neural network with a
configurable number of hidden layers and neurons per layer.
* **Multiple Activation Functions:** Supports `ReLU`, `Sigmoid`, `Tanh`,
and `Linear` activation functions for each hidden layer and the output layer.
* **Training:** Train the neural network using your own CSV data. The data is automatically split into training and testing sets.
* **Model Persistence:** Save and load trained models to/from `model.json` files.
* **Prediction:** Use a loaded model to make predictions on new input data.
* **He Initialization:** Weights are initialized using He initialization.
* **Backpropagation:** Implements the backpropagation algorithm for training.

## Getting Started

To run this application, you need to have Go installed on your system.

1.  **Clone the repository:**

    ```bash
    git clone https://github.com/andrewthecodertx/Neural-Network.git
    cd Neural-Network
    ```

2.  **Run the application:**

    The application now features a full-screen terminal user interface (TUI).

    ```bash
    go run .
    ```

## Usage

Upon launching the application, you will be greeted with the main menu. You can navigate through the interface using the arrow keys and press `Enter` to select an option. Press `q` or `Ctrl+C` to quit at any time.

### Train New Model

1.  **Select "Train New Model"** from the main menu.
2.  The application will automatically find any `.csv` files in the root directory.
3.  Fill out the configuration form:
    *   **Select CSV File:** The number corresponding to the dataset you want to use.
    *   **Hidden Layers:** A comma-separated list of neuron counts for each hidden layer (e.g., `20,20`).
    *   **Hidden Activations:** A comma-separated list of activation functions (`relu`, `sigmoid`, `tanh`, `linear`).
    *   **Output Activation:** The activation function for the output layer.
    *   **Epochs:** The number of training iterations.
    *   **Learning Rate:** The step size for gradient descent.
    *   **Error Goal:** The target error at which training will stop.
4.  Navigate to the **"[ Start Training ]"** button and press `Enter`.
5.  A live progress view will show the current epoch and loss.
6.  After training, the model will be evaluated on the test set, and the accuracy will be displayed.
7.  Once training is complete, you will be prompted to enter a name to save the model. The saved model will be placed in the `saved_models/` directory.

### Load Model & Predict

1.  **Select "Load Model & Predict"** from the main menu.
2.  The application will list all models found in the `saved_models/` directory.
3.  Fill out the prediction form:
    *   **Select Model:** The number corresponding to the model you want to use.
    *   **Input Data:** A comma-separated list of numerical values for prediction. The number of values must match the model's expected input size.
4.  Navigate to the **"[ Predict ]"** button and press `Enter`.
5.  The calculated prediction will be displayed on the screen.

## Datasets

The `redwinequality.csv` and `iris.csv` datasets are included as samples. These datasets are from the
UCI Machine Learning Repository.

Citation:
P. Cortez, A. Cerdeira, F. Almeida, T. Matos and J. Reis.
Modeling wine preferences by data mining from physicochemical properties.
In Decision Support Systems, Elsevier, 47(4):547-553, 2009.

## Future Enhancements

* **Command-Line Arguments:** Allow all parameters to be passed via
command-line arguments instead of interactive prompts.
* **Additional Optimizers:** Implement other optimization algorithms like
Adam or RMSprop.
* **Regularization:** Add support for L1/L2 regularization to prevent overfitting.
* **Testing:** Add a comprehensive test suite.

## Contributing

Contributions are welcome! If you'd like to improve this project, feel free to
open an issue or submit a pull request.

## License

This project is licensed under the terms of the [LICENSE](LICENSE) file.

## Docker

This project can also be run inside a Docker container, which handles all the setup for you.

### Prerequisites

*   [Docker](https://www.docker.com/get-started) installed on your system.

### Building the Image

To build the Docker image, run the following command from the project root:

```bash
docker build -t go-neuralnetwork .
```

### Running the Container

To run the application inside a Docker container, use the following command:

```bash
docker run -it --rm go-neuralnetwork
```

The `-it` flags are important for interacting with the TUI.