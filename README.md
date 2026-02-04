# Laptop Price Estimator

## Overview

The Laptop Price Estimator is a web application built using Streamlit that predicts the price of a laptop based on various features such as brand, type, processor, RAM, GPU, operating system, weight, touchscreen availability, IPS display, screen size, and screen resolution. The application uses a pre-trained machine learning model to provide accurate price predictions.

## Features

- **User-Friendly Interface**: Simple and intuitive interface for users to input laptop specifications.
- **Multiple Input Options**: Users can select from various options for each feature, including brand, processor type, RAM, GPU, and more.
- **Price Prediction**: The application predicts the price of the laptop based on the provided specifications.
- **Responsive Design**: The application is designed to be responsive and works well on different screen sizes.

## Installation

To run the Laptop Price Estimator locally, follow these steps:

1. **Clone the Repository**:
   ```bash
   git clone https://github.com/RevanthGangapur/Laptop_Price_Estimator.git
   cd Laptop_Price_Estimator
   ```

2. **Install Dependencies**:
   Ensure you have Python installed. Then, install the required packages using pip:
   ```bash
   pip install -r requirements.txt
   ```

3. **Run the Application**:
   Start the Streamlit application by running:
   ```bash
   streamlit run app.py
   ```

4. **Access the Application**:
   Open your web browser and navigate to `http://localhost:8501` to access the application.

## Usage

1. **Input Laptop Specifications**:
   - Select the brand, type, processor, RAM, GPU, and operating system from the dropdown menus.
   - Enter the weight of the laptop.
   - Choose whether the laptop has a touchscreen and IPS display.
   - Enter the screen size and select the screen resolution.

2. **Predict Price**:
   - Click the "Predict Price" button to get the estimated price of the laptop based on the provided specifications.

## Technologies Used

- **Streamlit**: For building the web application interface.
- **Python**: For backend logic and data processing.
- **XGBoost**: For the machine learning model used in price prediction.
- **Pandas**: For data manipulation and handling.
- **NumPy**: For numerical computations.

## Contributing

Contributions are welcome! If you have any suggestions, bug reports, or feature requests, please open an issue or submit a pull request.

## Acknowledgments

- Thanks to the Streamlit team for providing an excellent framework for building data applications.
- Special thanks to the open-source community for providing valuable resources and libraries.

---

Enjoy using the Laptop Price Estimator!
