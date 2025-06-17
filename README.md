# Canadian Inflation Forecast Using Ridge Regression

## Overview
This C++ program forecasts the future inflation rate of Canada for the next 10 years (120 months) using historical data and Ridge Regression.

## Features
- Reads historical monthly inflation data from CSV.
- Builds a feature matrix including lagged inflation and inflation target bounds.
- Trains a Ridge Regression model using the Eigen library.
- Forecasts future inflation using the trained model.
- Visualizes historical and predicted inflation using Matplot++.

## Libraries Used
- ✅ **Matplot++** — for data visualization
- ✅ **Eigen** — for matrix operations and Ridge regression

## How to Build
This project uses **CMake**.

### Requirements
- CMake 3.15+
- C++17 compiler (e.g., Clang or GCC)
- Eigen installed via Homebrew:  
  ```bash
  brew install eigen
  ```

### Build Steps
```bash
mkdir build
cd build
cmake ..
make
```

This will generate the executable named `untitled3`.

## How to Run
```bash
./untitled3
```

## Files Included
- `main.cpp` - Source code
- `CMakeLists.txt` - Build instructions
- `libs/` - Contains Matplot++ library
- `untitled3` - Precompiled binary 
- `README.md` - This file

## Author
Liyuan Cao  140029  
lc140029@student.sgh.waw.pl  
Date: June 2025
