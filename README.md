# Canadian Inflation Forecast Using Ridge Regression

## Overview
This C++ project forecasts Canada's future inflation rate for the next 10 years (120 months) by leveraging historical monthly inflation data and applying Ridge Regression.

The model builds a feature matrix using lagged inflation values and inflation target bounds, then trains a Ridge Regression model with the Eigen library. Results are visualized with Matplot++.

---

## Features
- Load historical monthly inflation data from CSV.
- Construct feature matrix with lagged inflation and inflation target bounds.
- Train Ridge Regression model using Eigen for linear algebra operations.
- Predict future inflation over 120 months.
- Visualize both historical and forecasted inflation using Matplot++.

---

## Libraries Used
- **Matplot++** — for visualization (introduced in Lecture 12)
- **Eigen** — for matrix operations and Ridge regression

---

## Requirements
- C++17 compatible compiler (e.g., Clang, GCC)
- CMake 3.15 or higher
- Eigen (installed via Homebrew)

### Installing Eigen on macOS
```bash
brew install eigen
