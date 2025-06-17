#define MATPLOT_HEADER_ONLY
#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <string>
#include <Eigen/Dense>
#include <matplot/matplot.h>

using namespace std;
using namespace Eigen;
using namespace matplot;

struct InflationData {
    int month_id;
    double inflation;
    double low_target;
    double up_target;
};

// ---------- 函数定义 ----------

// 读取 CSV 数据
vector<InflationData> read_csv(const string &filename) {
    vector<InflationData> data;
    ifstream file(filename);
    string line;
    getline(file, line); // skip header
    int month_id = 0;
    while (getline(file, line)) {
        stringstream ss(line);
        string token;
        InflationData row;
        row.month_id = month_id++;

        getline(ss, token, ','); // 日期，跳过
        getline(ss, token, ','); row.inflation = stod(token);
        getline(ss, token, ','); row.low_target = stod(token);
        getline(ss, token, ','); row.up_target = stod(token);

        data.push_back(row);
    }
    return data;
}

// 构造特征矩阵和目标变量
pair<MatrixXd, VectorXd> build_features(const vector<InflationData> &data) {
    int n = data.size();
    MatrixXd X(n, 4);
    VectorXd y(n);
    for (int i = 0; i < n; ++i) {
        double lag = (i == 0) ? data[i].inflation : data[i - 1].inflation;
        X(i, 0) = lag;
        X(i, 1) = data[i].month_id;
        X(i, 2) = data[i].low_target;
        X(i, 3) = data[i].up_target;
        y(i) = data[i].inflation;
    }
    return {X, y};
}

// Ridge 回归封装
VectorXd ridge_regression(const MatrixXd &X, const VectorXd &y, double lambda) {
    int d = X.cols();
    MatrixXd I = MatrixXd::Identity(d, d);
    return (X.transpose() * X + lambda * I).ldlt().solve(X.transpose() * y);
}

// ---------- 主函数 ----------

int main() {
    string filename = "/Users/macbookpro/Desktop/CPI-INFLATION-sd-1993-01-01-ed-2022-01-01.csv";
    auto data = read_csv(filename);

    auto [X, y] = build_features(data);

    // Train-test split
    int n_train = static_cast<int>(0.8 * X.rows());
    MatrixXd X_train = X.topRows(n_train);
    VectorXd y_train = y.head(n_train);
    MatrixXd X_test = X.bottomRows(X.rows() - n_train);
    VectorXd y_test = y.tail(y.size() - n_train);

    double lambda = 10.0;
    VectorXd beta = ridge_regression(X_train, y_train, lambda);

    VectorXd y_train_pred = X_train * beta;
    VectorXd y_test_pred = X_test * beta;

    double train_mse = (y_train - y_train_pred).squaredNorm() / y_train.size();
    double test_mse = (y_test - y_test_pred).squaredNorm() / y_test.size();

    cout << "==== Ridge Regression Forecast ====" << endl;
    cout << "Train MSE = " << train_mse << endl;
    cout << "Test  MSE = " << test_mse << endl;

    // 构造历史数据用于绘图
    vector<double> past_x, past_y;
    for (const auto &d : data) {
        past_x.push_back(d.month_id);
        past_y.push_back(d.inflation);
    }

    // 预测未来 120 个月
    vector<double> future_x, future_y;
    InflationData last = data.back();
    double last_infl = last.inflation;
    for (int i = 1; i <= 120; ++i) {
        InflationData f;
        f.month_id = last.month_id + i;
        f.low_target = 1.0;
        f.up_target = 3.0;

        RowVectorXd fx(4);
        fx << last_infl, f.month_id, f.low_target, f.up_target;
        double pred = fx * beta;

        future_x.push_back(f.month_id);
        future_y.push_back(pred);

        last_infl = pred;
    }

    cout << "\n==== Forecasted Inflation for Next 10 Years ====\n";
    for (size_t i = 0; i < future_x.size(); ++i) {
        int month_id = static_cast<int>(future_x[i]);
        int year = 1993 + month_id / 12;
        int month = month_id % 12 + 1;
        printf("Month %4d-%02d: %.2f%%\n", year, month, future_y[i]);
    }


    // 设置年份刻度标签
    vector<double> tick_positions;
    vector<string> tick_labels;
    int start_year = 1993;
    for (int m = 0; m <= future_x.back(); m += 12) {
        tick_positions.push_back(m);
        tick_labels.push_back(to_string(start_year + m / 12));
    }

    // ---------- 绘图 ----------
    figure();
    hold(on);

    auto h1 = plot(past_x, past_y, "b");
    h1->line_width(2);

    auto h2 = plot(future_x, future_y, "r--");
    h2->line_width(2);

    xlabel("Year");
    ylabel("Inflation Rate (%)");
    title("Canadian Inflation Forecast (Ridge Regression)");
    legend({"Historical", "Forecast (Next 10 years)"});

    xticks(tick_positions);
    xticklabels(tick_labels);

    hold(off);
    show();

    return 0;
}
