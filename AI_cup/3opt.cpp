// COMPILE AND RUN WITH -O3 OR -Ofast DEEPEST OPTIMIZATION
// g++ -I ./eigen -Ofast 3opt.cpp -o 3opt && ./3opt problems/fl1577.tsp
#include <fstream>
#include <iostream>
#include <numeric>
#include <random>
#include <sstream>
#include <string>
#include <vector>
#include <chrono>

// Using EIGEN for Linear Algebra
#include <Eigen/Dense>

using namespace std;
using namespace Eigen;
using namespace std::chrono;
using Eigen::MatrixXd;

class III_opt
{
private:
    string path;
    Matrix<double, Dynamic, 2> points;
    int n_points;
    vector<int> cities;
    MatrixXd d_matrix;

public:
    III_opt(string new_path)
    {
        path = new_path;
        points = initPoints(path);
        n_points = points.rows();
        cities = initCities(n_points);
        d_matrix = initDistanceMatrix(points, n_points);
    }

    vector<int> initCities(int n_points)
    {
        vector<int> result;
        for (int i = 0; i < n_points; i++)
        {
            result.push_back(i);
        }
        return result;
    }

    // Shower methods
    string stringifyVectorInt(vector<int> vector, int n)
    {
        string ss;
        for (int i = 0; i < n; i++)
        {
            ss += to_string(vector[i]);
            if (i != n - 1)
            {
                ss += " ";
            }
        }
        return ss;
    }

    // string stringify2DVector(vector<vector<double>> matrix, int m, int n) {
    //   string ss;
    //   for (int i = 0; i < m; i++) {
    //     for (int j = 0; j < n; j++) {
    //       ss += to_string(matrix[i][j]) + " ";
    //       if (j == n - 1 && i != m - 1) {
    //         ss += " \n";
    //       }
    //     }
    //   }
    //   return ss;
    // }

    // string stringifySquareMatrix(vector<vector<double>> matrix, int n) {
    //   string ss;
    //   for (int i = 0; i < n; i++) {
    //     for (int j = 0; j < n; j++) {
    //       ss += to_string(matrix[i][j]) + " ";
    //       if (j == n - 1 && i != n - 1) {
    //         ss += " \n";
    //       }
    //     }
    //   }
    //   return ss;
    // }

    void showColonyVerbose()
    {
        cout << "[Init] Colony hyperparameters" << endl;
        cout << "Points head ------ v" << endl
             << points.topRows(5) << endl;
        cout << "Point count ------ " << to_string(n_points) << endl;
    }

    void showColony()
    {
        cout
            << "[Init]"
            << " Problem -> " << path
            << ", n_points -> " << to_string(n_points)
            << endl;
    }

    // Calculation methods
    int euclidianDistance(Vector2d point_i, Vector2d point_j)
    {
        double x_i = point_i(0);
        double y_i = point_i(1);
        double x_j = point_j(0);
        double y_j = point_j(1);
        double distance =
            sqrt(((x_i - x_j) * (x_i - x_j)) + ((y_i - y_j) * (y_i - y_j)));
        int rounded = round(distance);
        return rounded;
    }

    MatrixXd initDistanceMatrix(Matrix<double, Dynamic, 2> points, int n_points)
    {
        MatrixXd result = MatrixXd::Zero(n_points, n_points);
        for (int i = 0; i < n_points; i++)
        {
            for (int j = i; j < n_points; j++)
            {
                if (i == j)
                {
                    result(i, j) = INFINITY;
                }
                else
                {
                    result(i, j) = euclidianDistance(points.row(i), points.row(j));
                }
            }
        }
        MatrixXd transposed = result.transpose();
        result += transposed;
        return result;
    }

    int pathLength(vector<int> list)
    {
        int tot_length = 0;

        for (int i = 0; i < list.size() - 1; i++)
        {
            tot_length += d_matrix(list[i], list[i + 1]);
        }
        return tot_length;
    }

    Matrix<double, Dynamic, 2> initPoints(string path)
    {
        vector<Vector2d> temp_matrix;
        int pass_7 = -7;
        int count = 0;
        ifstream file(path);
        string line;
        while (getline(file, line))
        {
            if (pass_7 >= 0 && line != "EOF")
            {
                string arr[3];
                int i = 0;
                stringstream ssin(line);
                while (ssin.good() && i < 3)
                {
                    ssin >> arr[i];
                    ++i;
                }
                Vector2d line_vector;
                line_vector << stod(arr[1]), stod(arr[2]);
                temp_matrix.push_back(line_vector);
                count++;
            }
            pass_7++;
        }
        file.close();
        Matrix<double, Dynamic, 2> file_points(count, 2);
        for (int i = 0; i < count; i++)
        {
            file_points.row(i) = temp_matrix[i];
        }
        return file_points;
    }

    vector<vector<int>> generateSegments(vector<int> list)
    {
        vector<vector<int>> segments;
        for (int i = 1; i < list.size(); i++)
        {
            for (int j = i + 2; j < list.size() - 1; j++)
            {
                for (int k = j + 2; k < list.size() - 2 + (i > 0); k++)
                {
                    vector<int> segment;
                    segment.push_back(i);
                    segment.push_back(j);
                    segment.push_back(k);
                    segments.push_back(segment);
                }
            }
        }
        return segments;
    }

    void swap(vector<int> *list, int i, int j, int k, int best_case)
    {

        if (best_case == 2)
        {
            reverse(list->begin() + j + 1, list->begin() + k + 1);
        }
        if (best_case == 3)
        {
            reverse(list->begin() + i + 1, list->begin() + j + 1);
        }
        if (best_case == 4)
        {
            reverse(list->begin() + i + 1, list->begin() + k + 1);
        }
        if (best_case == 5)
        {
            reverse(list->begin() + i + 1, list->begin() + j + 1);
            reverse(list->begin() + i + 1, list->begin() + k + 1);
        }
        if (best_case == 6)
        {
            reverse(list->begin() + j + 1, list->begin() + k + 1);
            reverse(list->begin() + i + 1, list->begin() + k + 1);
        }
        if (best_case == 7)
        {
            reverse(list->begin() + i + 1, list->begin() + j + 1);
            reverse(list->begin() + j + 1, list->begin() + k + 1);
        }
        if (best_case == 8)
        {
            vector<int> tempTour = vector<int>{};
            tempTour.insert(tempTour.end(), list->begin() + j + 1, list->begin() + k + 1);
            tempTour.insert(tempTour.end(), list->begin() + i + 1, list->begin() + j + 1);
            copy_n(tempTour.begin(), tempTour.size(), &(*list)[i + 1]);
        }
    }

    double run()
    {
        // Start timer
        auto start = chrono::high_resolution_clock::now();
        cout << "started timer" << endl;
        //vector<vector<int>> segments = generateSegments(cities);
        //cout << "segments: " << segments.size() << endl;

        vector<int> my_cities = cities;

        for (int i = 1; i < cities.size(); i++)
        {
            for (int j = i + 2; j < cities.size() - 1; j++)
            {
                for (int k = j + 2; k < cities.size() - 2 + (i > 0); k++)
                {

                    int A = my_cities[i];
                    int B = my_cities[i + 1];
                    int C = my_cities[j];
                    int D = my_cities[j + 1];
                    int E = my_cities[k];
                    int F = my_cities[k + 1];

                    // Base cycle
                    int cb = d_matrix(A, B) + d_matrix(C, D) + d_matrix(E, F);
                    int c2 = d_matrix(A, B) + d_matrix(C, E) + d_matrix(D, F);
                    int c3 = d_matrix(A, C) + d_matrix(B, D) + d_matrix(E, F);
                    int c4 = d_matrix(A, E) + d_matrix(B, F) + d_matrix(C, D);
                    int c5 = d_matrix(A, E) + d_matrix(B, D) + d_matrix(C, F);
                    int c6 = d_matrix(A, D) + d_matrix(B, F) + d_matrix(C, E);
                    int c7 = d_matrix(A, C) + d_matrix(B, E) + d_matrix(D, F);
                    int c8 = d_matrix(A, D) + d_matrix(B, E) + d_matrix(C, F);

                    int db = 0;
                    int d2 = c2 - cb;
                    int d3 = c3 - cb;
                    int d4 = c4 - cb;
                    int d5 = c5 - cb;
                    int d6 = c6 - cb;
                    int d7 = c7 - cb;
                    int d8 = c8 - cb;

                    vector<vector<int>> cases;
                    cases.push_back({1, db});
                    cases.push_back({2, d2});
                    cases.push_back({3, d3});
                    cases.push_back({4, d4});
                    cases.push_back({5, d5});
                    cases.push_back({6, d6});
                    cases.push_back({7, d7});
                    cases.push_back({8, d8});

                    int best_case = 1;

                    for (vector<int> c : cases)
                        if (c[1] < cases[best_case - 1][1])
                            best_case = c[0];

                    // cout << "Best case: " << cases[best_case -1][1] << " " << best_case << endl;
                    if (cases[best_case - 1][1] < 0)
                    {
                        swap(&my_cities, i, j, k, best_case);
                    }
                }
            }
        }

        int path_length = pathLength(cities);

        cout << "best path length: " << pathLength(my_cities) << endl;
        cout << "original path length: " << path_length << endl;
        // Stop timer
        auto stop = chrono::high_resolution_clock::now();
        auto duration = chrono::duration_cast<chrono::microseconds>(stop - start);
        int best_length = pathLength(my_cities);
        cout << "Best Length: " << best_length << " in " << duration.count() << " microseconds" << endl;
        return best_length;
    }
};

int main(int argc, char *argv[])
{
    string path = argv[1];
    III_opt colony(path);
    colony.run();
    return 0;
}
