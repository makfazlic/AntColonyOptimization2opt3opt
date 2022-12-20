// COMPILE AND RUN WITH -O3 OR -Ofast DEEPEST OPTIMIZATION
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
        for (int i = 0; i < list.size(); i++)
        {
            for (int j = i + 2; j < list.size(); j++)
            {
                for (int k = j + 2; k < list.size() + (i > 0); k++)
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

    int swap(vector<int> &list, int i, int j, int k)
    {
        int A = list[i - 1];
        int B = list[i];
        int C = list[j - 1];
        int D = list[j];
        int E = list[k - 1];
        int F = list[k % list.size()];
        int d0 = d_matrix(A, B) + d_matrix(C, D) + d_matrix(E, F);
        int d1 = d_matrix(A, C) + d_matrix(B, D) + d_matrix(E, F);
        int d2 = d_matrix(A, B) + d_matrix(C, E) + d_matrix(D, F);
        int d3 = d_matrix(A, D) + d_matrix(E, B) + d_matrix(C, F);
        int d4 = d_matrix(F, B) + d_matrix(C, D) + d_matrix(E, A);


        if (d0 > d1)
        {
            reverse(list.begin() + i, list.begin() + j);
            return -d0 + d1;
        }
        else if (d0 > d2)
        {
            reverse(list.begin() + j, list.begin() + k);
            return -d0 + d2;
        }
        else if (d0 > d4)
        {
            reverse(list.begin() + i, list.begin() + k);
            return -d0 + d4;
        }
        else if (d0 > d3)
        {
            vector<int> temp;
            temp.insert(temp.end(), list.begin() + j, list.begin() + k);
            temp.insert(temp.end(), list.begin() + i, list.begin() + j);
            copy(temp.begin(), temp.end(), list.begin() + i);
            return -d0 + d3;
        }
        return 0;
    }

    double run()
    {
        // Start timer
        auto start = chrono::high_resolution_clock::now();
        vector<vector<int>> segments = generateSegments(cities);
        while (true)
        {
            int delta = 0;
            for (int i = 0; i < segments.size(); i++)
            {
                delta += swap(cities, segments[i][0], segments[i][1], segments[i][2]);
            }       
            //cout << "Best delta: " << best_delta << endl;
            if (delta >= 0)
            {
                break;
            }

        }
        cout << "best path length: " << pathLength(cities) << endl;
                // Stop timer
        auto stop = chrono::high_resolution_clock::now();
        auto duration = chrono::duration_cast<chrono::microseconds>(stop - start);
        int best_length = pathLength(cities);
        cout << "Best Length: " << best_length  << " in " << duration.count() << " microseconds" << endl;
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
