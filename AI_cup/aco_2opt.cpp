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

class AntOpt
{
private:
    bool DEBUG = false;
    int seconds_to_run;
    string path;
    Matrix<double, Dynamic, 2> points;
    int n_points;
    vector<int> cities;
    MatrixXd d_matrix;
    int seed;
    int n_itter;
    int n_ants;
    double alpha;
    double beta;
    double rho;
    double Q;
    double tau0;
    MatrixXd pheromones;

public:
    AntOpt(int new_seconds_to_run, string new_path, int new_seed, int new_n_itter,
           int new_n_ants, double new_alpha, double new_beta, double new_rho,
           double new_Q, double new_tau0)
    {
        path = new_path;
        seconds_to_run = new_seconds_to_run;
        points = initPoints(path);
        n_points = points.rows();
        cities = initCities(n_points);
        d_matrix = initDistanceMatrix(points, n_points);
        seed = new_seed;
        n_itter = new_n_itter;
        n_ants = new_n_ants;
        alpha = new_alpha;
        beta = new_beta;
        rho = new_rho;
        Q = new_Q;
        tau0 = new_tau0;
        pheromones = initPheromones(n_points);
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
        cout << "Runtime (s) ------ " << to_string(seconds_to_run) << endl;
        cout << "Points head ------ v" << endl
             << points.topRows(5) << endl;
        cout << "Point count ------ " << to_string(n_points) << endl;
        cout << "Seed ------------- " << to_string(seed) << endl;
        cout << "Itterations ------ " << to_string(n_itter) << endl;
        cout << "Ants ------------- " << to_string(n_ants) << endl;
        cout << "Alpha ------------ " << to_string(alpha) << endl;
        cout << "Beta ------------- " << to_string(beta) << endl;
        cout << "Rho -------------- " << to_string(rho) << endl;
        cout << "Q ---------------- " << to_string(Q) << endl;
        cout << "Tau0 ------------- " << to_string(tau0) << endl;

        if (DEBUG)
        {
            cout << endl
                 << "[DEBUG MODE] Colony derived fields" << endl;
            cout << "Cities ----------- v" << endl
                 << stringifyVectorInt(cities, 2) << endl;
            cout << "Distance matrix -- v" << endl
                 << d_matrix << endl;
            cout << "Pheromones ------- v" << endl
                 << pheromones << endl;
        }
    }

    void showColony()
    {
        cout
            << "[Init]"
            << " Problem -> " << path
            << ", Runtime -> " << to_string(seconds_to_run)
            << ", n_points -> " << to_string(n_points)
            << ", Itterations -> " << to_string(n_itter)
            << ", Seed -> " << to_string(seed)
            << ", Ants -> " << to_string(n_ants)
            << ", Alpha -> " << to_string(alpha)
            << ", Beta -> " << to_string(beta)
            << ", Rho -> " << to_string(rho)
            << ", Q -> " << to_string(Q)
            << ", Tau0 -> " << to_string(tau0)
            << endl;

        if (DEBUG)
        {
            cout << endl
                 << "[DEBUG MODE] Colony derived fields" << endl;
            cout << "Cities ----------- v" << endl
                 << stringifyVectorInt(cities, 2) << endl;
            cout << "Distance matrix -- v" << endl
                 << d_matrix << endl;
            cout << "Pheromones ------- v" << endl
                 << pheromones << endl;
        }
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

    vector<int> makeTransition(vector<int> list)
    {
        // https://www.youtube.com/watch?v=783ZtAF4j5g
        int current_city = list.back();

        vector<int> options;
        for (int city : cities)
        {
            bool in = false;
            for (int list_element : list)
            {
                if (city == list_element)
                {
                    in = true;
                    break;
                }
            }
            if (!in)
            {
                options.push_back(city);
            }
        }

        vector<double> probs;
        for (int next_city : options)
        {
            double pheromone_component =
                pow(pheromones(current_city, next_city), alpha);
            double distance_component =
                pow((1 / d_matrix(current_city, next_city)), beta);
            probs.push_back(pheromone_component * distance_component);
        }

        double sum_probs = 0;
        for (double prob : probs)
            sum_probs += prob;

        for (int i = 0; i < probs.size(); i++)
        {
            probs[i] = probs[i] / sum_probs;
        }

        double r = ((double)rand() / (double)RAND_MAX);
        double cummulative_probs_current = 0;
        int selected_option = 0;
        for (int i = 0; i < options.size(); i++)
        {
            cummulative_probs_current += probs[i];
            if (r <= cummulative_probs_current)
            {
                selected_option = i;
                break;
            }
        }

        // Can optimize more to return just std::vector and work with those
        // That would involve making cities into a std::vector and printing with stringify
        // Because push_back() is O(1) and this shit bellow is O(n)
        list.push_back(options[selected_option]);
        return list;
    }

    MatrixXd initPheromones(int n_points)
    {
        MatrixXd result = (MatrixXd::Ones(n_points, n_points) -
                           MatrixXd::Identity(n_points, n_points)) *
                          tau0;
        return result;
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

    vector<int> swap(vector<int> list, int i, int j)
    {
        vector<int> new_list = list;
        reverse(new_list.begin() + i + 1, new_list.begin() + j + 1);
        return new_list;
    }

    double run()
    {
        // Last improvement
        int last_itter = 0;

        // To return final
        double optimal_length = INFINITY;

        // The path of the optimal length
        vector<int> optimal_path;

        // Keeping track of all good lengths
        vector<double> best_path_lengths;
        auto start = high_resolution_clock::now();
        for (int i = 0; i < n_itter; i++)
        {
            // Paths of this itteration reset
            vector<vector<int>> paths;
            // Path lengths of this itteration reset
            vector<double> path_lengths;

            // Let ants run
            for (int a = 0; a < n_ants; a++)
            {
                // Generate random number to take the first city
                int r = rand() % cities.size();

                // Initialize the path
                vector<int> ant_path;
                ant_path.push_back(cities[r]);
                ant_path = makeTransition(ant_path);

                // Make ant choose next node until it covered all nodes
                while (ant_path.size() < n_points)
                {
                    ant_path = makeTransition(ant_path);
                }

                // Return home
                ant_path.push_back(ant_path[0]);
                // Calculate path length
                int path_length = pathLength(ant_path);

                vector<int> new_cities = ant_path;
                int new_length;
                int swaps = 1;
                while (swaps != 0)
                { // loop until no improvements are made.
                    swaps = 0;
                    for (int i = 1; i < ant_path.size() - 2; i++)
                    {
                        for (int j = i + 1; j < ant_path.size() - 1; j++)
                        {
                            // difference in length if edge (i, i+1) and (j, j+1) were swapped
                            int diff = d_matrix(ant_path[i], ant_path[j]) + d_matrix(ant_path[i + 1], ant_path[j + 1]) - d_matrix(ant_path[i], ant_path[i + 1]) - d_matrix(ant_path[j], ant_path[j + 1]);
                            if (diff < 0)
                            {
                                // swap edges (i, i+1) and (j, j+1)
                                new_cities = swap(ant_path, i, j);
                                new_length = pathLength(new_cities);
                                if (new_length < path_length)
                                {
                                    //cout << "Improvement: " << path_length << " -> " << new_length << endl;
                                    ant_path = new_cities;
                                    path_length = new_length;
                                    swaps++;
                                }
                            }
                        }
                    }
                }



                paths.push_back(ant_path);
                path_lengths.push_back(path_length);

                if (path_length < optimal_length)
                {
                    optimal_path = ant_path;
                    optimal_length = path_length;
                    last_itter = i;
                }

                best_path_lengths.push_back(optimal_length);
            } // Ants stopped

            auto now = high_resolution_clock::now();
            auto runtime = (duration_cast<microseconds>(now - start) / 1e+6).count();

            if (runtime >= seconds_to_run)
            {
                cout << optimal_length << " - "
                     << "Time limit" << endl;
                // cout << "[Time limit] Breaking at iteration: " << i << " with best path length: " << optimal_length << endl;
                return optimal_length;
            }

            if ((i - last_itter) > 50)
            {
                cout << optimal_length << " - "
                     << "Improvement limit" << endl;
                // cout << "[Improvement limit] Breaking at iteration: " << i << " with best path length: " << optimal_length << endl;
                return optimal_length;
            }

            // Evaporate
            pheromones = rho * pheromones;

            // Update pheromones based on length
            int zip_length = min(paths.size(), path_lengths.size());
            for (int z = 0; z < zip_length; z++)
            {
                vector<int> path = paths[z];
                double path_length = path_lengths[z];
                for (int n = 0; n < n_points - 1; n++)
                {
                    pheromones(path[n], path[n + 1]) += Q / path_length;
                }
            }

            // Elite ants
            for (int k = 0; k < n_points - 1; k++)
            {
                pheromones(optimal_path[k], optimal_path[k + 1]) += Q / optimal_length;
            }

            // for (auto city : optimal_path)
            //   {
            //     cout << city << " ";
            //   }
        }
        cout << optimal_length << " - "
             << "Natural stop" << endl;
        return optimal_length;
    }
};

int main(int argc, char **argv)
{

    if (argc != 11)
    {
        cout << "[Error] Not able to initialize colony, please supply correct arguments" << endl;
        cout << "Expected: " << argv[0] << " run_seconds file_name seed itterations ants alpha beta evaporation Q init_pheromone_value" << endl;
        cout << "Example: " << argv[0] << " 120 problems/ch130.tsp 69 500 30 2 3 0.99 0.2 1e-4" << endl;
        return 1;
    }

    int seconds_to_run = stoi(argv[1]);
    string filename = argv[2];
    int seed = stoi(argv[3]);
    int itterations = stoi(argv[4]);
    int ants = stoi(argv[5]);
    double alpha = stod(argv[6]);
    double beta = stod(argv[7]);
    double evaporation = stod(argv[8]);
    double Q = stod(argv[9]);
    double init_pheromone_value = stod(argv[10]);

    // Set global variables
    srand(seed);
    cout.precision(10);

    // Solver
    AntOpt colony(seconds_to_run, filename, seed, itterations, ants, alpha, beta, evaporation, Q, init_pheromone_value);
    // colony.showColonyVerbose();
    // colony.showColony();
    colony.run();

    return 0;
}