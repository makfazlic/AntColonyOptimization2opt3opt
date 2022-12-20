import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from tqdm import trange
import sys
np.set_printoptions(threshold=sys.maxsize)
class AntOpt():
    def __init__(self,
                 points,
                 d_matrix = None,
                 seed=0,
                 n_iter=500,    # Number of iterations
                 n_ants=20,     # Number of ants
                 alpha=2,       # pheromone importance
                 beta=3,        # local importance heuristic
                 rho=0.99,      # evaporation factor
                 Q=0.5,         # pheromone amplification factor
                 tau0=1e-4      # initial pheromone level
                ):

        self.n_iter = n_iter
        self.n_ants = n_ants
        self.alpha = alpha
        self.beta = beta
        self.rho = rho
        self.Q = Q
        self.tau0 = tau0

        self.points = points
        self.n_points = len(self.points) # number of nodes/cities
        self.cities = np.arange(self.n_points) # list of nodes/cities


        if d_matrix is None:
            self.d_matrix = self.calc_distance_matrix(self.points)
        else:
            self.d_matrix = d_matrix
        # Check distance matrix is symmetric
        assert (self.d_matrix == self.d_matrix.transpose()).all()

        self.pheremons = self.tau0*np.ones_like(self.d_matrix)
        np.fill_diagonal(self.pheremons, 0)  #  no transition to the same node
        
        # set seed
        np.random.seed(seed)

    @staticmethod
    def euclid_distance(p1, p2):
        rounding = 0
        "Calculate Euclidean distance between two points in 2d"
        assert p1.shape
        return round(np.sqrt((p1[0]-p2[0])**2 + (p1[1]-p2[1])**2), rounding)

    def calc_distance(self, p1, p2):
        """
        Calculate distance between two points
        dist: distance metric [euclid or geo]
        """
        return self.euclid_distance(p1, p2)


    def calc_distance_matrix(self, points: np.array):
        "Calculate distance matrix for array of points"
        n_points = len(points)
        d_matrix = np.zeros((len(points), len(points)), dtype=np.float32)
        for i in range(n_points):
            for j in range(i):
                d_matrix[i,j] = self.calc_distance(points[i,:], points[j, :])
        return d_matrix + d_matrix.transpose()  # symmetric

    def path_length(self, path):
        tot_length = 0
        for i in range(len(path)-1):
            tot_length += self.d_matrix[path[i],path[i+1]]
        return tot_length

    def _make_transition(self, ant_tour):
        "Make single ant transition"
        crnt = ant_tour[-1]
        options = [i for i in self.cities if i not in ant_tour]  # no repetition
        probs = np.array([self.pheremons[crnt, nxt]**self.alpha*(1/self.d_matrix[crnt,nxt])**self.beta for nxt in options])
        probs = probs/sum(probs)  # normalize
        next_city = np.random.choice(options, p=probs)
        ant_tour.append(next_city)


    def run_ants(self):
        "Run ants optimization"

        # Initizlize last improvement iteration
        last_iter = 0

        # Initizlie optimal length
        optimal_length = np.inf

        # Keep track of path length improvement
        best_path_lengths = []

        for it in range(self.n_iter):
            paths = []
            path_lengths = []
            # release ants
            for j in range(self.n_ants):
                # Place ant on random city
                ant_path = [np.random.choice(self.cities)]

                # Make ant choose next node until it covered all nodes
                self._make_transition(ant_path)
                while len(ant_path) < self.n_points:
                    self._make_transition(ant_path)

                # Return to starting node
                ant_path += [ant_path[0]]


                # Calculate path length
                path_length = self.path_length(ant_path)
                
                paths.append(ant_path)
                path_lengths.append(path_length)

                # Check if new optimal
                if path_length < optimal_length:
                    optimal_path = ant_path
                    optimal_length = path_length
                    last_iter = it
                best_path_lengths.append(optimal_length)

            # Break if no improvements for more than 50 iterations
            if (it - last_iter) > 50:
                print(f'breaking at iteration: {it} with best path length: {optimal_length}')
                break

            # Evaporate pheromons
            self.pheremons = self.rho*self.pheremons
            
            # Update pheremons based on path lengths
            for path, length in zip(paths, path_lengths):
                for i in range(self.n_points - 1):
                    self.pheremons[path[i],path[i+1]] += self.Q/length
            
            # Elitist ant
            for k in range(self.n_points - 1):
                self.pheremons[optimal_path[k],optimal_path[k+1]] += self.Q/optimal_length
            print(optimal_length)     

        return optimal_path, optimal_length

    def greedy(self):
        "Generate path by moving to closest node to current node"
        start = np.random.choice(self.cities)
        print(f"start: {start}")
        path = [start]
        while len(path) < len(self.cities):
            options = np.argsort(self.d_matrix[start,:])  # find nearest node
            nxt = [op for op in options if op not in path][0]
            start = nxt
            path.append(nxt)

        # return home
        path += [path[0]]

        return path


    def plot_cities(self):
        "Plot the nodes"
        plt.scatter(self.points[:, 0], self.points[:, 1], s=7, color='k')
        plt.axis('square');

    def plot_path(self, path):
        "Plot a path"
        self.plot_cities()
        plt.plot(self.points[path,0], self.points[path,1], color='k', linewidth=0.6)
        plt.title(f'Path Length: {self.path_length(path):,.1f}')

    def __repr__(self):
        return f"Optimizing with {self.n_points} cities, n_iter={self.n_iter}, n_ants={self.n_ants}, alpha={self.alpha}, beta={self.beta}, rho={self.rho}, Q={self.Q}"

points = [[100, 0], [2, 4.2], [5, 7], [2, 0], [4, 4]]
points = [[334.5909245845, 161.7809319139],
[397.6446634067, 262.8165330708],
[503.8741827107, 172.8741151168],
[444.0479403502, 384.6491809647],
[311.6137146746, 2.0091699828],
[662.8551011379, 549.2301263653],
[40.0979030612, 187.2375430791],
[526.8941409181, 215.7079092185],
[209.1887938487, 691.0262291948],
[683.2674131973, 414.2096286906],
[280.7494438748, 5.9206392047],
[252.7493090080, 535.7430385019],
[698.7850451923, 348.4413729766],
[678.7574678104, 410.7256424438],
[220.0041131179, 409.1225812873],
[355.1528556851, 76.3912076444],
[296.9724227786, 313.1312792361],
[504.5154071733, 240.8866564499],
[224.1079496785, 358.4872228907],
[470.6801296968, 309.6259188406],
[554.2530513223, 279.4242466521],
[567.6332684419, 352.7162027273],
[599.0532671093, 361.0948690386],
[240.5232959211, 430.6036007844],
[32.0825972787, 345.8551009775],
[91.0538736891, 148.7213270256],
[248.2179894723, 343.9528017384],
[488.8909044347, 3.6122311393],
[206.0467939820, 437.7639406167],
[575.8409415632, 141.9670960195],
[282.6089948164, 329.4183805862],
[27.6581484868, 424.7684581747],
[568.5737309870, 287.0975660546],
[269.4638933331, 295.9464636385],
[417.8004856811, 341.2596589955],
[32.1680938737, 448.8998721172],
[561.4775136009, 357.3543930067],
[342.9482167470, 492.3321423839],
[399.6752075383, 156.8435035519],
[571.7371050025, 375.7575350833],
[370.7559842751, 151.9060751898],
[509.7093253204, 435.7975189314],
[177.0206999750, 295.6044772584],
[526.1674198605, 409.4859418161],
[316.5725171854, 65.6400108214],
[469.2908100279, 281.9891445025],
[572.7630641427, 373.3208821255],
[29.5176994283, 330.0382309000],
[454.0082936692, 537.2178547659],
[416.1546762271, 227.6133100741],
[535.2514330806, 471.0648643744],
[265.4455533675, 684.9987192464],
[478.0542110167, 509.6452028741],
[370.4781203413, 332.5390063041],
[598.3479202004, 446.8693279856],
[201.1521139175, 649.0260268945],
[193.6925360026, 680.2322840744],
[448.5792598859, 532.7934059740],
[603.2853485624, 134.4006473609],
[543.0102490781, 481.5168231148],
[214.5750793346, 43.6460117543],
[426.3501451825, 61.7285415996],
[89.0447037063, 277.1158385868],
[84.4920100219, 31.8474816424],
[220.0468614154, 623.0778103080],
[688.4613313444, 0.4702312726],
[687.2857531630, 373.5346236130],
[75.4934933967, 312.9175377486],
[63.4170993511, 23.7039309674],
[97.9363495877, 211.0910930878],
[399.5255884970, 170.8221968365],
[456.3167017346, 597.1937161677],
[319.8855102422, 626.8396604886],
[295.9250894897, 664.6291554845],
[288.4868857235, 667.7284070537],
[268.3951858954, 52.9010181645],
[140.4709056068, 513.5566720960],
[689.8079027159, 167.5947003748],
[280.5784506848, 458.7533546925],
[453.3884433554, 282.9082328989],
[213.5704943432, 525.8681817779],
[133.6953004520, 677.1757808026],
[521.1658690522, 132.8617086506],
[30.2657946347, 450.0754502986],
[657.0199585283, 39.7772908299],
[6.9252241961, 23.8749241575],
[252.4286967767, 535.1659364856],
[42.8551682504, 63.8232081774],
[145.8999393902, 399.5255884970],
[638.4885715591, 62.6262558472],
[489.2756391122, 665.3131282446],
[361.2231139311, 564.2347787901],
[519.9475425732, 347.9711417040],
[129.3349741063, 435.6692740389],
[259.7172815016, 454.6495181318],
[676.3421890013, 371.0979706551],
[84.5133841706, 183.3260738572],
[77.7164048671, 354.3833863300],
[335.9802442534, 660.6321896676],
[264.3554717810, 377.5743377274],
[51.6826916855, 676.0429509187],
[692.1376849300, 543.8010925819],
[169.2191356800, 547.8194325476],
[194.0131482339, 263.4791316822],
[415.1928395332, 78.9133571973],
[415.0432204919, 479.0801701569],
[169.8389859939, 245.6103433244],
[525.0987124228, 213.5063718969],
[238.6851191283, 33.4932910965],
[116.2112467718, 363.5742702940],
[16.9283258126, 656.5711014044],
[434.3440768162, 92.6996831431],
[40.5253860363, 424.6829615797],
[530.4849979086, 183.8390534273],
[484.3595848990, 49.2460387276],
[263.6501248722, 426.5852608187],
[450.2891917862, 126.3853415784],
[441.7822805823, 299.7724362653],
[24.2169105375, 500.3474481664],
[503.7886861157, 514.6895019799],
[635.5389390312, 200.9811207275],
[614.5922732529, 418.8691931188],
[21.7161351334, 660.9741760476],
[143.8266469611, 92.6996831431],
[637.7191022040, 54.2048412384],
[566.5645610042, 199.9551615873],
[196.6849168280, 221.8209157619],
[384.9270448985, 87.4630166986],
[178.1107815614, 104.6905805938],
[403.2874386776, 205.8971749407]]
points = np.array(points, dtype=np.float128)
aco = AntOpt(points, None, 5, 500, 30, 2, 3, 0.99, 0.2, 1e-4)
print(aco.run_ants())