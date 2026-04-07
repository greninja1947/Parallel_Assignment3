#include <bits/stdc++.h>
#include <mpi.h>
using namespace std;
struct Graph{
    int N,E,B;
    vector<int>profit;
    vector<int>cost;
    vector<vector<int>>adjacency_matrix;
    vector<vector<int>>adjacency_list;
};

vector<int>bestClique;
vector<int>currentClique;
int maxProfit;
Graph graph;
bool compDecreasingProfit(int a, int b){
    return graph.profit[a] > graph.profit[b];
}

bool compDecreasingRatio(int a, int b){
    double r1 = (graph.profit[a] * 1.0) / graph.cost[a];
    double r2 = (graph.profit[b] * 1.0) / graph.cost[b];
    return r1 > r2 ;
}
void readInput(string infile, Graph& graph){
    ifstream input(infile);
    if(!input){
        cout<<"could not read file\n";return;
    }
    int u,v;
    int N,E,B;
    input>>N>>E>>B;
    graph.N = N;
    graph.E = E;
    graph.B = B;
    graph.profit.resize(N);
    graph.cost.resize(N);
    graph.adjacency_matrix.resize(N,vector<int>(N,0));
    graph.adjacency_list.resize(N);
    for(int i = 0;i<N;i++){
        input >> graph.profit[i] >> graph.cost[i];
    }
    for(int i = 0;i<E;i++){
        input >>u >> v;
        graph.adjacency_matrix[u][v] = 1;
        graph.adjacency_matrix[v][u] = 1;

        graph.adjacency_list[u].push_back(v);
        graph.adjacency_list[v].push_back(u);
    }
}

void writeOutput(string outfile){
    ofstream output(outfile);
    output << maxProfit <<"\n";
    sort(bestClique.begin(), bestClique.end());
    for(int cand : bestClique){
        output<<cand<<" ";
    }output<<"\n";
}
int structureBound(vector<int>candidates){
    int ans = 0;
    sort(candidates.begin(),candidates.end(),compDecreasingProfit);
    vector<vector<int>>colors;
    for(auto cand : candidates){

        int tot_color = colors.size();
        int pushed = 0;
        for(int col = 0;col<tot_color;col++){
            int isneighbour = 0;
            for(int vertex : colors[col]){
                if(graph.adjacency_matrix[vertex][cand]){
                    isneighbour = 1;
                    break;
                }
            }
            if(!isneighbour){
                colors[col].push_back(cand);
                pushed = 1;
                break;
            }
        }
        if(!pushed){
            vector<int>temp;
            temp.push_back(cand);
            ans += graph.profit[cand];
            colors.push_back(temp);
        }
    }
    return ans;
}

int resourceBound(vector<int>candidates, int remainingWeight){
    int ans = 0;
    int currWeight = 0;
    sort(candidates.begin(),candidates.end(),compDecreasingRatio);
    for(int cand : candidates){
        if(currWeight + graph.cost[cand] <= remainingWeight){
            ans += graph.profit[cand];
            currWeight += graph.cost[cand];
        }else{
            double ratio = (remainingWeight - currWeight) * 1.0 / graph.cost[cand] ; 
            ans += ceil ( ratio * graph.profit[cand]);
            return ans;
        }
    }
    return ans;
}

void findClique(vector<int>candidates, int currProfit, int currWeight){
    int ucolor = structureBound(candidates);
    if(currProfit + ucolor <= maxProfit)return;

    int uknap = resourceBound(candidates,graph.B - currWeight);
    if(currProfit + uknap <= maxProfit)return;

    while(candidates.size()){
        int v = candidates[candidates.size()-1];
        candidates.pop_back();
        if(currWeight + graph.cost[v] <= graph.B){
            currentClique.push_back(v);
            if(currProfit + graph.profit[v] >= maxProfit){
                maxProfit = currProfit + graph.profit[v];
                bestClique = currentClique;
            }
            vector<int>cand;
            for(int vertex : candidates){
                if(graph.adjacency_matrix[v][vertex]) cand.push_back(vertex);
            }
            findClique(cand, currProfit + graph.profit[v], currWeight + graph.cost[v]);
            currentClique.pop_back();
        }
    }   
}

int main(int argc, char* argv[]){
    if(argc!=3){
        cout<<"Usage: ./main input.txt output.txt";
        return -1;
    }

    MPI_Init(&argc, &argv);
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    MPI_Barrier(MPI_COMM_WORLD);  // everyone starts together
    double start = MPI_Wtime();


    string infile = argv[1];
    string outfile = argv[2];

    maxProfit = 0;

    readInput(infile,graph);

    /// using set for faster check of neighbour try vectors also 
    vector<int>candidates;
    for(int i = 0;i<graph.N;i++){
        candidates.push_back(i);
    }
    findClique(candidates,0,0);

    MPI_Barrier(MPI_COMM_WORLD);  // wait for slowest process
    if(rank == 0){
        writeOutput(argv[2]);
        double end = MPI_Wtime();
        cout << "Time: " << (end - start) << " seconds" << endl;
    }
    
    MPI_Finalize();
    return 0;
}