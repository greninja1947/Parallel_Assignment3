#include <bits/stdc++.h>
#include <mpi.h>
using namespace std;
const int MAXN = 1201;
struct Graph{
    int N,E,B;
    vector<int>profit;
    vector<int>cost;
    bitset<MAXN>adj[MAXN];
    vector<vector<int>>adjacency_list;
    vector<int> byProfit;  
    vector<int> byRatio;
};

vector<int>bestClique;
vector<int>currentClique;
int maxProfit;       // local pruning threshold (may be raised from window)
int localMaxProfit;  // only what THIS rank actually found (for winner det.)
Graph graph;
MPI_Win win_g;       // global RMA window handle (needed inside findClique)


long long findCliqueCallCount = 0;

static void syncMaxProfit(); // forward declaration (defined after findClique)

bool compDecreasingProfit(int a, int b){
    return graph.profit[a] > graph.profit[b];
}
bool compDecreasingRatio(int a, int b){
    double r1=(graph.profit[a]*1.0)/graph.cost[a];
    double r2=(graph.profit[b]*1.0)/graph.cost[b];
    return r1>r2;
}
void readInput(string infile, Graph& graph){
    ifstream input(infile);
    if(!input){ cout<<"could not read file\n"; return; }
    int u,v,N,E,B;
    input>>N>>E>>B;
    graph.N=N; graph.E=E; graph.B=B;
    graph.profit.resize(N);
    graph.cost.resize(N);
    graph.adjacency_list.resize(N);
    for(int i=0;i<N;i++) input>>graph.profit[i]>>graph.cost[i];
    for(int i=0;i<E;i++){
        input>>u>>v;
        graph.adjacency_list[u].push_back(v);
        graph.adjacency_list[v].push_back(u);
        graph.adj[u].set(v);
        graph.adj[v].set(u);
    }
    graph.byProfit.resize(N); 
    graph.byRatio.resize(N);
    iota(graph.byProfit.begin(), graph.byProfit.end(), 0);
    sort(graph.byProfit.begin(), graph.byProfit.end(), compDecreasingProfit);
    iota(graph.byRatio.begin(),  graph.byRatio.end(),  0);
    sort(graph.byRatio.begin(),  graph.byRatio.end(),  compDecreasingRatio);
}
void writeOutput(string outfile){
    ofstream output(outfile);
    output<<maxProfit<<"\n";
    sort(bestClique.begin(),bestClique.end());
    for(int cand:bestClique) output<<cand<<" ";
    output<<"\n";
}

int structureBound(const bitset<MAXN>& cands){
    int ans = 0;
    vector<bitset<MAXN>> exclude;
    int seen = 0, total = (int)cands.count();
    for(int v : graph.byProfit){
        if(!cands[v]) continue;
        bool placed = false;
        seen+=1;
        int numberOfColors = exclude.size();
        for(int c = 0; c < numberOfColors; c++){
            if(!exclude[c][v]){
                exclude[c] |= graph.adj[v];
                placed = true;
                break;
            }
        }
        if(!placed){
            ans += graph.profit[v];
            exclude.push_back(graph.adj[v]);
        }
        if(seen == total) break;
    }
    return ans;
}

int resourceBound(const bitset<MAXN>& cands, int remainingWeight){
    int usedWeight = 0;
    int ans = 0; 
    for(int v : graph.byRatio){
        if(!cands[v]) continue;
        int c = graph.cost[v];
        if(usedWeight + c <= remainingWeight){
            ans += graph.profit[v];
            usedWeight += c;
        } else {
            ans += (int)((double)(remainingWeight - usedWeight) * graph.profit[v] / c);
            return ans;
        }
    }
    return ans;
}

void findClique(bitset<MAXN> cands, int currProfit, int currWeight){
    findCliqueCallCount++;
    if((findCliqueCallCount & 0xFFFF) == 0)
        syncMaxProfit();

    if(currProfit + structureBound(cands) <= maxProfit) return;
    if(currProfit + resourceBound(cands, graph.B - currWeight) <= maxProfit) return;


    bitset<MAXN> remaining = cands;
    int left = (int)cands.count();            
    for(int v : graph.byProfit){
        if(!remaining[v]) continue;
        remaining.reset(v);
        left = left - 1;
        if(currWeight + graph.cost[v] <= graph.B){
            currentClique.push_back(v);
            int newProfit = currProfit + graph.profit[v];
            if(newProfit > maxProfit){
                maxProfit = newProfit;
                bestClique = currentClique;
                if(maxProfit > localMaxProfit){
                    localMaxProfit = maxProfit;
                }
            }
            findClique(remaining & graph.adj[v], newProfit, currWeight + graph.cost[v]);
            currentClique.pop_back();
        }
        if(left == 0) break;
    }
}


static void syncMaxProfit(){
    int globalVal;
    MPI_Fetch_and_op(&maxProfit, &globalVal, MPI_INT, 0, 0, MPI_MAX, win_g);
    MPI_Win_flush(0, win_g);
    if(globalVal > maxProfit) maxProfit = globalVal;
}


static int getNextWork(){
    int one = 1, myIdx;
    MPI_Fetch_and_op(&one, &myIdx, MPI_INT, 0, 1, MPI_SUM, win_g);
    MPI_Win_flush(0, win_g);
    return myIdx;  
}

static int greedyWarmStart(){
    int bestP = 0;
    vector<int> bestG;
    for(int seed : graph.byRatio){
        if(graph.cost[seed] > graph.B) continue;
        vector<int> clique = {seed};
        int W = graph.cost[seed], P = graph.profit[seed];
        vector<int> cands = graph.adjacency_list[seed];
        sort(cands.begin(), cands.end(), compDecreasingProfit);
        for(int nb : cands){
            if(W + graph.cost[nb] > graph.B) continue;
            bool ok = true;
            for(int c : clique)
                if(!graph.adj[nb][c]){ ok = false; break; }
            if(ok){ clique.push_back(nb); W += graph.cost[nb]; P += graph.profit[nb]; }
        }
        if(P > bestP){ bestP = P; bestG = clique; }
    }
    if(bestP > maxProfit){ maxProfit = bestP; bestClique = bestG; }
    return bestP;
}

int main(int argc, char* argv[]){
    if(argc!=3){ cout<<"Usage: ./main input.txt output.txt"; return -1; }

    string infile = argv[1];
    string outfile = argv[2];
    MPI_Init(&argc,&argv);
    int rank,size;
    MPI_Comm_rank(MPI_COMM_WORLD,&rank);
    MPI_Comm_size(MPI_COMM_WORLD,&size);

    double start;
    if(rank ==0 ){
        start = MPI_Wtime();
    }
    maxProfit=0; localMaxProfit=0;

    readInput(infile,graph);




    int* win_buf;
    MPI_Win_allocate(2 * sizeof(int), sizeof(int), MPI_INFO_NULL,
                     MPI_COMM_WORLD, &win_buf, &win_g);
    win_buf[0] = 0;   
    win_buf[1] = 0;   
    MPI_Barrier(MPI_COMM_WORLD);   
    MPI_Win_lock_all(0, win_g);    
 
    int myGreedyP = greedyWarmStart();
    int globalGreedyP;
    MPI_Allreduce(&myGreedyP, &globalGreedyP, 1, MPI_INT, MPI_MAX, MPI_COMM_WORLD);
    maxProfit = globalGreedyP;
    localMaxProfit = globalGreedyP; 
    win_buf[0] = globalGreedyP;
    MPI_Barrier(MPI_COMM_WORLD);

    vector<int> workQueue(graph.N);
    iota(workQueue.begin(), workQueue.end(), 0);
    sort(workQueue.begin(), workQueue.end(), compDecreasingProfit);

   
    syncMaxProfit();


    while(true){
        int idx = getNextWork();
        if(idx >= graph.N) break;           // all vertices assigned
        int i = workQueue[idx];

       
        bitset<MAXN> cands;
        for(int nbr:graph.adjacency_list[i])
            if(nbr > i) cands.set(nbr);
        currentClique.push_back(i);
        if(graph.cost[i] <= graph.B){
            if(graph.profit[i] > maxProfit){
                maxProfit = graph.profit[i];
                localMaxProfit = maxProfit;
                bestClique = currentClique;
            }
            findClique(cands, graph.profit[i], graph.cost[i]);
        }
        currentClique.pop_back();

       

        syncMaxProfit();  
    }


    MPI_Win_unlock_all(win_g);
    MPI_Win_free(&win_g);

    MPI_Barrier(MPI_COMM_WORLD);

    int finalGlobalMax;
    MPI_Allreduce(&localMaxProfit, &finalGlobalMax, 1, MPI_INT, MPI_MAX, MPI_COMM_WORLD);

    int winner;
    int myRank=(localMaxProfit==finalGlobalMax) ? rank : size;
    MPI_Allreduce(&myRank, &winner, 1, MPI_INT, MPI_MIN, MPI_COMM_WORLD);

    if(winner!=0){
        if(rank==winner){
            int cliqueSize=bestClique.size();
            MPI_Send(&cliqueSize,1,MPI_INT,0,0,MPI_COMM_WORLD);
            MPI_Send(bestClique.data(),cliqueSize,MPI_INT,0,1,MPI_COMM_WORLD);
        }
        if(rank==0){
            int cliqueSize; MPI_Status status;
            MPI_Recv(&cliqueSize,1,MPI_INT,winner,0,MPI_COMM_WORLD,&status);
            bestClique.resize(cliqueSize);
            MPI_Recv(bestClique.data(),cliqueSize,MPI_INT,winner,1,MPI_COMM_WORLD,&status);
            maxProfit=finalGlobalMax;
        }
    }
   

    if(rank==0){
        writeOutput(outfile);
        double end=MPI_Wtime();
        cout<<"\nMPI Time : "<<(end-start)<<" seconds"<<endl;
    }

    MPI_Finalize();
    return 0;
}