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
int maxProfit;       // local pruning threshold (may be raised from window)
int localMaxProfit;  // only what THIS rank actually found (for winner det.)
Graph graph;

// ── Instrumentation globals ──────────────────────────────────────────────────
long long findCliqueCallCount = 0;
struct ProfitEvent { int vertexId; int newProfit; double wallTime; };
vector<ProfitEvent> profitHistory;
double compStart_g;
int topLevelVertex_g = -1;
// ────────────────────────────────────────────────────────────────────────────

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
    graph.adjacency_matrix.resize(N,vector<int>(N,0));
    graph.adjacency_list.resize(N);
    for(int i=0;i<N;i++) input>>graph.profit[i]>>graph.cost[i];
    for(int i=0;i<E;i++){
        input>>u>>v;
        graph.adjacency_matrix[u][v]=1;
        graph.adjacency_matrix[v][u]=1;
        graph.adjacency_list[u].push_back(v);
        graph.adjacency_list[v].push_back(u);
    }
}
void writeOutput(string outfile){
    ofstream output(outfile);
    output<<maxProfit<<"\n";
    sort(bestClique.begin(),bestClique.end());
    for(int cand:bestClique) output<<cand<<" ";
    output<<"\n";
}
int structureBound(vector<int>candidates){
    int ans=0;
    sort(candidates.begin(),candidates.end(),compDecreasingProfit);
    vector<vector<int>>colors;
    for(auto cand:candidates){
        int tot_color=colors.size(), pushed=0;
        for(int col=0;col<tot_color;col++){
            int isneighbour=0;
            for(int vertex:colors[col]){
                if(graph.adjacency_matrix[vertex][cand]){ isneighbour=1; break; }
            }
            if(!isneighbour){ colors[col].push_back(cand); pushed=1; break; }
        }
        if(!pushed){
            vector<int>temp; temp.push_back(cand);
            ans+=graph.profit[cand];
            colors.push_back(temp);
        }
    }
    return ans;
}
int resourceBound(vector<int>candidates, int remainingWeight){
    int ans=0, currWeight=0;
    sort(candidates.begin(),candidates.end(),compDecreasingRatio);
    for(int cand:candidates){
        if(currWeight+graph.cost[cand]<=remainingWeight){
            ans+=graph.profit[cand]; currWeight+=graph.cost[cand];
        } else {
            double ratio=(remainingWeight-currWeight)*1.0/graph.cost[cand];
            ans+=ceil(ratio*graph.profit[cand]);
            return ans;
        }
    }
    return ans;
}
void findClique(vector<int>candidates, int currProfit, int currWeight){
    findCliqueCallCount++;
    int ucolor=structureBound(candidates);
    if(currProfit+ucolor<=maxProfit) return;
    int uknap=resourceBound(candidates,graph.B-currWeight);
    if(currProfit+uknap<=maxProfit) return;
    while(candidates.size()){
        int v=candidates[candidates.size()-1];
        candidates.pop_back();
        if(currWeight+graph.cost[v]<=graph.B){
            currentClique.push_back(v);
            if(currProfit+graph.profit[v]>=maxProfit){
                maxProfit=currProfit+graph.profit[v];
                bestClique=currentClique;
                if(maxProfit>localMaxProfit){
                    localMaxProfit=maxProfit;
                    profitHistory.push_back({topLevelVertex_g, maxProfit,
                                             MPI_Wtime()-compStart_g});
                }
            }
            vector<int>cand;
            for(int vertex:candidates)
                if(graph.adjacency_matrix[v][vertex]) cand.push_back(vertex);
            findClique(cand,currProfit+graph.profit[v],currWeight+graph.cost[v]);
            currentClique.pop_back();
        }
    }
}

// ── One atomic RMA call: publishes our maxProfit and reads the global best.
//    MPI_Fetch_and_op atomically does:
//      old = win[rank0]
//      win[rank0] = max(win[rank0], maxProfit)   ← our value goes in
//    Then we check: if old > maxProfit, take the global value.
//    No rank waits for another. Everyone runs at its own pace.
static void syncMaxProfit(MPI_Win win){
    int globalVal;
    MPI_Fetch_and_op(&maxProfit, &globalVal, MPI_INT, 0, 0, MPI_MAX, win);
    MPI_Win_flush(0, win);          // ensure operation completed at rank 0
    if(globalVal > maxProfit) maxProfit = globalVal;
}

int main(int argc, char* argv[]){
    if(argc!=3){ cout<<"Usage: ./main input.txt output.txt"; return -1; }

    MPI_Init(&argc,&argv);
    int rank,size;
    MPI_Comm_rank(MPI_COMM_WORLD,&rank);
    MPI_Comm_size(MPI_COMM_WORLD,&size);

    MPI_Barrier(MPI_COMM_WORLD);
    double start=MPI_Wtime();

    maxProfit=0; localMaxProfit=0;

    double ioStart=MPI_Wtime();
    readInput(argv[1],graph);
    double ioEnd=MPI_Wtime();

    // ── One-sided window: one int on rank 0 holds the global max ─────────────
    // MPI_Fetch_and_op lets any rank atomically publish its local max and
    // read back the current global max — no synchronisation, no waiting.
    double winSetupStart=MPI_Wtime();
    MPI_Win win;
    int* win_buf;
    MPI_Win_allocate(sizeof(int), sizeof(int), MPI_INFO_NULL,
                     MPI_COMM_WORLD, &win_buf, &win);
    *win_buf = 0;
    MPI_Barrier(MPI_COMM_WORLD);    // ensure window is initialised everywhere
    MPI_Win_lock_all(0, win);       // open passive-target epoch for full run
    // ─────────────────────────────────────────────────────────────────────────

    double compStart=MPI_Wtime();
    double winSetupTime=compStart-winSetupStart;
    compStart_g=compStart;

    // Read the global window once before starting (in case other ranks already
    // wrote something, though usually all start at 0).
    syncMaxProfit(win);

    struct VertexStat {
        int id, degree;
        double time;
        int profitBefore, profitAfter;
        long long callsBefore, callsAfter;
    };
    vector<VertexStat> vertexStats;

    for(int i=rank; i<graph.N; i+=size){
        double vStart=MPI_Wtime();
        int profBefore=maxProfit;
        long long callsBefore=findCliqueCallCount;

        vector<int>candidates;
        for(int nbr:graph.adjacency_list[i]){
            if(nbr > i)candidates.push_back(nbr);
        }
        topLevelVertex_g=i;
        currentClique.push_back(i);
        if(graph.cost[i] <= graph.B){                                             
            if(graph.profit[i] > maxProfit){                                      
                maxProfit = graph.profit[i];                                      
                localMaxProfit = maxProfit;                                       
                bestClique = currentClique;                                       
            }                                                                     
            findClique(candidates,graph.profit[i],graph.cost[i]);                 
        }
        currentClique.pop_back();

        double vEnd=MPI_Wtime();
        vertexStats.push_back({i,(int)graph.adjacency_list[i].size(),
                               vEnd-vStart, profBefore, maxProfit,
                               callsBefore, findCliqueCallCount});

        // ── KEY: non-blocking async sync after each vertex ────────────────────
        // Atomically: win[0] = max(win[0], maxProfit), and we read back
        // the old win[0] to see if any other rank found something better.
        // This rank does NOT wait for anyone else — it just continues to
        // its next vertex with the most up-to-date global best available.
        syncMaxProfit(win);
        // ─────────────────────────────────────────────────────────────────────
    }

    double compEnd=MPI_Wtime();
    double compTime=compEnd-compStart;

    // Close the one-sided epoch
    double rmaCleanupStart=MPI_Wtime();
    MPI_Win_unlock_all(win);
    MPI_Win_free(&win);
    double rmaCleanupTime=MPI_Wtime()-rmaCleanupStart;

    double barrierStart=MPI_Wtime();
    MPI_Barrier(MPI_COMM_WORLD);
    double barrierWait=MPI_Wtime()-barrierStart;

    // Winner = lowest rank that locally found the global best
    double commStart=MPI_Wtime();
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
    double commTime=MPI_Wtime()-commStart;

    // ── Per-rank diagnostic output ────────────────────────────────────────────
    for(int r=0;r<size;r++){
        MPI_Barrier(MPI_COMM_WORLD);
        if(rank==r){
            vector<VertexStat> sorted=vertexStats;
            sort(sorted.begin(),sorted.end(),
                 [](const VertexStat&a,const VertexStat&b){ return a.time>b.time; });

            cout<<"\n=== Rank "<<r<<" ==="<<endl;
            cout<<"  I/O time        : "<<(ioEnd-ioStart)<<" s"<<endl;
            cout<<"  Win setup time  : "<<winSetupTime<<" s"<<endl;
            cout<<"  Compute time    : "<<compTime<<" s"<<endl;
            cout<<"  RMA cleanup time: "<<rmaCleanupTime<<" s"<<endl;
            cout<<"  Barrier wait    : "<<barrierWait<<" s"<<endl;
            cout<<"  Comm time       : "<<commTime<<" s"<<endl;
            cout<<"  Total accounted : "<<(ioEnd-ioStart)+winSetupTime+compTime+rmaCleanupTime+barrierWait+commTime<<" s"<<endl;
            cout<<"  Nodes assigned  : "<<vertexStats.size()<<endl;
            cout<<"  findClique calls: "<<findCliqueCallCount<<endl;
            cout<<"  localMaxProfit  : "<<localMaxProfit<<endl;

            cout<<"  Top-5 slowest vertices (id, degree, time, calls, profitBefore->After):"<<endl;
            for(int k=0;k<min((int)sorted.size(),5);k++){
                auto&s=sorted[k];
                cout<<"    v="<<s.id
                    <<"  deg="<<s.degree
                    <<"  time="<<s.time<<"s"
                    <<"  calls="<<(s.callsAfter-s.callsBefore)
                    <<"  profit "<<s.profitBefore<<"->"<<s.profitAfter<<endl;
            }
            cout<<"  Profit improvement history:"<<endl;
            if(profitHistory.empty()) cout<<"    (none)"<<endl;
            else for(auto&ev:profitHistory)
                cout<<"    v="<<ev.vertexId<<"  profit="<<ev.newProfit
                    <<"  at "<<ev.wallTime<<"s"<<endl;
            cout.flush();
        }
    }
    MPI_Barrier(MPI_COMM_WORLD);
    // ─────────────────────────────────────────────────────────────────────────

    if(rank==0){
        writeOutput(argv[2]);
        double end=MPI_Wtime();
        cout<<"\nMPI Time (rank 0 wall): "<<(end-start)<<" seconds"<<endl;
    }

    MPI_Finalize();
    return 0;
}