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
MPI_Win win_g;       // global RMA window handle (needed inside findClique)

// ── Bitset adjacency & pre-sorted orderings ───────────────────────────────────
// MAXN must cover the largest N in any test case (1024 gives ample headroom).
// bitset<MAXN> adj[MAXN]: adj[u] has bit v set iff edge (u,v) exists.
// Bitset ops (AND, OR, any()) run in N/64 machine words — far cheaper than
// iterating adjacency_matrix[u][v] one cell at a time.
const int MAXN = 1200;
bitset<MAXN> adj[MAXN];
vector<int> byProfit;  // all vertex indices sorted by profit desc (computed once)
vector<int> byRatio;   // all vertex indices sorted by profit/cost ratio desc
// ─────────────────────────────────────────────────────────────────────────────

// ── Instrumentation globals ──────────────────────────────────────────────────
long long findCliqueCallCount = 0;
struct ProfitEvent { int vertexId; int newProfit; double wallTime; };
vector<ProfitEvent> profitHistory;
double compStart_g;
int topLevelVertex_g = -1;
// ────────────────────────────────────────────────────────────────────────────

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
    graph.adjacency_matrix.resize(N,vector<int>(N,0));
    graph.adjacency_list.resize(N);
    for(int i=0;i<N;i++) input>>graph.profit[i]>>graph.cost[i];
    for(int i=0;i<E;i++){
        input>>u>>v;
        graph.adjacency_matrix[u][v]=1;
        graph.adjacency_matrix[v][u]=1;
        graph.adjacency_list[u].push_back(v);
        graph.adjacency_list[v].push_back(u);
        adj[u].set(v);
        adj[v].set(u);
    }
}
void writeOutput(string outfile){
    ofstream output(outfile);
    output<<maxProfit<<"\n";
    sort(bestClique.begin(),bestClique.end());
    for(int cand:bestClique) output<<cand<<" ";
    output<<"\n";
}
// exclude[c] = bitset of vertices that CANNOT join color class c
// (the union of adj[v] for every v already in class c).
// Checking "is cand adjacent to any member of class c?" = exclude[c][cand], O(1).
// Adding cand to class c: exclude[c] |= adj[cand], O(N/64).
// We iterate byProfit (pre-sorted desc) so no sort needed here.
// MAX_COLORS=64: 64*128 bytes=8 KB per stack frame — fits in L1 cache, safe with 8 MB stack.
// (Capping at 64 keeps the bound valid: vertices beyond cap are treated as new classes,
//  which is an overestimate — correct for pruning, just slightly less tight.)
static const int MAX_COLORS = 64;
int structureBound(const bitset<MAXN>& cands){
    int ans = 0;
    bitset<MAXN> exclude[MAX_COLORS];
    int nc = 0;
    int seen = 0, total = (int)cands.count();  // O(N/64) once; used for early exit
    for(int v : byProfit){
        if(!cands[v]) continue;
        bool placed = false;
        for(int c = 0; c < nc; c++){
            if(!exclude[c][v]){          // O(1): v not adjacent to any member of class c
                exclude[c] |= adj[v];   // O(N/64): add v's neighbours to exclusion set
                placed = true;
                break;
            }
        }
        if(!placed){
            ans += graph.profit[v];     // v is max-profit member of its new class
            if(nc < MAX_COLORS){
                exclude[nc] = adj[v];   // new class: exclude v's neighbours
                nc++;
            }
        }
        if(++seen == total) break;      // all candidates processed — stop scanning byProfit
    }
    return ans;
}
// Iterate byRatio (pre-sorted desc by profit/cost) and test cands with O(1) bitset lookup.
// No sort, no vector copies — just a tight loop over the global sorted order.
int resourceBound(const bitset<MAXN>& cands, int remainingWeight){
    int ans = 0, usedWeight = 0;
    for(int v : byRatio){
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
// cands: bitset of vertices still available to extend the current clique.
// Passing by value is O(MAXN/64) — 16 words for MAXN=1024 — much cheaper
// than copying a vector<int> of ~300 elements + heap allocation.
void findClique(bitset<MAXN> cands, int currProfit, int currWeight){
    findCliqueCallCount++;
    if((findCliqueCallCount & 0xFFFF) == 0)
        syncMaxProfit();

    if(currProfit + structureBound(cands) <= maxProfit) return;
    if(currProfit + resourceBound(cands, graph.B - currWeight) <= maxProfit) return;

    // Branch in DECREASING profit order (byProfit is pre-sorted globally).
    // Trying high-profit vertices first finds good solutions earlier → maxProfit
    // rises quickly → subsequent bounds prune more branches.
    // In the "proving phase" (after optimum is found), high-profit vertices shrink
    // the remaining candidate set's bound fastest, so we prove impossibility sooner.
    //
    // `remaining` tracks which candidates haven't been branched on yet; after
    // removing vertex v, the recursive call receives remaining ∩ adj[v] — only
    // vertices that are still in play AND adjacent to v (clique property).
    // We use a counter to exit the byProfit loop early once all candidates are done.
    bitset<MAXN> remaining = cands;
    int left = (int)cands.count();             // O(N/64); avoids scanning full byProfit
    for(int v : byProfit){
        if(!remaining[v]) continue;
        remaining.reset(v);

        if(currWeight + graph.cost[v] <= graph.B){
            currentClique.push_back(v);
            int newProfit = currProfit + graph.profit[v];
            if(newProfit > maxProfit){
                maxProfit = newProfit;
                bestClique = currentClique;
                if(maxProfit > localMaxProfit){
                    localMaxProfit = maxProfit;
                    profitHistory.push_back({topLevelVertex_g, maxProfit,
                                             MPI_Wtime()-compStart_g});
                }
            }
            findClique(remaining & adj[v], newProfit, currWeight + graph.cost[v]);
            currentClique.pop_back();
        }
        if(--left == 0) break;
    }
}

// ── One atomic RMA call: publishes our maxProfit and reads the global best.
//    MPI_Fetch_and_op atomically does:
//      old = win[rank0]
//      win[rank0] = max(win[rank0], maxProfit)   ← our value goes in
//    Then we check: if old > maxProfit, take the global value.
//    No rank waits for another. Everyone runs at its own pace.
static void syncMaxProfit(){
    int globalVal;
    MPI_Fetch_and_op(&maxProfit, &globalVal, MPI_INT, 0, 0, MPI_MAX, win_g);
    MPI_Win_flush(0, win_g);
    if(globalVal > maxProfit) maxProfit = globalVal;
}

// ── Lock-free dynamic work scheduler ─────────────────────────────────────────
// win_buf[1] is an atomic counter (offset 1 in the 2-int RMA window).
// Every rank atomically increments it and gets a unique work-queue index.
// No rank ever waits: whoever finishes a vertex immediately grabs the next one.
static int getNextWork(){
    int one = 1, myIdx;
    MPI_Fetch_and_op(&one, &myIdx, MPI_INT, 0, 1, MPI_SUM, win_g);
    MPI_Win_flush(0, win_g);
    return myIdx;   // caller processes workQueue[myIdx]
}
// ─────────────────────────────────────────────────────────────────────────────

// ── Greedy warm-start: deterministic clique heuristic run by every rank.
// Uses integer cross-multiplication to avoid floating-point non-determinism.
// O(N * avg_degree^2) — fast even on dense graphs.
static int greedyWarmStart(){
    int bestP = 0;
    vector<int> bestG;

    vector<int> order;
    for(int i = 0; i < graph.N; i++)
        if(graph.cost[i] <= graph.B) order.push_back(i);

    // Sort by profit/cost descending; break ties by vertex id (deterministic)
    sort(order.begin(), order.end(), [](int a, int b){
        long long lhs = (long long)graph.profit[a] * graph.cost[b];
        long long rhs = (long long)graph.profit[b] * graph.cost[a];
        if(lhs != rhs) return lhs > rhs;
        return a < b;
    });

    for(int seed : order){
        vector<int> clique = {seed};
        int W = graph.cost[seed], P = graph.profit[seed];

        // Neighbours of seed sorted by ratio desc (same tie-break)
        vector<int> cands = graph.adjacency_list[seed];
        sort(cands.begin(), cands.end(), [](int a, int b){
            long long lhs = (long long)graph.profit[a] * graph.cost[b];
            long long rhs = (long long)graph.profit[b] * graph.cost[a];
            if(lhs != rhs) return lhs > rhs;
            return a < b;
        });

        for(int nb : cands){
            if(W + graph.cost[nb] > graph.B) continue;
            bool ok = true;
            for(int c : clique)
                if(!graph.adjacency_matrix[nb][c]){ ok = false; break; }
            if(ok){ clique.push_back(nb); W += graph.cost[nb]; P += graph.profit[nb]; }
        }
        if(P > bestP){ bestP = P; bestG = clique; }
    }

    if(bestP > maxProfit){ maxProfit = bestP; bestClique = bestG; }
    return bestP;
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

    // Build byProfit and byRatio once (used by structureBound/resourceBound on every call)
    byProfit.resize(graph.N); byRatio.resize(graph.N);
    iota(byProfit.begin(), byProfit.end(), 0);
    iota(byRatio.begin(),  byRatio.end(),  0);
    sort(byProfit.begin(), byProfit.end(), compDecreasingProfit);
    sort(byRatio.begin(),  byRatio.end(),  compDecreasingRatio);

    // ── One-sided window: one int on rank 0 holds the global max ─────────────
    // MPI_Fetch_and_op lets any rank atomically publish its local max and
    // read back the current global max — no synchronisation, no waiting.
    double winSetupStart=MPI_Wtime();
    // Two-int RMA window:
    //   win_buf[0] = global maxProfit  (MPI_MAX operations, offset 0)
    //   win_buf[1] = work counter      (MPI_SUM +1 operations, offset 1)
    int* win_buf;
    MPI_Win_allocate(2 * sizeof(int), sizeof(int), MPI_INFO_NULL,
                     MPI_COMM_WORLD, &win_buf, &win_g);
    win_buf[0] = 0;   // maxProfit starts at 0
    win_buf[1] = 0;   // work counter starts at 0
    MPI_Barrier(MPI_COMM_WORLD);    // ensure window is initialised everywhere
    MPI_Win_lock_all(0, win_g);     // open passive-target epoch for full run
    // ─────────────────────────────────────────────────────────────────────────

    // ── Fix 2: greedy warm-start ─────────────────────────────────────────────
    // Every rank runs the same deterministic greedy heuristic (same graph →
    // same result).  MPI_Allreduce ensures all ranks share the global best
    // greedy profit as the initial pruning threshold before B&B begins.
    double greedyStart = MPI_Wtime();
    int myGreedyP = greedyWarmStart();
    int globalGreedyP;
    MPI_Allreduce(&myGreedyP, &globalGreedyP, 1, MPI_INT, MPI_MAX, MPI_COMM_WORLD);
    maxProfit = globalGreedyP;
    localMaxProfit = globalGreedyP; // all ranks claim the greedy result;
                                    // winner = rank 0 if B&B never beats it
    double greedyTime = MPI_Wtime() - greedyStart;
    // Publish our greedy profit to the RMA window before B&B starts
    win_buf[0] = globalGreedyP;
    // win_buf[1] (work counter) stays 0 — already initialised above
    MPI_Barrier(MPI_COMM_WORLD);
    // ─────────────────────────────────────────────────────────────────────────

    // ── Dynamic work queue ───────────────────────────────────────────────────
    // Sort vertices by decreasing profit: the optimal clique's anchor vertex
    // tends to have high profit, so it gets picked up first → 650 is discovered
    // and broadcast early → all subsequent work prunes aggressively.
    vector<int> workQueue(graph.N);
    iota(workQueue.begin(), workQueue.end(), 0);
    sort(workQueue.begin(), workQueue.end(), compDecreasingProfit);
    // ─────────────────────────────────────────────────────────────────────────

    double compStart=MPI_Wtime();
    double winSetupTime=compStart-winSetupStart;
    compStart_g=compStart;

    // Read the global window once before starting
    syncMaxProfit();

    struct VertexStat {
        int id, degree;
        double time;
        int profitBefore, profitAfter;
        long long callsBefore, callsAfter;
    };
    vector<VertexStat> vertexStats;

    // Dynamic scheduling: every rank grabs the next available vertex via an
    // atomic RMA fetch-and-add.  No rank ever sits idle waiting for a slow
    // neighbour — whoever finishes first takes the next item immediately.
    while(true){
        int idx = getNextWork();
        if(idx >= graph.N) break;           // all vertices assigned
        int i = workQueue[idx];

        double vStart=MPI_Wtime();
        int profBefore=maxProfit;
        long long callsBefore=findCliqueCallCount;

        bitset<MAXN> cands;
        for(int nbr:graph.adjacency_list[i])
            if(nbr > i) cands.set(nbr);
        topLevelVertex_g=i;
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

        double vEnd=MPI_Wtime();
        vertexStats.push_back({i,(int)graph.adjacency_list[i].size(),
                               vEnd-vStart, profBefore, maxProfit,
                               callsBefore, findCliqueCallCount});

        syncMaxProfit();   // publish our best and pull the global best
    }

    double compEnd=MPI_Wtime();
    double compTime=compEnd-compStart;

    // Close the one-sided epoch
    double rmaCleanupStart=MPI_Wtime();
    MPI_Win_unlock_all(win_g);
    MPI_Win_free(&win_g);
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
            cout<<"  Greedy time     : "<<greedyTime<<" s  (warm-start profit="<<globalGreedyP<<")"<<endl;
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