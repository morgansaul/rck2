// This file is a part of RCKangaroo software
// (c) 2024, RetiredCoder (RC)
// License: GPLv3, see "LICENSE.TXT" file
// https://github.com/RetiredC


#pragma once

#include "Ec.h"
#include <vector>

#define STATS_WND_SIZE      16

struct EcJMP
{
    EcPoint p;
    EcInt dist;
};

//96bytes size
struct TPointPriv
{
    u64 x[4];
    u64 y[4];
    u64 priv[4];
};

class RCGpuKang
{
private:
    bool StopFlag;
    std::vector<EcPoint> PntsToSolve;
    int Range; //in bits
    int DP; //in bits
    Ec ec;

    u32* DPs_out;
    TKparams Kparams;

    EcInt HalfRange;
    EcPoint PntHalfRange;
    EcPoint NegPntHalfRange;
    TPointPriv* RndPnts;
    EcJMP* EcJumps1;
    EcJMP* EcJumps2;
    EcJMP* EcJumps3;

    EcPoint PntA;
    EcPoint PntB;

    int cur_stats_ind;
    int SpeedStats[STATS_WND_SIZE];

    void GenerateRndPnts();
    void SetKangParams();

public:
    RCGpuKang();
    ~RCGpuKang();
    int CudaIndex;
    int mpCnt;
    bool IsOldGpu;
    int persistingL2CacheMaxSize;
    u64 dbg[1024];

    int CalcKangCnt();

    //executes in main thread
    bool Prepare(std::vector<EcPoint>& _PntsToSolve, int _Range, int _DP, EcJMP* _EcJumps1, EcJMP* _EcJumps2, EcJMP* _EcJumps3);
    void Execute();
    int GetStatsSpeed();
    void Stop() { StopFlag = true; };
};
