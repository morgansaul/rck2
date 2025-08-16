// This file is a part of RCKangaroo software
// (c) 2024, RetiredCoder (RC)
// License: GPLv3, see "LICENSE.TXT" file
// https://github.com/RetiredC


#pragma once

#include "defs.h"
#include "Ec.h"
#include <vector>

void Init_Utils();

//global vars
extern volatile bool gIsOpsLimit;
extern volatile int gSolvedKeyIndex;
extern volatile bool gSolved;
extern u64 gTotalOps;
extern u64 gTotalErrors;
extern u64 gWildCnt;
extern u64 gTameCnt;
extern u64 gTotalSolutions;
extern u64 gTotalCollisions;
extern volatile bool gUseGpu;

extern u64 Int_HalfRange[4];
extern u64 Int_TameOffset[4];
extern EcPoint Pnt_HalfRange;
extern EcPoint Pnt_NegHalfRange;
extern EcPoint g_G;

//utils.cpp
void LoadJumps(const char* file, EcJMP* arr);
void SaveJumps(const char* file, EcJMP* arr);

int Collision_SOTA(EcPoint& W, u64* priv, std::vector<EcPoint>& PntsToSolve, u64* priv_w, u64* Kp);
bool Collision_SOTA_Gen(EcPoint& T, u64* T_priv, u64* Kp);

EcPoint GetRandomG();

void SetKangParams(EcPoint Pnt, int range);

//common functions
void AddPointsToList(u32* data, int cnt, u64 ops_cnt);
