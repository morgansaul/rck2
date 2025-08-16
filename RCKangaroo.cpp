// This file is a part of RCKangaroo software
// (c) 2024, RetiredCoder (RC)
// License: GPLv3, see "LICENSE.TXT" file
// https://github.com/RetiredC


#include <iostream>
#include <vector>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <time.h>
#include <inttypes.h>
#include <stdint.h>
#include <random>

#include "cuda_runtime.h"
#include "cuda.h"

#include "defs.h"
#include "utils.h"
#include "GpuKang.h"

#ifndef _WIN32
#include <unistd.h>
#endif

time_t program_start_time = time(NULL);  // Capture the start time

EcJMP EcJumps1[JMP_CNT];
EcJMP EcJumps2[JMP_CNT];
EcJMP EcJumps3[JMP_CNT];

RCGpuKang* GpuKangs[MAX_GPU_CNT];
int GpuCnt;
volatile long ThrCnt;
volatile bool gSolved;

EcInt Int_HalfRange;
EcPoint Pnt_HalfRange;
EcPoint Pnt_NegHalfRange;
EcInt Int_TameOffset;
Ec ec;

CriticalSection csAddPoints;
TFastBase gDPTable;

bool gGenMode = false;
int gRange = 0;
int gDP = 0;
bool gSearchMode = false;
u64 gStart = 0;
std::vector<EcPoint> gPntsToSolve;
std::vector<std::string> gPntsToSolveStr;
char gTameFileName[1024];
int gTameMax;

int gTotalErrors = 0;

void trim_leading_zeros(char* s)
{
	int i, j;
	for (i = 0; s[i] == '0' || s[i] == ' '; i++);
	if (s[i] == '\0')
	{
		s[0] = '0';
		s[1] = '\0';
		return;
	}
	for (j = 0; s[i]; j++)
		s[j] = s[i++];
	s[j] = '\0';
}

void CheckArgs(int argc, char* argv[])
{
	for (int i = 1; i < argc; i++)
	{
		if (strcmp(argv[i], "-start") == 0)
		{
			i++;
			if (i >= argc) break;
			gStart = strtoull(argv[i], NULL, 16);
		}
		if (strcmp(argv[i], "-dp") == 0)
		{
			i++;
			if (i >= argc) break;
			gDP = atoi(argv[i]);
		}
		if (strcmp(argv[i], "-range") == 0)
		{
			i++;
			if (i >= argc) break;
			gRange = atoi(argv[i]);
		}
		if (strcmp(argv[i], "-gen") == 0)
		{
			gGenMode = true;
		}
		if (strcmp(argv[i], "-pubkey") == 0)
		{
			i++;
			if (i >= argc) break;
			gPntsToSolveStr.push_back(std::string(argv[i]));
		}
		if (strcmp(argv[i], "-tames") == 0)
		{
			i++;
			if (i >= argc) break;
			strcpy(gTameFileName, argv[i]);
		}
		if (strcmp(argv[i], "-max") == 0)
		{
			i++;
			if (i >= argc) break;
			gTameMax = atoi(argv[i]);
		}
	}
}

void PrintHelp()
{
	printf("RCKangaroo usage:\n");
	printf("  -dp <value>      Set the DP (distinguished point) value (e.g., 16).\n");
	printf("  -range <value>   Set the search range in bits (e.g., 78).\n");
	printf("  -pubkey <key>    Add a public key to solve. This option can be used multiple times.\n");
	printf("  -start <value>   Set the starting point for the wild kangaroos (in hex).\n");
	printf("  -tames <file>    Use or generate a tame file.\n");
	printf("  -max <value>     Set the max number of tames to generate.\n");
	printf("  -gen             Enable tame generation mode.\n");
	printf("\nExample to solve multiple keys:\n");
	printf("RCKangaroo.exe -dp 16 -range 84 -pubkey 0329c457... -pubkey 021234...\n");
	printf("RCKangaroo.exe -gen -dp 16 -range 76 -tames tames76.dat -max 10\n");
}


//this function adds new points to the DP Table, must be locked
void AddPointsToList(u32* data, int cnt, u64 ops_cnt)
{
	u8* rec_data = (u8*)data;
	u64* p_in;
	
	for (int i = 0; i < cnt; i++)
	{
		p_in = (u64*)rec_data;
		
		EcPoint p;
		p.x.data[0] = p_in[0]; p.x.data[1] = p_in[1]; p.x.data[2] = p_in[2]; p.x.data[3] = p_in[3];
		p.y.data[0] = p_in[4]; p.y.data[1] = p_in[5]; p.y.data[2] = p_in[6]; p.y.data[3] = p_in[7];
		
		EcInt pk;
		pk.data[0] = p_in[8]; pk.data[1] = p_in[9]; pk.data[2] = p_in[10]; pk.data[3] = p_in[11];

		u8 tmp[96];
		memcpy(tmp, p.x.data, 32);
		memcpy(tmp + 32, p.y.data, 32);
		memcpy(tmp + 64, pk.data, 32);
		
		u8* found = gDPTable.FindOrAddDataBlock(tmp);
		if (found)
		{
			//collision
			EcInt pk2;
			pk2.data[0] = ((u64*)found)[8];
			pk2.data[1] = ((u64*)found)[9];
			pk2.data[2] = ((u64*)found)[10];
			pk2.data[3] = ((u64*)found)[11];
			
			printf("Collision found after %" PRIu64 " jumps\n", ops_cnt);
			
			//solve
			EcInt diff;
			diff.SetZero();

			//find which one is smaller
			if (pk.IsLessThanU(pk2))
				diff.Assign(pk2);
			else
				diff.Assign(pk);
			
			//subtract smaller from larger
			if (pk.IsLessThanU(pk2))
				diff.Sub(pk);
			else
				diff.Sub(pk2);
			
			char s[100];
			diff.GetHexStr(s);
			trim_leading_zeros(s);
			printf("Collision private key diff: %s\n", s);
			
			//save to file
			FILE* fp = fopen("COLLISIONS.TXT", "a");
			if (fp)
			{
				fprintf(fp, "%s\n", s);
				fclose(fp);
			}
			gSolved = true;
		}
		rec_data += GPU_DP_SIZE;
	}
}

void StartThreads()
{
	ThrCnt = GpuCnt;
	for (int i = 0; i < GpuCnt; i++)
		GpuKangs[i]->Start();
}

void StopThreads()
{
	for (int i = 0; i < GpuCnt; i++)
		GpuKangs[i]->Stop();
	//wait for all threads to stop
	while (ThrCnt > 0)
		Sleep(10);
}

// FIX: Changed signature to const std::vector<EcPoint>&
bool SolvePoint(const std::vector<EcPoint>& PntsToSolve, int Range, int DP, EcInt* pk_found)
{
	gSolved = false;

	for (int i = 0; i < GpuCnt; i++)
		GpuKangs[i]->Prepare(PntsToSolve, Range, DP);

	StartThreads();

	while (!gSolved)
	{
		Sleep(500);
		//check for solution
		for (int i = 0; i < GpuCnt; i++)
		{
			int solved_key_index = GpuKangs[i]->CheckForSolution();
			if (solved_key_index != -1)
			{
				gSolved = true;
				EcInt wild_priv = GpuKangs[i]->GetWildPriv();
				EcInt tame_priv = PntsToSolve[solved_key_index].priv;
				pk_found->Assign(wild_priv);
				pk_found->Add(tame_priv);
				break;
			}
		}

		if (gSolved)
			break;
		
		printf("\n");
		
		for (int i = 0; i < GpuCnt; i++)
			GpuKangs[i]->PrintStats();
	}
	
	StopThreads();
	
	return gSolved;
}

int main(int argc, char* argv[])
{
	CheckArgs(argc, argv);

	if (gGenMode)
		gDPTable.Header[0] = 'K';

	cudaGetDeviceCount(&GpuCnt);

	if (!GpuCnt)
	{
		printf("No CUDA capable devices found!\r\n");
		return 0;
	}

	printf("Found %d CUDA devices\r\n\r\n", GpuCnt);
	for (int i = 0; i < GpuCnt; i++)
	{
		cudaDeviceProp prop;
		cudaGetDeviceProperties(&prop, i);
		printf("Device %d: %s, Compute Capability %d.%d\n", i, prop.name, prop.major, prop.minor);
	}
	printf("\r\n");

	//init
	Ec::Init();

	//load DP table if it exists
	if (IsFileExist(gTameFileName))
	{
		printf("Loading DP table...\n");
		gDPTable.LoadFromFile(gTameFileName);
		printf("Loaded %" PRIu64 " tames\n", gDPTable.GetBlockCnt());
	}
	
	//generate jumps
	ec.GenerateJumps(EcJumps1, JMP_CNT, gRange, 1, 0, 0);

	for (int i = 0; i < GpuCnt; i++)
		GpuKangs[i] = new RCGpuKang(i);

	//if we want to solve a single key
	if (!gPntsToSolveStr.empty())
	{
		for (const auto& key_str : gPntsToSolveStr)
		{
			EcPoint pnt;
			if (!pnt.SetHexStr(key_str.c_str()))
			{
				printf("Error: Invalid public key format for %s\n", key_str.c_str());
				return 1;
			}
			gPntsToSolve.push_back(pnt);
		}
		
		// This is the new logic for solving a list of keys
		for (int i = 0; i < GpuCnt; i++) {
			GpuKangs[i]->Prepare(gPntsToSolve, gRange, gDP, EcJumps1, EcJumps2, EcJumps3);
		}

		StartThreads();

		int solved_key_index = -1;
		while (solved_key_index == -1)
		{
			Sleep(500);
			for (int i = 0; i < GpuCnt; i++)
			{
				solved_key_index = GpuKangs[i]->CheckForSolution();
				if (solved_key_index != -1)
					break;
			}
			if (solved_key_index != -1) {
				printf("A solution was found for key index %d.\n", solved_key_index);
				break;
			}
		}

		StopThreads();
		
		EcInt pk_found;
		EcPoint PntToSolve = gPntsToSolve[solved_key_index];
		SolvePoint(gPntsToSolve, gRange, gDP, &pk_found);
		char s[100];
		pk_found.GetHexStr(s);
		trim_leading_zeros(s);
		printf("\r\nPRIVATE KEY: %s\r\n\r\n", s);
		FILE* fp = fopen("RESULTS.TXT", "a");
		if (fp)
		{
			fprintf(fp, "PRIVATE KEY: %s\n", s);
			fclose(fp);
		}
		else
		{
			printf("WARNING: Cannot save the key to RESULTS.TXT!\r\n");
			while (1)
				Sleep(100);
		}
	}
	else
	{
		if (gGenMode)
			printf("\r\nTAMES GENERATION MODE\r\n");
		else
			printf("\r\nBENCHMARK MODE\r\n");
		
		while (1)
		{
			EcInt pk, pk_found;
			EcPoint PntToSolve;
			
			if (!gRange)
				gRange = 78;
			if (!gDP)
				gDP = 16;
			
			pk.RndBits(gRange);
			PntToSolve = ec.MultiplyG(pk);
			
			// FIX: Creating a local vector to pass to SolvePoint
			std::vector<EcPoint> tempPntsToSolve = { PntToSolve };
			if (!SolvePoint(tempPntsToSolve, gRange, gDP, &pk_found))
				break; // Stop if SolvePoint returns false
			
			char s[100];
			PntToSolve.GetHexStr(s);
			trim_leading_zeros(s);
			printf("Pubkey: %s\n", s);
			
			pk_found.GetHexStr(s);
			trim_leading_zeros(s);
			printf("Private Key: %s\n", s);
			
			Sleep(1000);
		}
	}

	// Cleanup
	for (int i = 0; i < GpuCnt; i++)
		delete GpuKangs[i];

	printf("Press any key to exit...\n");
	getchar();

	return 0;
}
