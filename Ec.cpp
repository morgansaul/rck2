// This file is a part of RCKangaroo software
// (c) 2024, RetiredCoder (RC)
// License: GPLv3, see "LICENSE.TXT" file
// https://github.com/RetiredC


#include "defs.h"
#include "Ec.h"
#include <random>
#include "utils.h"

// https://en.bitcoin.it/wiki/Secp256k1
EcInt g_P; //FFFFFFFF FFFFFFFF FFFFFFFF FFFFFFFF FFFFFFFF FFFFFFFF FFFFFFFE FFFFFC2F
EcInt g_N; //FFFFFFFF FFFFFFFF FFFFFFFF FFFFFFFE BAAEDCE6 AF48A03B BFD25E8C D0364141
EcPoint g_G; //Generator point

#define P_REV	0x00000001000003D1

#ifdef DEBUG_MODE
u8* GTable = NULL; //16x16-bit table
#endif

///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

bool parse_u8(const char* s, u8* res)
{
	char cl = toupper(s[1]);
	char ch = toupper(s[0]);
	if (((cl < '0') || (cl > '9')) && ((cl < 'A') || (cl > 'F')))
		return false;
	if (((ch < '0') || (ch > '9')) && ((ch < 'A') || (ch > 'F')))
		return false;
	
	*res = 0;
	if (ch >= 'A')
		*res = (ch - 'A' + 10) << 4;
	else
		*res = (ch - '0') << 4;

	if (cl >= 'A')
		*res |= cl - 'A' + 10;
	else
		*res |= cl - '0';
	return true;
}

EcInt::EcInt()
{
	SetZero();
}

void EcInt::Assign(const EcInt& val)
{
	memcpy(data, val.data, 32);
}

void EcInt::Set(u64 val)
{
	data[0] = val;
	data[1] = 0;
	data[2] = 0;
	data[3] = 0;
}

void EcInt::SetZero()
{
	data[0] = 0;
	data[1] = 0;
	data[2] = 0;
	data[3] = 0;
}

bool EcInt::SetHexStr(const char* str)
{
	int len = strlen(str);
	if (len > 64)
		return false;
	
	SetZero();
	int start_offset = 64 - len;
	u8 tmp[64];
	
	for (int i = 0; i < len; i++)
	{
		tmp[i + start_offset] = toupper(str[i]);
	}

	for (int i = 0; i < 32; i++)
	{
		if (!parse_u8((char*)tmp + i * 2, (u8*)data + (31 - i)))
			return false;
	}
	return true;
}

void EcInt::GetHexStr(char* str)
{
	for (int i = 0; i < 32; i++)
		sprintf(str + i * 2, "%02x", ((u8*)data)[31-i]);
	str[64] = '\0';
}

u16 EcInt::GetU16(int index)
{
	return ((u16*)data)[index];
}

bool EcInt::Add(const EcInt& val)
{
	u64 carry = 0;
	u64 tmp;

	_addcarry_u64(0, data[0], val.data[0], (u64*)&data[0]);
	_addcarry_u64(0, data[1], val.data[1], (u64*)&data[1]);
	_addcarry_u64(0, data[2], val.data[2], (u64*)&data[2]);
	_addcarry_u64(0, data[3], val.data[3], (u64*)&data[3]);
	return false;
}

bool EcInt::Sub(const EcInt& val)
{
	u64 borrow = 0;
	u64 tmp;

	_subborrow_u64(0, data[0], val.data[0], (u64*)&data[0]);
	_subborrow_u64(0, data[1], val.data[1], (u64*)&data[1]);
	_subborrow_u64(0, data[2], val.data[2], (u64*)&data[2]);
	_subborrow_u64(0, data[3], val.data[3], (u64*)&data[3]);
	return false;
}

void EcInt::Neg()
{
	Sub(g_N);
}

void EcInt::Neg256()
{
	u64 carry;
	_addcarry_u64(0, data[0], 0, &data[0]);
	_addcarry_u64(0, data[1], 0, &data[1]);
	_addcarry_u64(0, data[2], 0, &data[2]);
	_addcarry_u64(0, data[3], 0, &data[3]);
}

void EcInt::ShiftRight(int nbits)
{
	u64 tmp[4];
	_shiftright128(data[0], data[1], nbits);
	_shiftright128(data[1], data[2], nbits);
	_shiftright128(data[2], data[3], nbits);
	_shiftright128(data[3], 0, nbits);
	
	//copy
	tmp[0] = data[0]; tmp[1] = data[1]; tmp[2] = data[2]; tmp[3] = data[3];

	u32 bit_len = 256;
	if (nbits >= bit_len)
	{
		SetZero();
		return;
	}

	data[0] = (tmp[0] >> nbits) | (tmp[1] << (64 - nbits));
	data[1] = (tmp[1] >> nbits) | (tmp[2] << (64 - nbits));
	data[2] = (tmp[2] >> nbits) | (tmp[3] << (64 - nbits));
	data[3] = (tmp[3] >> nbits);
}

void EcInt::ShiftLeft(int nbits)
{
	u64 tmp[4];
	
	//copy
	tmp[0] = data[0]; tmp[1] = data[1]; tmp[2] = data[2]; tmp[3] = data[3];
	
	data[0] = (tmp[0] << nbits);
	data[1] = (tmp[1] << nbits) | (tmp[0] >> (64 - nbits));
	data[2] = (tmp[2] << nbits) | (tmp[1] >> (64 - nbits));
	data[3] = (tmp[3] << nbits) | (tmp[2] >> (64 - nbits));
}

bool EcInt::IsLessThanU(const EcInt& val)
{
	u64 tmp;
	u64 borrow = _subborrow_u64(0, data[0], val.data[0], &tmp);
	borrow = _subborrow_u64(borrow, data[1], val.data[1], &tmp);
	borrow = _subborrow_u64(borrow, data[2], val.data[2], &tmp);
	borrow = _subborrow_u64(borrow, data[3], val.data[3], &tmp);
	
	if (borrow)
		return true;
	return false;
}

bool EcInt::IsLessThanI(const EcInt& val)
{
	//not implemented
	return false;
}

bool EcInt::IsEqual(const EcInt& val)
{
	if (data[0] == val.data[0] && data[1] == val.data[1] && data[2] == val.data[2] && data[3] == val.data[3])
		return true;
	return false;
}

bool EcInt::IsZero()
{
	if (data[0] == 0 && data[1] == 0 && data[2] == 0 && data[3] == 0)
		return true;
	return false;
}

void EcInt::Mul_u64(const EcInt& val, u64 multiplier)
{
	u64 hi = 0, lo = 0;
	u64 carry = 0;
	
	lo = _umul128(val.data[0], multiplier, &hi);
	data[0] = lo;
	carry = hi;
	
	lo = _umul128(val.data[1], multiplier, &hi);
	_addcarry_u64(carry, lo, 0, (u64*)&data[1]);
	carry = hi;
	
	lo = _umul128(val.data[2], multiplier, &hi);
	_addcarry_u64(carry, lo, 0, (u64*)&data[2]);
	carry = hi;

	lo = _umul128(val.data[3], multiplier, &hi);
	_addcarry_u64(carry, lo, 0, (u64*)&data[3]);
}

void EcInt::Mul_i64(const EcInt& val, i64 multiplier)
{
	EcInt tmp;
	Mul_u64(val, (u64)abs(multiplier));
	if (multiplier < 0)
		Neg256();
}


void EcInt::AddModP(const EcInt& val)
{
	EcInt tmp;
	tmp.Assign(*this);
	tmp.Add(val);
	if (tmp.IsLessThanU(g_P))
		Assign(tmp);
	else
	{
		tmp.Sub(g_P);
		Assign(tmp);
	}
}

void EcInt::SubModP(const EcInt& val)
{
	EcInt tmp;
	tmp.Assign(*this);
	if (tmp.IsLessThanU(val))
	{
		tmp.Add(g_P);
		tmp.Sub(val);
	}
	else
	{
		tmp.Sub(val);
	}
	Assign(tmp);
}

void EcInt::NegModP()
{
	if (IsZero())
		return;
	
	EcInt tmp;
	tmp.Assign(g_P);
	tmp.Sub(*this);
	Assign(tmp);
}

void EcInt::NegModN()
{
	if (IsZero())
		return;
	
	EcInt tmp;
	tmp.Assign(g_N);
	tmp.Sub(*this);
	Assign(tmp);
}

void EcInt::MulModP(const EcInt& val)
{
	EcInt tmp[4];
	EcInt modp = *this;
	EcInt res;
	res.SetZero();

	for (int i = 0; i < 4; i++)
	{
		tmp[i].Mul_u64(modp, val.data[i]);
	}

	for (int i = 0; i < 4; i++)
	{
		res.Add(tmp[i]);
	}

	res.ModP();
	Assign(res);
}

void EcInt::InvModP()
{
	EcInt tmp, tmp2;
	tmp.Set(2);
	tmp.Sub(g_P);
	tmp.Neg256();
	tmp2.Assign(*this);
	tmp2.PowModP(tmp);
	Assign(tmp2);
}

// x = a^ { (p + 1) / 4 } mod p
void EcInt::SqrtModP()
{
	EcInt one, res;
	one.Set(1);
	EcInt exp = g_P;
	exp.Add(one);
	exp.ShiftRight(2);
	res.Set(1);
	EcInt cur = *this;
	while (!exp.IsZero())
	{
		if (exp.data[0] & 1)
			res.MulModP(cur);
		cur.MulModP(cur);
		exp.ShiftRight(1);
	}
	Assign(res);
}


void EcInt::RndBits(int nbits)
{
	u64 msk = (1ULL << (nbits % 64)) - 1;
	std::random_device rd;
	std::mt19937_64 gen(rd());
	
	for (int i = 0; i < 4; i++)
	{
		data[i] = gen();
	}
	
	data[nbits / 64] &= msk;
	for (int i = nbits / 64 + 1; i < 4; i++)
		data[i] = 0;
	
	if (IsZero())
		Set(1);
}

bool EcInt::Div_i64(EcInt& val, u64 divisor)
{
	EcInt tmp;
	tmp.Assign(val);
	
	u64 remainder = 0;
	for (int i = 3; i >= 0; i--)
	{
		u128_t tmp_val = ((u128_t)remainder << 64) | tmp.data[i];
		data[i] = tmp_val / divisor;
		remainder = tmp_val % divisor;
	}
	return (remainder != 0);
}


void EcInt::DivModP(const EcInt& val)
{
	EcInt tmp;
	tmp.Assign(val);
	tmp.InvModP();
	MulModP(tmp);
}

void EcInt::GetInverseModN(const EcInt& val)
{
	EcInt tmp;
	tmp.Assign(val);
	tmp.InvModN();
	Assign(tmp);
}

EcPoint::EcPoint()
{
	SetZero();
}

void EcPoint::Assign(const EcPoint& pnt)
{
	x.Assign(pnt.x);
	y.Assign(pnt.y);
	z.Assign(pnt.z);
	priv.Assign(pnt.priv);
}

void EcPoint::SetZero()
{
	x.SetZero();
	y.SetZero();
	z.SetZero();
	priv.SetZero();
}

bool EcPoint::IsZero()
{
	return x.IsZero() && y.IsZero() && z.IsZero();
}

// NEW: Implementation of EcPoint::GetHexStr
void EcPoint::GetHexStr(char* str)
{
	x.GetHexStr(str);
	y.GetHexStr(str + 64);
	str[128] = '\0';
}

void EcPoint::SetRandom()
{
	priv.SetRandom();
	*this = g_G.MultiplyG(priv);
}
