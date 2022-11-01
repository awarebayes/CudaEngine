//
// Created by dev on 10/31/22.
//

#include "const.h"

int USE_THREADS = 32;

int get_grid_size(int n)
{
	return (n  / USE_THREADS) + 1;
}

int get_block_size(int n)
{
	return USE_THREADS;
}