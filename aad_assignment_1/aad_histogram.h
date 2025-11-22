#ifndef AAD_HISTOGRAM_H
#define AAD_HISTOGRAM_H

#include "aad_data_types.h"

// Configuration
#define HIST_BIN_SIZE_MS 0.1f   
#define HIST_NUM_BINS 100       
#define HIST_COINS_NUM_BINS 20 

// API Functions
void update_time_histogram(float elapsed_ms);
void update_coin_histogram(int coins_found);
void save_gnuplot_data();

#endif