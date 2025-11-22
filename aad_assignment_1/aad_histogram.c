#include <stdio.h>
#include <stdlib.h>
#include "aad_histogram.h"

// Static storage for the histograms
static u64_t kernel_time_histogram[HIST_NUM_BINS] = {0};
static u64_t coins_count_histogram[HIST_COINS_NUM_BINS] = {0};

// Function to record kernel execution time
void update_time_histogram(float elapsed_ms) {
    int bin = (int)(elapsed_ms / HIST_BIN_SIZE_MS);
    if(bin >= 0 && bin < HIST_NUM_BINS) {
        kernel_time_histogram[bin]++;
    } else if (bin >= HIST_NUM_BINS) {
        kernel_time_histogram[HIST_NUM_BINS - 1]++; // Clamp to last bin
    }
}

// Function to record number of coins found in a batch
void update_coin_histogram(int coins_found) {
    if (coins_found >= HIST_COINS_NUM_BINS) {
        coins_found = HIST_COINS_NUM_BINS - 1; // Clamp to max bin
    }
    coins_count_histogram[coins_found]++;
}

// The logic to generate data files and the Gnuplot script
void save_gnuplot_data() {
    FILE *fp;

    // 1. Save Time Data & Calculate Dynamic Range
    float min_time_display = 0.0f;
    float max_time_display = (float)HIST_NUM_BINS * HIST_BIN_SIZE_MS;
    int first_idx = -1;
    int last_idx = -1;

    fp = fopen("hist_time.dat", "w");
    if (fp) {
        for(int i = 0; i < HIST_NUM_BINS; i++) {
            // Track range of data for auto-zoom
            if (kernel_time_histogram[i] > 0) {
                if (first_idx == -1) first_idx = i;
                last_idx = i;
            }
            
            float time_ms = i * HIST_BIN_SIZE_MS;
            fprintf(fp, "%.2f %llu\n", time_ms, (unsigned long long)kernel_time_histogram[i]);
        }
        fclose(fp);
    }
    
    // Calculate X-Axis Zoom (Padding: 2 bins left/right)
    if (first_idx != -1) {
        int start_bin = (first_idx > 2) ? first_idx - 2 : 0;
        int end_bin   = (last_idx < HIST_NUM_BINS - 2) ? last_idx + 2 : HIST_NUM_BINS;
        min_time_display = start_bin * HIST_BIN_SIZE_MS;
        max_time_display = end_bin * HIST_BIN_SIZE_MS;
    }

    // 2. Save Coin Data & Calculate Max Coins
    int max_coins_found = 0;
    fp = fopen("hist_coins.dat", "w");
    if (fp) {
        for(int i = 0; i < HIST_COINS_NUM_BINS; i++) {
            if (coins_count_histogram[i] > 0) max_coins_found = i;
            fprintf(fp, "%d %llu\n", i, (unsigned long long)coins_count_histogram[i]);
        }
        fclose(fp);
    }

    // 3. Generate Smart Gnuplot Script
    fp = fopen("plot_histograms.gp", "w");
    if (fp) {
        fprintf(fp, "set terminal png size 1000,800 enhanced font 'Arial,10'\n");
        fprintf(fp, "set output 'miner_histograms.png'\n");
        fprintf(fp, "set multiplot layout 2,1 title 'CUDA Miner Statistics'\n");
        
        // Plot 1: Wall Time (Zoomed & Log)
        fprintf(fp, "set title 'Kernel Execution Wall Time (Zoomed)'\n");
        fprintf(fp, "set xlabel 'Time (ms)'\n");
        fprintf(fp, "set ylabel 'Occurrences (Log Scale)'\n");
        fprintf(fp, "set grid\n");
        fprintf(fp, "set style fill solid 0.5\n");
        fprintf(fp, "set logscale y\n");  
        fprintf(fp, "set xrange [%.2f:%.2f]\n", min_time_display, max_time_display); 
        fprintf(fp, "plot 'hist_time.dat' using 1:2 with boxes title 'Kernel Duration' lc rgb '#0060ad'\n");

        // Plot 2: Coins per Run (Zoomed & Log)
        fprintf(fp, "set title 'Coins Found per Kernel Launch'\n");
        fprintf(fp, "set xlabel 'Number of Coins'\n");
        fprintf(fp, "set ylabel 'Frequency (Log Scale)'\n");
        fprintf(fp, "set grid\n");
        fprintf(fp, "set style fill solid 0.5\n");
        fprintf(fp, "set boxwidth 0.5\n");
        fprintf(fp, "set logscale y\n"); 
        fprintf(fp, "set xtics 1\n");
        fprintf(fp, "set xrange [-0.5:%.1f]\n", (float)max_coins_found + 1.5);
        fprintf(fp, "plot 'hist_coins.dat' using 1:2 with boxes title 'Coins Found' lc rgb '#228b22'\n");

        fprintf(fp, "unset multiplot\n");
        fclose(fp);
    }

    printf("\n[INFO] Histogram data saved to 'hist_time.dat' and 'hist_coins.dat'.\n");
    printf("[INFO] Gnuplot script saved to 'plot_histograms.gp' (Log-Scale & Zoom enabled).\n");
    
    int ret = system("gnuplot plot_histograms.gp");
    if (ret == 0) {
        printf("[SUCCESS] Generated chart: 'miner_histograms.png'\n");
    } else {
        printf("[TIP] Run manually: gnuplot plot_histograms.gp\n");
    }
}