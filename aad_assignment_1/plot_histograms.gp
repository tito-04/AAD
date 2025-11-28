set terminal png size 1000,800 enhanced font 'Arial,10'
set output 'miner_histograms.png'
set multiplot layout 2,1 title 'CUDA Miner Statistics'
set title 'Kernel Execution Wall Time (Zoomed)'
set xlabel 'Time (ms)'
set ylabel 'Occurrences (Log Scale)'
set grid
set style fill solid 0.5
set logscale y
set xrange [2.90:10.00]
plot 'hist_time.dat' using 1:2 with boxes title 'Kernel Duration' lc rgb '#0060ad'
set title 'Coins Found per Kernel Launch'
set xlabel 'Number of Coins'
set ylabel 'Frequency (Log Scale)'
set grid
set style fill solid 0.5
set boxwidth 0.5
set logscale y
set xtics 1
set xrange [-0.5:2.5]
plot 'hist_coins.dat' using 1:2 with boxes title 'Coins Found' lc rgb '#228b22'
unset multiplot
