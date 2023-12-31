#!/bin/bash

# Set the path to your C program executable
program_path="./your_c_program"

# Set the number of iterations for measuring FPS
iterations=1000

# Record the start time
start_time=$(date +%s.%N)

# Run the program for the specified iterations
for ((i=1; i<=$iterations; i++)); do
    tiled_nn
done

# Record the end time
end_time=$(date +%s.%N)

# Calculate the total time
total_time=$(echo "$end_time - $start_time" | bc)

# Calculate FPS
fps=$(echo "scale=2; $iterations / $total_time" | bc)

# Print the FPS
echo "FPS: $fps"

