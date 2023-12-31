CC := gcc
CFLAGS := -Wall -Iinclude
SRC_DIR := src
INCLUDE_DIR := include
BUILD_DIR := build

# List all source files in the src directory
SRC_FILES := $(wildcard $(SRC_DIR)/*.c)

# List all header files in the include directory
INCLUDE_FILES := $(wildcard $(INCLUDE_DIR)/*.h)


# Define the model directory variable (default to "model" if not provided)
MODEL_DIR := $(or models/$(MODEL), model)
EXECUTABLE := $(or $(MODEL), model)

# List all source files in the model directory
MODEL_SRC_FILES := $(wildcard $(MODEL_DIR)/*.c)

# List all header files in the model directory
MODEL_INCLUDE_FILES := $(wildcard $(MODEL_DIR)/*.h)

# Combine all source files
ALL_SRC_FILES := $(SRC_FILES) $(MODEL_SRC_FILES)

# Combine all header files
ALL_INCLUDE_FILES := $(INCLUDE_FILES) $(MODEL_INCLUDE_FILES)

# Generate object file names
OBJ_FILES := $(patsubst %.c,$(BUILD_DIR)/%.o,$(notdir $(ALL_SRC_FILES)))

# Define the target binary
TARGET := $(EXECUTABLE)

# Build the binary
$(TARGET): $(OBJ_FILES)
	$(CC) $(CFLAGS) $^ -o $@

# Build object files from source files
$(BUILD_DIR)/%.o: $(SRC_DIR)/%.c $(ALL_INCLUDE_FILES)
	mkdir -p $(BUILD_DIR)
	$(CC) $(CFLAGS) -c $< -o $@

# Build object files from model source files
$(BUILD_DIR)/%.o: $(MODEL_DIR)/%.c $(ALL_INCLUDE_FILES)
	mkdir -p $(BUILD_DIR)
	$(CC) $(CFLAGS) -c $< -o $@




.PHONY: clean

clean:
	rm -f $(TARGET)
	rm -rf $(BUILD_DIR) 
	rm -f $(OBJ_FILES)





#ARDUINO_BUILD := 1
#ifeq ($(ARDUINO_BUILD), 1)

#This part is for converting .c to .cpp files and moving .h and .cpp files into a single folder.  
#This makes the code compatible with Arduino
# Define source and destination directories
DEST_DIR := arduino_libs/$(MODEL_DIR)

# Define the script name and path
SCRIPT := scripts/arduino_library.sh

# Define the target to copy and rename files
copy_and_rename:
	rm -rf arduino_libs/$(MODEL_DIR)
	mkdir arduino_libs/$(MODEL_DIR)
	chmod +x ./$(SCRIPT)
	./$(SCRIPT) $(DEST_DIR) $(MODEL_DIR)

#.DEFAULT_GOAL := copy_and_rename

#endif


# Default target
all: clean $(TARGET)

# Target for building with a specific model
build-model: clean $(TARGET)

build-arduino: clean $(TARGET)  copy_and_rename