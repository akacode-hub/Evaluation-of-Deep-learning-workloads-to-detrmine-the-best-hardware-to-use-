#!/bin/bash

# Created Balaji Sundareshan -- All Rights Reserved


TARGET = sort.bin
CC = g++
CFLAGS=-Wall -O2 -std=c++0x -pthread
LDFLAGS=

ROOT_DIR=$(PWD)
CODE_DIR=$(ROOT_DIR)/src/code
HEADER_DIR=$(ROOT_DIR)/src/lib
LIB_DIR=$(ROOT_DIR)/src/lib
OBJ_DIR=$(ROOT_DIR)/src/obj

OBJ=$(CODE_DIR)/default.o $(OBJ_DIR)/functions.o $(OBJ_DIR)/debugger.o
OBJ_WITH_TARGET=$(OBJ_DIR)/default.o $(OBJ_DIR)/functions.o $(OBJ_DIR)/debugger.o



all: $(OBJ)
	$(CC) $(CFLAGS) $(LDFLAGS) -o $(ROOT_DIR)/$(TARGET) $(OBJ_WITH_TARGET)
#	cp $(BIN_DIR)/$(TARGET) $(ROOT_DIR)

$(OBJ_DIR)/%.o: $(LIB_DIR)/%.cpp
	$(CC) -I$(LIB_DIR) $(CFLAGS) -c $<
	mv $(@F) $(OBJ_DIR)

$(CODE_DIR)/%.o: $(CODE_DIR)/%.cpp
	$(CC) -I$(LIB_DIR) $(CFLAGS) -c $<
	mv $(@F) $(OBJ_DIR)


clean:
	printf "\n\e[1;32mCleaning all files\e[0m\n"
	rm -f $(OBJ_DIR)/*.o $(ROOT_DIR)/*.bin

run: all
	$(ROOT_DIR)/$(TARGET)

test:
	echo "Test Start"
