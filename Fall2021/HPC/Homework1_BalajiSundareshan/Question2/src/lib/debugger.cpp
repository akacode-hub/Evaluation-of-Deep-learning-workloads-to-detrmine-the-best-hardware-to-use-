// Created by Kaustubh Shivdikar
//
// (C) All Rights Reserved


#include "debugger.h"
#include "functions.h"

using namespace std;


void red_start () {
cout << "\e[1;31m";
}

void green_start() {
cout << "\e[1;32m";
}

void yellow_start() {
cout << "\e[1;33m";
}

void cyan_start() {
cout << "\e[1;36m";
}

void underline_start() {
cout << "\e[1;4m";
}

void color_reset() {
cout << "\e[0m";
}
